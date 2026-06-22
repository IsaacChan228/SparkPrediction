"""Prediction helpers: Spark preprocessing, model I/O, and CSV inference.

Utilities to create a Spark session, preprocess CSV inputs into numeric
features compatible with the training pipeline, load the trained PyTorch
MLP, and run batch inference producing a pandas DataFrame of predicted
`rating` values.

Notes:
- `predict_csv()` returns floating-point predictions in the `rating` column.
- `main()` writes two CSVs to `prediction_output/`: a float predictions file
    (`predictions_float.csv`) and a rounded-and-capped integer file
    (`predictions_rounded.csv`).
- This module focuses on preprocessing and inference.
"""
from __future__ import annotations

from pathlib import Path
import os
import configparser
import re
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import tempfile
import shutil

MODEL_PATH = Path("Model/pytorch_mlp.pt")

# Enforce these text columns must be represented by embeddings
ALLOWED_BERT_COLS = ["comment", "title", "prod_title", "prod_features"]


def get_spark(app_name: str = "spark-pytorch-mlp") -> SparkSession:
    """Create and return a SparkSession, with Windows/Hadoop helpers.

    Uses sensible defaults for driver memory and shuffle partitions which can
    be tuned via environment variables to avoid Java heap OOM on large runs.
    """
    
    if os.name == "nt":
        candidate = Path("C:/hadoop")
        candidate_winutils = candidate / "bin" / "winutils.exe"
        if candidate_winutils.exists():
            os.environ.setdefault("HADOOP_HOME", str(candidate))
            binpath = str(candidate / "bin")
            path_env = os.environ.get("PATH", "")
            if binpath not in path_env.split(os.pathsep):
                os.environ["PATH"] = path_env + os.pathsep + binpath if path_env else binpath
                print(f"INFO: detected winutils.exe at {candidate_winutils}; set HADOOP_HOME={candidate} and added {binpath} to PATH")

    # Tuned defaults for a 16-core / 32GB machine; can be overridden via env
    driver_mem = os.environ.get("SPARK_DRIVER_MEMORY", "30g")
    exec_mem = os.environ.get("SPARK_EXECUTOR_MEMORY", driver_mem)
    shuffle_parts = os.environ.get("SPARK_SQL_SHUFFLE_PARTITIONS", "16")
    max_result = os.environ.get("SPARK_DRIVER_MAX_RESULT_SIZE", "8g")

    builder = (
        SparkSession.builder.appName(app_name)
        .master(os.environ.get("SPARK_MASTER", "local[12]"))
        .config("spark.driver.memory", driver_mem)
        .config("spark.executor.memory", exec_mem)
        .config("spark.driver.maxResultSize", max_result)
        .config("spark.sql.shuffle.partitions", shuffle_parts)
        .config("spark.sql.debug.maxToStringFields", os.environ.get("SPARK_SQL_DEBUG_MAX_FIELDS", "1000"))
    )

    spark = builder.getOrCreate()
    # no runtime snapshot here; print configured Spark settings below
    try:
        configured = spark.conf.get("spark.driver.memory")
    except Exception:
        # fallback to SparkContext conf
        try:
            configured = spark.sparkContext.getConf().get("spark.driver.memory")
        except Exception:
            configured = os.environ.get("SPARK_DRIVER_MEMORY", "unknown")
    # Also print other relevant Spark memory settings to help diagnose JVM OOMs
    try:
        exec_mem = spark.conf.get("spark.executor.memory")
    except Exception:
        try:
            exec_mem = spark.sparkContext.getConf().get("spark.executor.memory")
        except Exception:
            exec_mem = os.environ.get("SPARK_EXECUTOR_MEMORY", "unset")

    try:
        max_res = spark.conf.get("spark.driver.maxResultSize")
    except Exception:
        try:
            max_res = spark.sparkContext.getConf().get("spark.driver.maxResultSize")
        except Exception:
            max_res = os.environ.get("SPARK_DRIVER_MAX_RESULT_SIZE", "unset")

    try:
        shuffle_parts = spark.conf.get("spark.sql.shuffle.partitions")
    except Exception:
        try:
            shuffle_parts = spark.sparkContext.getConf().get("spark.sql.shuffle.partitions")
        except Exception:
            shuffle_parts = os.environ.get("SPARK_SQL_SHUFFLE_PARTITIONS", "unset")

    print(f"Configured Spark driver memory: {configured}")
    print(f"Configured Spark executor memory: {exec_mem}")
    print(f"Configured spark.driver.maxResultSize: {max_res}")
    print(f"Configured spark.sql.shuffle.partitions: {shuffle_parts}")
    return spark


def load_and_preprocess(
    spark: SparkSession,
    csv_path: str,
    bert_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Load CSV via Spark, sanitize columns, and return a numeric pandas frame.

    This implementation streams selected columns from Spark into pandas in
    batches to avoid a single large `toPandas()` call that can exhaust the
    JVM heap on wide or large CSVs. The final returned frame contains numeric
    feature columns and a `label` column for training.
    """

    df = (
        spark.read.option("header", True)
        .option("nullValue", "NA")
        .option("treatEmptyValuesAsNulls", "true")
        .option("encoding", "UTF-8")
        .option("sep", ",")
        .option("quote", '"')
        .option("escape", '"')
        .option("multiLine", "true")
        .option("ignoreLeadingWhiteSpace", "true")
        .option("ignoreTrailingWhiteSpace", "true")
        .option("mode", "PERMISSIVE")
        .csv(csv_path)
    )

    if bert_cols is None:
        bert_cols = ALLOWED_BERT_COLS

    # sanitize numeric-ish fields with Spark expressions, producing df2
    votes_clean = F.regexp_replace(F.col("votes"), "[^0-9.\\-]", "")
    votes_num = F.when(votes_clean.rlike(r"^-?\d+(\.\d+)?$"), votes_clean.cast("double")).otherwise(F.lit(0.0)).alias("votes")
    time_clean = F.regexp_replace(F.col("time"), "[^0-9.\\-]", "")
    time_num = F.when(time_clean.rlike(r"^-?\d+(\.\d+)?$"), time_clean.cast("double")).otherwise(F.lit(0.0)).alias("time")

    df2 = (
        df.select(
            votes_num,
            (F.when(F.upper(F.col("purchased")) == "TRUE", 1.0).otherwise(0.0)).alias("purchased"),
            time_num,
            F.when(
                F.regexp_replace(F.col("rating"), "[^0-9.]", "").rlike(r"^[1-5](\\.0+)?$"),
                F.regexp_replace(F.col("rating"), "[^0-9.]", "").cast("double"),
            ).otherwise(F.lit(None)).alias("rating"),
        )
        .na.fill({"votes": 0.0, "time": 0.0, "purchased": 0.0})
    )

    # detect embedding/product columns in the raw frame
    emb_cols = [c for c in df.columns if re.search(r"_emb_\d+$", c)]
    prod_cols = [c for c in ("prod_price", "prod_rating_number", "prod_main_category", "prod_store") if c in df.columns]

    batch_size = int(os.environ.get("SPARK_TO_PANDAS_BATCH_SIZE", "1000"))
    sel_cols = list(df2.columns) + emb_cols + prod_cols

    buffer: list[dict] = []
    it = df.select(*sel_cols).toLocalIterator()
    # prepare temporary file to stream processed batches to disk
    tmpdir = tempfile.mkdtemp(prefix="spark_parts_")
    merged_path = os.path.join(tmpdir, "merged_parts.csv")
    first_write = True
    # stream rows in batches from the JVM and write processed batches to disk
    for row in it:
        buffer.append(row.asDict())
        if len(buffer) >= batch_size:
            try:
                try:
                    print(f"\rProcessing batch: buffered_rows={len(buffer)}", end="", flush=True)
                except Exception:
                    # fallback to plain print if terminal doesn't support carriage returns
                    print(f"Processing batch: buffered_rows={len(buffer)}")
                part = pd.DataFrame(buffer)
            except Exception:
                # propagate exception; no runtime diagnostic capture
                raise
            # coerce types for embeddings and product cols
            for c in emb_cols:
                if c in part.columns:
                    part[c] = pd.to_numeric(part[c], errors="coerce").fillna(0.0)
            if "prod_price" in part.columns:
                part["prod_price"] = pd.to_numeric(part["prod_price"].astype(str).str.replace(r"[^0-9.\\-]", "", regex=True), errors="coerce").fillna(0.0)
            if "prod_rating_number" in part.columns:
                part["prod_rating_number"] = pd.to_numeric(part["prod_rating_number"].astype(str).str.replace(r"[^0-9.\\-]", "", regex=True), errors="coerce").fillna(0.0)
            if "prod_main_category" in part.columns:
                part["prod_main_category"] = pd.Categorical(part["prod_main_category"].fillna("")).codes.astype(float)
            if "prod_store" in part.columns:
                part["prod_store"] = pd.Categorical(part["prod_store"].fillna("")).codes.astype(float)

            # append processed part to merged CSV on disk to avoid accumulating in memory
            try:
                part.to_csv(merged_path, mode="a", header=first_write, index=False)
                first_write = False
            except Exception:
                # propagate exception; no runtime diagnostic capture
                raise

            # clear buffer and free part
            buffer = []
            del part

    if buffer:
        part = pd.DataFrame(buffer)
        for c in emb_cols:
            if c in part.columns:
                part[c] = pd.to_numeric(part[c], errors="coerce").fillna(0.0)
        if "prod_price" in part.columns:
            part["prod_price"] = pd.to_numeric(part["prod_price"].astype(str).str.replace(r"[^0-9.\\-]", "", regex=True), errors="coerce").fillna(0.0)
        if "prod_rating_number" in part.columns:
            part["prod_rating_number"] = pd.to_numeric(part["prod_rating_number"].astype(str).str.replace(r"[^0-9.\\-]", "", regex=True), errors="coerce").fillna(0.0)
        if "prod_main_category" in part.columns:
            part["prod_main_category"] = pd.Categorical(part["prod_main_category"].fillna("")).codes.astype(float)
        if "prod_store" in part.columns:
            part["prod_store"] = pd.Categorical(part["prod_store"].fillna("")).codes.astype(float)
        try:
            part.to_csv(merged_path, mode="a", header=first_write, index=False)
            first_write = False
        except Exception:
            # propagate exception; no runtime diagnostic capture
            raise
        del part

    # finish progress line and move to next line
    try:
        print("", flush=True)
    except Exception:
        pass

    # If we wrote a merged CSV to disk, read it back efficiently with explicit dtypes
    try:
        if os.path.exists(merged_path) and os.path.getsize(merged_path) > 0:
            # build dtype map for numeric columns to use float32
            dtype_map = {c: "float32" for c in sel_cols}
            pdf = pd.read_csv(merged_path, dtype=dtype_map)
        else:
            pdf = pd.DataFrame(columns=sel_cols)
    finally:
        # cleanup temporary directory
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass

    # validate rating presence
    if "rating" not in pdf.columns:
        raise RuntimeError("Input CSV missing 'rating' column required for training")
    n_missing = int(pdf["rating"].isnull().sum())
    if n_missing > 0:
        raise ValueError(f"Found {n_missing} rows with missing or invalid 'rating' — training requires ratings for all rows")

    pdf = pdf.copy()
    pdf["label"] = pdf["rating"].astype(float)

    def _collect_emb_cols(df: pd.DataFrame, bert_cols_list: list[str]) -> list[str]:
        emb_cols_: list[str] = []
        for c in bert_cols_list:
            prefix = f"{c}_emb_"
            matches = [col for col in df.columns if col.startswith(prefix)]
            if matches:
                try:
                    matches = sorted(matches, key=lambda x: int(x.rsplit("_", 1)[1]))
                except Exception:
                    matches = sorted(matches)
                emb_cols_.extend(matches)
        return emb_cols_

    feat_cols = ["votes", "purchased", "time"]
    for pc in ["prod_price", "prod_rating_number", "prod_main_category", "prod_store"]:
        if pc in pdf.columns:
            feat_cols.append(pc)
    features = pdf[feat_cols].astype(float)
    pdf_features = features.fillna(0.0)

    emb_columns = _collect_emb_cols(pdf, bert_cols)
    missing = [c for c in ALLOWED_BERT_COLS if not any(col.startswith(f"{c}_emb_") for col in pdf.columns)]
    if missing:
        raise RuntimeError(f"Missing required embedding columns for: {missing}. Run preprocessing (data_merging) to generate embeddings.")

    pdf_embs = pdf[emb_columns].astype(float).fillna(0.0)
    pdf_features = pd.concat([pdf_features, pdf_embs], axis=1)

    pdf_final = pd.concat([pdf_features, pdf[["label"]].astype(float)], axis=1)
    return pdf_final


class TabularDataset(Dataset):
    def __init__(self, arr: np.ndarray, labels: np.ndarray):
        self.x = torch.from_numpy(arr).float()
        self.y = torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class MLP(nn.Module):
    """Feed-forward MLP used for training and inference.

    Architecture: four hidden layers by default (`hidden1`..`hidden4`), each
    followed by BatchNorm, ReLU and Dropout. The final layer projects to the
    single output value.
    """
    def __init__(self, input_dim: int, hidden1: int = 128, hidden2: int = 64, hidden3: int = 32, hidden4: int = 16, out: int = 1, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, hidden3),
            nn.BatchNorm1d(hidden3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden3, hidden4),
            nn.BatchNorm1d(hidden4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden4, out),
        )

    def forward(self, x):
        return self.net(x)


def predict_csv(
    spark: SparkSession,
    input_csv: str,
    model_path: Path = MODEL_PATH,
    report_path: Path | None = None,
    bert_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Run inference on `input_csv` using the trained model and return a
    pandas DataFrame with `id` and predicted `rating`.
    """

    df = (
        spark.read.option("header", True)
        .option("nullValue", "NA")
        .option("treatEmptyValuesAsNulls", "true")
        .option("encoding", "UTF-8")
        .option("sep", ",")
        .option("quote", '"')
        .option("escape", '"')
        .option("multiLine", "true")
        .option("ignoreLeadingWhiteSpace", "true")
        .option("ignoreTrailingWhiteSpace", "true")
        .option("mode", "PERMISSIVE")
        .csv(input_csv)
    )

    votes_clean = F.regexp_replace(F.col("votes"), "[^0-9.\\-]", "")
    votes_num = F.when(votes_clean.rlike(r"^-?\d+(\.\d+)?$"), votes_clean.cast("double")).otherwise(F.lit(0.0)).alias("votes")
    time_clean = F.regexp_replace(F.col("time"), "[^0-9.\\-]", "")
    time_num = F.when(time_clean.rlike(r"^-?\d+(\.\d+)?$"), time_clean.cast("double")).otherwise(F.lit(0.0)).alias("time")

    df2 = (
        df.select(
            F.col("id").alias("id"),
            votes_num,
            (F.when(F.upper(F.col("purchased")) == "TRUE", 1.0).otherwise(0.0)).alias("purchased"),
            time_num,
        )
        .na.fill({"votes": 0.0, "time": 0.0, "purchased": 0.0})
    )

    # materialize only the small numeric core to pandas (select exact columns)
    core_cols = [c for c in ("id", "votes", "purchased", "time") if c in df2.columns]
    pdf = df2.select(*core_cols).toPandas()
    ids = pdf.get("id")

    # materialize only embedding/product columns from the raw frame
    emb_cols = [c for c in df.columns if re.search(r"_emb_\d+$", c)]
    prod_cols = [c for c in ("prod_main_category", "prod_price", "prod_store", "prod_rating_number") if c in df.columns]
    if emb_cols or prod_cols:
        small = df.select(*(emb_cols + prod_cols)).toPandas()
        # Build new columns in a separate DataFrame then concat once
        small = small.reset_index(drop=True)
        pdf = pdf.reset_index(drop=True)
        add_cols: dict = {}
        for c in emb_cols:
            if c in small.columns:
                add_cols[c] = pd.to_numeric(small[c], errors="coerce").fillna(0.0)
        if "prod_price" in small.columns:
            add_cols["prod_price"] = pd.to_numeric(small["prod_price"].astype(str).str.replace(r"[^0-9.\\-]", "", regex=True), errors="coerce").fillna(0.0)
        if "prod_rating_number" in small.columns:
            add_cols["prod_rating_number"] = pd.to_numeric(small["prod_rating_number"].astype(str).str.replace(r"[^0-9.\\-]", "", regex=True), errors="coerce").fillna(0.0)
        if "prod_main_category" in small.columns:
            add_cols["prod_main_category"] = pd.Categorical(small["prod_main_category"].fillna("")).codes.astype(float)
        if "prod_store" in small.columns:
            add_cols["prod_store"] = pd.Categorical(small["prod_store"].fillna("")).codes.astype(float)
        if add_cols:
            add_df = pd.DataFrame(add_cols)
            pdf = pd.concat([pdf, add_df], axis=1)
        # Defragment the DataFrame for better performance
        try:
            pdf = pdf.copy()
        except Exception:
            pass

    if bert_cols is None:
        bert_cols = ALLOWED_BERT_COLS

    feature_cols = ["votes", "purchased", "time"]
    for pc in ["prod_price", "prod_rating_number", "prod_main_category", "prod_store"]:
        if pc in pdf.columns:
            feature_cols.append(pc)

    def _collect_emb_cols_local(df_local: pd.DataFrame, bert_cols_list: list[str]) -> list[str]:
        emb_cols_local: list[str] = []
        for c in bert_cols_list:
            prefix = f"{c}_emb_"
            matches = [col for col in df_local.columns if col.startswith(prefix)]
            if matches:
                try:
                    matches = sorted(matches, key=lambda x: int(x.rsplit("_", 1)[1]))
                except Exception:
                    matches = sorted(matches)
                emb_cols_local.extend(matches)
        return emb_cols_local

    emb_cols_local = _collect_emb_cols_local(pdf, bert_cols)
    missing = [c for c in ALLOWED_BERT_COLS if not any(col.startswith(f"{c}_emb_") for col in pdf.columns)]
    if missing:
        raise RuntimeError(f"Missing required embedding columns for: {missing}. Run preprocessing (data_merging) to generate embeddings.")

    feature_cols.extend(emb_cols_local)
    X = pdf[feature_cols].astype(float).values

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_dim=X.shape[1])

    loaded = torch.load(str(model_path), map_location=device)
    scaler = None
    if isinstance(loaded, dict) and "model_state" in loaded:
        state = loaded["model_state"]
        scaler = loaded.get("scaler")
        model.load_state_dict(state)
    else:
        model.load_state_dict(loaded)

    model.to(device)
    model.eval()

    if scaler is not None:
        mean = np.array(scaler.get("mean", 0.0))
        std = np.array(scaler.get("std", 1.0))
        std[std == 0] = 1.0
        X = (X - mean) / std

    with torch.no_grad():
        xb = torch.from_numpy(X).float().to(device)
        start = time.perf_counter()
        preds = model(xb).squeeze(1).cpu().numpy()
        elapsed = time.perf_counter() - start
        preds = np.clip(preds, 1.0, 5.0)

    # Keep predicted ratings as floats for better loss calculation downstream
    preds_float = preds.astype(float)

    if ids is not None:
        out = pd.DataFrame({"id": ids.astype(int), "rating": preds_float})
    else:
        out = pd.DataFrame({"id": np.arange(len(preds_float)), "rating": preds_float})

    total = float(elapsed)
    avg = total / max(1, len(preds_float))
    print(f"Inference total time: {total:.6f}s, average per record: {avg:.6f}s")

    if report_path is not None:
        rp = Path(report_path)
        try:
            rp.parent.mkdir(parents=True, exist_ok=True)
            # If the report exists, remove any existing Inference lines so
            # there's always only a single inference-time row.
            if rp.exists():
                try:
                    text = rp.read_text(encoding="utf-8")
                    lines = text.splitlines()
                    filtered = [ln for ln in lines if not ln.strip().startswith("Inference:")]
                except Exception:
                    filtered = []
            else:
                filtered = []

            filtered.append(f"Inference: {len(preds_float)} records, total_time_s={total:.6f}, avg_time_s={avg:.6f}")
            rp.write_text("\n".join(filtered) + "\n", encoding="utf-8")
        except Exception:
            pass

    return out


def main():
    import os
    cp = configparser.ConfigParser()
    cp.read("config.cfg")

    input_csv = cp.get("paths", "input_csv", fallback="prediction_input/test_merged.csv")
    model_path = Path(cp.get("paths", "model_path", fallback=str(MODEL_PATH)))
    report_path = cp.get("paths", "report_path", fallback=None)

    spark = get_spark()
    print("Spark session started")

    print(f"Running prediction on: {input_csv}")
    out = predict_csv(spark, input_csv, model_path=Path(model_path), report_path=report_path)

    out_dir = Path("prediction_output")
    out_dir.mkdir(parents=True, exist_ok=True)

    float_path = out_dir / "predictions_float.csv"
    rounded_path = out_dir / "predictions_int.csv"

    # write float predictions
    out.to_csv(float_path, index=False)

    # create rounded, capped integer predictions
    try:
        rounded = out.copy()
        rounded["rating"] = np.rint(rounded["rating"].astype(float)).astype(int)
        rounded["rating"] = np.clip(rounded["rating"], 1, 5)
        rounded.to_csv(rounded_path, index=False)
    except Exception:
        # fallback: still write float file if rounding fails
        rounded_path = None

    print(f"Wrote float predictions to {float_path}")
    if rounded_path:
        print(f"Wrote rounded predictions to {rounded_path}")


if __name__ == "__main__":
    main()
