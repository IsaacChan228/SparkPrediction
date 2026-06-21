"""Prediction helpers: Spark preprocessing, model I/O, and CSV inference.

This module contains utilities to create a Spark session, preprocess CSV
inputs into numeric features compatible with the training pipeline, load the
trained PyTorch MLP, and run batch inference producing a pandas DataFrame of
predicted `rating` values.
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
    driver_mem = os.environ.get("SPARK_DRIVER_MEMORY", "16g")
    shuffle_parts = os.environ.get("SPARK_SQL_SHUFFLE_PARTITIONS", "32")
    max_result = os.environ.get("SPARK_DRIVER_MAX_RESULT_SIZE", "8g")

    builder = (
        SparkSession.builder.appName(app_name)
        .master(os.environ.get("SPARK_MASTER", "local[12]"))
        .config("spark.driver.memory", driver_mem)
        .config("spark.driver.maxResultSize", max_result)
        .config("spark.sql.shuffle.partitions", shuffle_parts)
        .config("spark.sql.debug.maxToStringFields", os.environ.get("SPARK_SQL_DEBUG_MAX_FIELDS", "1000"))
    )

    return builder.getOrCreate()


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

    batch_size = int(os.environ.get("SPARK_TO_PANDAS_BATCH_SIZE", "5000"))
    sel_cols = list(df2.columns) + emb_cols + prod_cols

    parts: list[pd.DataFrame] = []
    buffer: list[dict] = []
    it = df.select(*sel_cols).toLocalIterator()
    for row in it:
        buffer.append(row.asDict())
        if len(buffer) >= batch_size:
            part = pd.DataFrame(buffer)
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
            parts.append(part)
            buffer = []

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
        parts.append(part)

    if parts:
        pdf = pd.concat(parts, ignore_index=True)
    else:
        pdf = pd.DataFrame(columns=sel_cols)

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
    def __init__(self, input_dim: int, hidden1: int = 128, hidden2: int = 64, out: int = 1, dropout: float = 0.2):
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
            nn.Linear(hidden2, out),
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

    pdf = df2.toPandas()
    ids = pdf.get("id")

    # materialize only embedding/product columns from the raw frame
    emb_cols = [c for c in df.columns if re.search(r"_emb_\d+$", c)]
    prod_cols = [c for c in ("prod_main_category", "prod_price", "prod_store", "prod_rating_number") if c in df.columns]
    if emb_cols or prod_cols:
        small = df.select(*(emb_cols + prod_cols)).toPandas()
        for c in emb_cols:
            if c in small.columns:
                pdf[c] = pd.to_numeric(small[c], errors="coerce").fillna(0.0)
        if "prod_price" in small.columns:
            pdf["prod_price"] = pd.to_numeric(small["prod_price"].astype(str).str.replace(r"[^0-9.\\-]", "", regex=True), errors="coerce").fillna(0.0)
        if "prod_rating_number" in small.columns:
            pdf["prod_rating_number"] = pd.to_numeric(small["prod_rating_number"].astype(str).str.replace(r"[^0-9.\\-]", "", regex=True), errors="coerce").fillna(0.0)
        if "prod_main_category" in small.columns:
            pdf["prod_main_category"] = pd.Categorical(small["prod_main_category"].fillna("")).codes.astype(float)
        if "prod_store" in small.columns:
            pdf["prod_store"] = pd.Categorical(small["prod_store"].fillna("")).codes.astype(float)

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

    ratings = np.rint(preds).astype(int)
    ratings = np.clip(ratings, 1, 5)

    if ids is not None:
        out = pd.DataFrame({"id": ids.astype(int), "rating": ratings})
    else:
        out = pd.DataFrame({"id": np.arange(len(ratings)), "rating": ratings})

    total = float(elapsed)
    avg = total / max(1, len(ratings))
    print(f"Inference total time: {total:.6f}s, average per record: {avg:.6f}s")

    if report_path is not None:
        rp = Path(report_path)
        if rp.exists():
            try:
                with rp.open("a", encoding="utf-8") as fh:
                    fh.write(f"\nInference: {len(ratings)} records, total_time_s={total:.6f}, avg_time_s={avg:.6f}\n")
            except Exception:
                pass

    return out
