"""Prediction helpers: Spark preprocessing, model I/O, and CSV inference.

This module contains utilities to create a Spark session, preprocess CSV
inputs into numeric features compatible with the training pipeline, load the
trained PyTorch MLP, and run batch inference producing a pandas DataFrame of
predicted `rating` values.
"""
from __future__ import annotations

from pathlib import Path
import os
from typing import Tuple
import configparser

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time
from torch.utils.data import Dataset
from pyspark.sql import SparkSession
import pyspark.sql.functions as F


MODEL_PATH = Path("Model/pytorch_mlp.pt")


def get_spark(app_name: str = "spark-pytorch-mlp") -> SparkSession:
    """Create and return a SparkSession, with Windows/Hadoop helpers.

    On Windows Spark may require a working `winutils.exe`. The function
    attempts to detect a sensible `C:\hadoop\bin\winutils.exe` location and
    sets `HADOOP_HOME`/`PATH` for the current process to reduce confusing
    startup errors. It does not modify the global environment permanently.
    """
    if os.name == "nt":
        # If the user placed a compiled Hadoop + winutils in C:\hadoop, prefer it
        candidate = Path("C:/hadoop")
        candidate_winutils = candidate / "bin" / "winutils.exe"
        if candidate_winutils.exists():
            os.environ.setdefault("HADOOP_HOME", str(candidate))
            # ensure the bin folder is on PATH for the current process
            binpath = str(candidate / "bin")
            path_env = os.environ.get("PATH", "")
            if binpath not in path_env.split(os.pathsep):
                os.environ["PATH"] = path_env + os.pathsep + binpath if path_env else binpath
            print(f"INFO: detected winutils.exe at {candidate_winutils}; set HADOOP_HOME={candidate} and added {binpath} to PATH")
        else:
            # fallback behavior: use HADOOP_HOME if set, else set a sensible default
            hadoop_home = os.environ.get("HADOOP_HOME")
            default_h = candidate
            if not hadoop_home:
                os.environ.setdefault("HADOOP_HOME", str(default_h))
                hadoop_home = str(default_h)

            winutils_path = Path(hadoop_home) / "bin" / "winutils.exe"
            if not winutils_path.exists():
                print(
                    "WARNING: winutils.exe not found at {0}.\n"
                    "To silence this message and avoid Hadoop/Windows issues,\n"
                    "download a matching winutils.exe for your Hadoop version and\n"
                    "place it in C:\\hadoop\\bin, then set the HADOOP_HOME\n"
                    "environment variable to C:\\hadoop. Example (PowerShell):\n\n"
                    "  mkdir C:\\hadoop\\bin -Force\n"
                    "  # copy winutils.exe into C:\\hadoop\\bin\\winutils.exe\n"
                    "  setx HADOOP_HOME 'C:\\hadoop' -m\n"
                    "  setx PATH ($env:PATH + ';C:\\hadoop\\bin') -m\n",
                    str(winutils_path),
                )

    return SparkSession.builder.appName(app_name).getOrCreate()


def load_and_preprocess(
    spark: SparkSession,
    csv_path: str,
    bert_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Load CSV via Spark, perform robust sanitization, and return a pandas
    DataFrame of numeric features + `label` (when present).

    The function uses permissive CSV options to allow multiline quoted fields
    and then applies column-level cleaning for `votes`, `time`, `purchased`,
    and `comment_len`. Precomputed embedding columns (``<col>_emb_<i>``) are
    detected later in the pandas frame and included automatically.
    """

    # Treat common textual null markers like 'NA' and empty strings as nulls
    # Allow quoted fields with embedded newlines and double-quote escaping
    df = (
        spark.read.option("header", True)
        .option("nullValue", "NA")
        .option("treatEmptyValuesAsNulls", "true")
        .option("encoding", "UTF-8")
        .option("sep", ",")
        .option("quote", '"')
        # CSV standard escapes quotes by doubling them (""), so set escape to '"'
        .option("escape", '"')
        # allow multiline fields so comments containing newlines are parsed as single field
        .option("multiLine", "true")
        .option("ignoreLeadingWhiteSpace", "true")
        .option("ignoreTrailingWhiteSpace", "true")
        .option("mode", "PERMISSIVE")
        .csv(csv_path)
    )


    # If bert columns are not present in the merged/preprocessed data, they
    # will be handled later. Precomputed embedding columns (e.g. "comment_emb_0")
    # will be detected and used when building the numeric feature matrix.
    if bert_cols is None:
        bert_cols = ["comment", "title", "prod_title", "prod_features"]

    # Basic feature engineering: votes, purchased, time, comment length
    # Sanitize numeric input before casting to avoid failures when fields contain
    # stray text (e.g. malformed CSV values inside comments). We strip any
    # non-numeric characters, validate the cleaned string matches a number,
    # and fallback to 0.0 when invalid.
    votes_clean = F.regexp_replace(F.col("votes"), "[^0-9.\\-]", "")
    votes_num = F.when(votes_clean.rlike(r"^-?\d+(\.\d+)?$"), votes_clean.cast("double")).otherwise(F.lit(0.0)).alias("votes")

    time_clean = F.regexp_replace(F.col("time"), "[^0-9.\\-]", "")
    time_num = F.when(time_clean.rlike(r"^-?\d+(\.\d+)?$"), time_clean.cast("double")).otherwise(F.lit(0.0)).alias("time")

    df2 = (
        df.select(
            votes_num,
            (F.when(F.upper(F.col("purchased")) == "TRUE", 1.0).otherwise(0.0)).alias("purchased"),
            time_num,
            F.length(F.col("comment")).cast("double").alias("comment_len"),
            F.when(
                F.regexp_replace(F.col("rating"), "[^0-9.]", "").rlike(r"^[1-5](\.0+)?$"),
                F.regexp_replace(F.col("rating"), "[^0-9.]", "").cast("double"),
            ).otherwise(F.lit(None)).alias("rating"),
        )
        .na.fill({"votes": 0.0, "time": 0.0, "comment_len": 0.0, "purchased": 0.0})
    )

    pdf = df2.toPandas()
    # Note: any precomputed embedding columns with names like
    # '<col>_emb_<i>' will be detected and included in the numeric features
    # below; no on-the-fly BERT encoding is performed here.
    # Drop rows without label and make an explicit copy to avoid chained-assignment warnings
    pdf = pdf.dropna(subset=["rating"]).copy()
    pdf["label"] = pdf["rating"].astype(float)
    # Build numeric feature frame. If embedding columns exist they will be
    # included in the order of `bert_cols`; otherwise fall back to
    # `comment_len`.
    # collect embedding columns if present
    def _collect_emb_cols(df: pd.DataFrame, bert_cols_list: list[str]) -> list[str]:
        emb_cols: list[str] = []
        for c in bert_cols_list:
            prefix = f"{c}_emb_"
            matches = [col for col in df.columns if col.startswith(prefix)]
            if matches:
                # sort by trailing index
                try:
                    matches = sorted(matches, key=lambda x: int(x.rsplit("_", 1)[1]))
                except Exception:
                    matches = sorted(matches)
                emb_cols.extend(matches)
        return emb_cols

    features = pdf[["votes", "purchased", "time"]].astype(float)
    pdf_features = features.fillna(0.0)
    emb_columns = _collect_emb_cols(pdf, bert_cols)
    if emb_columns:
        # ensure embedding columns are numeric and present
        pdf_embs = pdf[emb_columns].astype(float).fillna(0.0)
        pdf_features = pd.concat([pdf_features, pdf_embs], axis=1)
    else:
        pdf_features = pd.concat([pdf_features, pdf[["comment_len"]].astype(float).fillna(0.0)], axis=1)
    # If BERT embeddings were attached and the caller requested to use them
    # (handled by training/prediction code), those columns will be present in
    # `pdf` and can be used when constructing feature matrices.
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

    The function expects preprocessed numeric columns or precomputed embedding
    columns named like ``<col>_emb_<i>`` to be present in the CSV. No on-the-
    fly BERT encoding is performed.
    """

    # Read input and extract features; preserve `id` if present for submission format
    # Treat common textual null markers like 'NA' and empty strings as nulls
    # Use same robust CSV options for prediction inputs
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
    # Sanitize numeric fields similarly to the training preprocessing to
    # tolerate malformed string values in CSV inputs.
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
            F.length(F.col("comment")).cast("double").alias("comment_len"),
        )
        .na.fill({"votes": 0.0, "time": 0.0, "comment_len": 0.0, "purchased": 0.0})
    )

    pdf = df2.toPandas()
    ids = pdf.get("id")

    # Precomputed embedding columns (named like '<col>_emb_<i>') will be used
    # when constructing the numeric feature matrix. No on-the-fly BERT
    # encoding is performed during prediction.
    if bert_cols is None:
        bert_cols = ["comment", "title", "prod_title", "prod_features"]

    # Build feature list by including any numeric embedding columns that were
    # precomputed during preprocessing. Embedding dims are detected by column
    # name pattern '<col>_emb_<i>' and added in bert_cols order.
    feature_cols = ["votes", "purchased", "time"]
    import re

    def _collect_emb_cols(df: pd.DataFrame, bert_cols_list: list[str]) -> list[str]:
        emb_cols: list[str] = []
        for c in bert_cols_list:
            prefix = f"{c}_emb_"
            matches = [col for col in df.columns if col.startswith(prefix)]
            if matches:
                try:
                    matches = sorted(matches, key=lambda x: int(x.rsplit("_", 1)[1]))
                except Exception:
                    matches = sorted(matches)
                emb_cols.extend(matches)
        return emb_cols

    emb_cols = _collect_emb_cols(pdf, bert_cols)
    if emb_cols:
        feature_cols.extend(emb_cols)
    else:
        feature_cols.append("comment_len")

    X = pdf[feature_cols].astype(float).values

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_dim=X.shape[1])

    # load model; saved object may contain scaler and model_state
    loaded = torch.load(str(model_path), map_location=device)
    scaler = None
    if isinstance(loaded, dict) and "model_state" in loaded:
        state = loaded["model_state"]
        scaler = loaded.get("scaler")
        model.load_state_dict(state)
    else:
        # legacy: state_dict only
        model.load_state_dict(loaded)

    model.to(device)
    model.eval()

    # apply scaler if available
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

    # Round/clamp predictions to integer ratings 1-5
    ratings = np.rint(preds).astype(int)
    ratings = np.clip(ratings, 1, 5)

    if ids is not None:
        out = pd.DataFrame({"id": ids.astype(int), "rating": ratings})
    else:
        out = pd.DataFrame({"id": np.arange(len(ratings)), "rating": ratings})

    # Print total and average inference time (do not modify output CSV schema)
    total = float(elapsed)
    avg = total / max(1, len(ratings))
    print(f"Inference total time: {total:.6f}s, average per record: {avg:.6f}s")

    # If a training report exists, append the inference timing to it so both
    # training and inference times appear in the same report.
    if report_path is not None:
        rp = Path(report_path)
        if rp.exists():
            try:
                with rp.open("a", encoding="utf-8") as fh:
                    fh.write(f"\nInference: {len(ratings)} records, total_time_s={total:.6f}, avg_time_s={avg:.6f}\n")
            except Exception:
                pass

    return out



def load_config(path: str | Path) -> dict:
    cp = configparser.ConfigParser()
    cp.read(path)
    cfg = {}
    cfg["input_csv"] = cp.get("paths", "input_csv", fallback="prediction_input/test_merged.csv")
    cfg["model_path"] = cp.get("paths", "model_path", fallback=str(MODEL_PATH))
    cfg["report_path"] = cp.get("paths", "report_path", fallback=None)
    cfg["output_csv"] = cp.get("paths", "output_csv", fallback="prediction_output/prediction_result.csv")
    # feature flags
    cfg["bert_model"] = cp.get("features", "bert_model", fallback="all-MiniLM-L6-v2")
    return cfg


def main():
    cfg = load_config("config.cfg")

    spark = get_spark()

    preds = predict_csv(
        spark,
        cfg["input_csv"],
        model_path=Path(cfg["model_path"]),
        report_path=Path(cfg["report_path"]) if cfg.get("report_path") else None,
    )

    output = cfg.get("output_csv") or "prediction_output/prediction_result.csv"
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(output, index=False)
    print(f"Predictions written to {output}")


if __name__ == "__main__":
    main()
