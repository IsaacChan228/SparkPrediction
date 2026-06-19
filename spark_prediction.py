"""Prediction helpers extracted from spark_training.

Provides Spark session helper, preprocessing, model class, and CSV
prediction function so prediction logic is separated from training.
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
    # On Windows, Spark/Hadoop may look for winutils.exe. If HADOOP_HOME is
    # not set the log will show a FileNotFoundException about winutils.exe.
    # Provide a clearer message and a reasonable default location to help the
    # user fix the environment.
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


def load_and_preprocess(spark: SparkSession, csv_path: str) -> pd.DataFrame:
    df = spark.read.option("header", True).csv(csv_path)

    # Basic feature engineering: votes, purchased, time, comment length
    # Sanitize numeric input before casting to avoid failures when fields contain
    # stray text (e.g. malformed CSV values inside comments). We strip any
    # non-numeric characters, validate the cleaned string matches a number,
    # and fallback to 0.0 when invalid.
    votes_clean = F.regexp_replace(F.col("votes"), "[^0-9.\\-]", "")
    votes_num = F.when(votes_clean.rlike(r"^-?\\d+(\\.\\d+)?$"), votes_clean.cast("double")).otherwise(F.lit(0.0)).alias("votes")

    time_clean = F.regexp_replace(F.col("time"), "[^0-9.\\-]", "")
    time_num = F.when(time_clean.rlike(r"^-?\\d+(\\.\\d+)?$"), time_clean.cast("double")).otherwise(F.lit(0.0)).alias("time")

    df2 = (
        df.select(
            votes_num,
            (F.when(F.upper(F.col("purchased")) == "TRUE", 1.0).otherwise(0.0)).alias("purchased"),
            time_num,
            F.length(F.col("comment")).cast("double").alias("comment_len"),
            # Only cast rating when it is a single digit 1-5; otherwise use NULL
            F.when(F.col("rating").rlike(r"^[1-5]$"), F.col("rating").cast("int")).otherwise(F.lit(None)).alias("rating"),
        )
        .na.fill({"votes": 0.0, "time": 0.0, "comment_len": 0.0, "purchased": 0.0})
    )

    pdf = df2.toPandas()
    # Drop rows without label and make an explicit copy to avoid chained-assignment warnings
    pdf = pdf.dropna(subset=["rating"]).copy()
    pdf["label"] = pdf["rating"].astype(float)
    features = pdf[["votes", "purchased", "time", "comment_len"]].astype(float)
    pdf_features = features.fillna(0.0)
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


def predict_csv(spark: SparkSession, input_csv: str, model_path: Path = MODEL_PATH, report_path: Path | None = None) -> pd.DataFrame:
    # Read input and extract features; preserve `id` if present for submission format
    df = spark.read.option("header", True).csv(input_csv)
    # Sanitize numeric fields similarly to the training preprocessing to
    # tolerate malformed string values in CSV inputs.
    votes_clean = F.regexp_replace(F.col("votes"), "[^0-9.\\-]", "")
    votes_num = F.when(votes_clean.rlike(r"^-?\\d+(\\.\\d+)?$"), votes_clean.cast("double")).otherwise(F.lit(0.0)).alias("votes")

    time_clean = F.regexp_replace(F.col("time"), "[^0-9.\\-]", "")
    time_num = F.when(time_clean.rlike(r"^-?\\d+(\\.\\d+)?$"), time_clean.cast("double")).otherwise(F.lit(0.0)).alias("time")

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
    X = pdf[["votes", "purchased", "time", "comment_len"]].astype(float).values

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
    # input/model paths may be under [paths]
    cfg["input_csv"] = cp.get("paths", "input_csv", fallback="prediction_input/test_merged.csv")
    cfg["model_path"] = cp.get("paths", "model_path", fallback=str(MODEL_PATH))
    cfg["output_csv"] = cp.get("prediction", "output_csv", fallback=None)
    cfg["app_name"] = cp.get("prediction", "app_name", fallback="spark-pytorch-predict")
    cfg["report_path"] = cp.get("paths", "report_path", fallback="training_report.txt")
    return cfg


def main():
    cfg = load_config("config.cfg")
    spark = get_spark(app_name=cfg.get("app_name"))
    preds = predict_csv(
        spark,
        cfg.get("input_csv"),
        model_path=Path(cfg.get("model_path")),
        report_path=Path(cfg.get("report_path")) if cfg.get("report_path") else None,
    )
    output = cfg.get("output_csv") or "prediction_output/prediction_result.csv"
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(output, index=False)
    print(f"Predictions written to {output}")


if __name__ == "__main__":
    main()
