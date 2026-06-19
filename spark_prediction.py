"""Prediction helpers extracted from spark_training.

Provides Spark session helper, preprocessing, model class, and CSV
prediction function so prediction logic is separated from training.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple
import configparser

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from pyspark.sql import SparkSession
import pyspark.sql.functions as F


MODEL_PATH = Path("Model/pytorch_mlp.pt")


def get_spark(app_name: str = "spark-pytorch-mlp") -> SparkSession:
    return SparkSession.builder.appName(app_name).getOrCreate()


def load_and_preprocess(spark: SparkSession, csv_path: str) -> pd.DataFrame:
    df = spark.read.option("header", True).csv(csv_path)

    # Basic feature engineering: votes, purchased, time, comment length
    df2 = (
        df.select(
            F.col("votes").cast("double").alias("votes"),
            (F.when(F.upper(F.col("purchased")) == "TRUE", 1.0).otherwise(0.0)).alias("purchased"),
            F.col("time").cast("double").alias("time"),
            F.length(F.col("comment")).cast("double").alias("comment_len"),
            F.col("rating").cast("int").alias("rating"),
        )
        .na.fill({"votes": 0.0, "time": 0.0, "comment_len": 0.0, "purchased": 0.0})
    )

    pdf = df2.toPandas()
    # Drop rows without label
    pdf = pdf[pd.notnull(pdf["rating"])] 
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


def predict_csv(spark: SparkSession, input_csv: str, model_path: Path = MODEL_PATH) -> pd.DataFrame:
    # Read input and extract features; preserve `id` if present for submission format
    df = spark.read.option("header", True).csv(input_csv)
    df2 = (
        df.select(
            F.col("id").alias("id"),
            F.col("votes").cast("double").alias("votes"),
            (F.when(F.upper(F.col("purchased")) == "TRUE", 1.0).otherwise(0.0)).alias("purchased"),
            F.col("time").cast("double").alias("time"),
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
        preds = model(xb).squeeze(1).cpu().numpy()
        preds = np.clip(preds, 1.0, 5.0)

    # Round/clamp predictions to integer ratings 1-5
    ratings = np.rint(preds).astype(int)
    ratings = np.clip(ratings, 1, 5)

    if ids is not None:
        out = pd.DataFrame({"id": ids.astype(int), "rating": ratings})
    else:
        out = pd.DataFrame({"id": np.arange(len(ratings)), "rating": ratings})

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
    return cfg


def main():
    cfg = load_config("config.cfg")
    spark = get_spark(app_name=cfg.get("app_name"))
    preds = predict_csv(spark, cfg.get("input_csv"), model_path=Path(cfg.get("model_path")))
    output = cfg.get("output_csv") or "prediction_output/prediction_result.csv"
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(output, index=False)
    print(f"Predictions written to {output}")


if __name__ == "__main__":
    main()
