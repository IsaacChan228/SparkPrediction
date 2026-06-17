"""Spark + PyTorch MLP training and prediction helper.

This script reads cleaned training CSV via PySpark, performs light
feature engineering, trains a small PyTorch MLP to predict the
`rating` field (1-5), and saves the trained model to
`Model/pytorch_mlp.pt`.

Usage examples:
  Train:
    python spark_training.py --mode train --train-csv training_data/train_merged.csv

  Predict (CSV in same format):
    python spark_training.py --mode predict --input-csv prediction_input/test_merged.csv

Notes:
- This implementation collects features to the driver as a pandas
  dataframe before feeding them to PyTorch. It's intended for small to
  medium datasets. For large-scale training, replace the collection step
  with a distributed training approach (Horovod, PyTorch on Spark, etc.).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
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
    # For regression use the raw rating as the target (1.0..5.0)
    pdf["label"] = pdf["rating"].astype(float)
    features = pdf[["votes", "purchased", "time", "comment_len"]].astype(float)
    pdf_features = features.fillna(0.0)
    pdf_final = pd.concat([pdf_features, pdf[["label"]].astype(float)], axis=1)
    return pdf_final


class TabularDataset(Dataset):
    def __init__(self, arr: np.ndarray, labels: np.ndarray):
        self.x = torch.from_numpy(arr).float()
        # regression target as float
        self.y = torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64, out: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, out),
        )

    def forward(self, x):
        return self.net(x)


def train_model(
    data: pd.DataFrame,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    model_path: Path = MODEL_PATH,
) -> Tuple[MLP, dict]:
    X = data[["votes", "purchased", "time", "comment_len"]].values
    y = data["label"].values

    ds = TabularDataset(X, y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_dim=X.shape[1])
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # Use MSE as the base loss; report RMSE (sqrt of MSE)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total = 0
        abs_err = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).squeeze(1)
            loss = loss_fn(preds, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

            batch_size_eff = xb.size(0)
            total_loss += loss.item() * batch_size_eff
            abs_err += torch.abs(preds - yb).sum().item()
            total += batch_size_eff

        mse = total_loss / max(1, total)
        rmse = float(np.sqrt(mse))
        mae = abs_err / max(1, total)
        print(f"Epoch {epoch}/{epochs}: mse={mse:.4f} rmse={rmse:.4f} mae={mae:.4f}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(model_path))
    print(f"Model saved to {model_path}")
    return model, {"mse": mse, "rmse": rmse, "mae": mae}


def predict_csv(spark: SparkSession, input_csv: str, model_path: Path = MODEL_PATH) -> pd.DataFrame:
    data = load_and_preprocess(spark, input_csv)
    X = data[["votes", "purchased", "time", "comment_len"]].values

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_dim=X.shape[1])
    model.load_state_dict(torch.load(str(model_path), map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        xb = torch.from_numpy(X).float().to(device)
        preds = model(xb).squeeze(1).cpu().numpy()
        # clip predictions to valid rating range
        preds = np.clip(preds, 1.0, 5.0)

    out = pd.DataFrame({"prediction": preds})
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "predict"], required=True)
    p.add_argument("--train-csv", default="training_data/train_merged.csv")
    p.add_argument("--input-csv", default="prediction_input/test_merged.csv")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--model-path", default=str(MODEL_PATH))
    return p.parse_args()


def main():
    args = parse_args()
    spark = get_spark()

    if args.mode == "train":
        df = load_and_preprocess(spark, args.train_csv)
        train_model(df, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, model_path=Path(args.model_path))
    else:
        preds = predict_csv(spark, args.input_csv, model_path=Path(args.model_path))
        print(preds.head())


if __name__ == "__main__":
    main()
