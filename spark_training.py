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


def train_model(
    data: pd.DataFrame,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 3e-4,
    model_path: Path = MODEL_PATH,
    weight_decay: float = 1e-5,
    early_stopping_patience: int = 5,
) -> Tuple[MLP, dict]:
    X = data[["votes", "purchased", "time", "comment_len"]].values
    y = data["label"].values

    # 80/20 train/validation split
    n = len(y)
    if n == 0:
        raise ValueError("Empty dataset passed to train_model")
    perm = np.random.permutation(n)
    split = int(0.8 * n)
    split = max(1, min(split, n - 1))

    train_idx = perm[:split]
    val_idx = perm[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # Standardize features using train statistics
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    train_ds = TabularDataset(X_train, y_train)
    val_ds = TabularDataset(X_val, y_val)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_dim=X.shape[1])
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    # scheduler and early stopping
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=3, factor=0.5, verbose=True)
    best_val_mse = float("inf")
    best_state: dict | None = None
    no_improve = 0

    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total = 0
        abs_err = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            if use_amp:
                with torch.cuda.amp.autocast():
                    preds = model(xb).squeeze(1)
                    loss = loss_fn(preds, yb)
                scaler.scale(loss).backward()
                # gradient clipping
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(opt)
                scaler.update()
            else:
                preds = model(xb).squeeze(1)
                loss = loss_fn(preds, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()

            bs = xb.size(0)
            total_loss += loss.item() * bs
            abs_err += torch.abs(preds - yb).sum().item()
            total += bs

        train_mse = total_loss / max(1, total)
        train_rmse = float(np.sqrt(train_mse))
        train_mae = abs_err / max(1, total)

        # Validation pass
        model.eval()
        with torch.no_grad():
            v_loss = 0.0
            v_total = 0
            v_abs_err = 0.0
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).squeeze(1)
                loss = loss_fn(preds, yb)
                bs = xb.size(0)
                v_loss += loss.item() * bs
                v_abs_err += torch.abs(preds - yb).sum().item()
                v_total += bs

            val_mse = v_loss / max(1, v_total)
            val_rmse = float(np.sqrt(val_mse))
            val_mae = v_abs_err / max(1, v_total)

        print(
            f"Epoch {epoch}/{epochs}: train_rmse={train_rmse:.4f} train_mae={train_mae:.4f} "
            f"val_rmse={val_rmse:.4f} val_mae={val_mae:.4f}"
        )

        # scheduler step (ReduceLROnPlateau expects a metric)
        scheduler.step(val_mse)

        # checkpoint best model by validation MSE
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        model.train()

        # early stopping
        if no_improve >= early_stopping_patience:
            print(f"Early stopping after {epoch} epochs (no improvement in {no_improve} epochs)")
            break

    # Save best model and scaler
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        save_obj = {"model_state": best_state, "scaler": {"mean": mean.tolist(), "std": std.tolist()}}
        torch.save(save_obj, str(model_path))
        print(f"Best model (val_mse={best_val_mse:.6f}) saved to {model_path}")
    else:
        # fallback: save final state
        torch.save(model.state_dict(), str(model_path))
        print(f"Model saved to {model_path}")

    return model, {"train_mse": train_mse, "train_rmse": train_rmse, "train_mae": train_mae, "val_mse": val_mse, "val_rmse": val_rmse, "val_mae": val_mae}


def predict_csv(spark: SparkSession, input_csv: str, model_path: Path = MODEL_PATH) -> pd.DataFrame:
    data = load_and_preprocess(spark, input_csv)
    X = data[["votes", "purchased", "time", "comment_len"]].values

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
