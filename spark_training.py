"""Spark + PyTorch MLP training and prediction helper.

This script reads cleaned training CSV via PySpark, performs light
feature engineering, trains a small PyTorch MLP to predict the
`rating` field (1-5). It reads cleaned training CSV via
PySpark, performs light feature engineering, trains the model, and
saves the trained model to `Model/pytorch_mlp.pt`.

Prediction functionality was moved to `spark_prediction.py`. To run
predictions use that module (it includes a CLI). Examples:
    Train:
        python spark_training.py --mode train

    Predict:
        python spark_prediction.py

        test
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple
from contextlib import nullcontext
import configparser
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

from spark_prediction import MODEL_PATH, get_spark, load_and_preprocess, TabularDataset, MLP, predict_csv


def train_model(
    data: pd.DataFrame,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 3e-4,
    model_path: Path = MODEL_PATH,
    weight_decay: float = 1e-5,
    early_stopping_patience: int = 5,
) -> Tuple[MLP, dict]:
    print("Preparing training data...")
    X = data[["votes", "purchased", "time", "comment_len"]].values
    y = data["label"].values

    # Diagnostics removed to reduce noisy output

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

    print(f"Total samples: {n}, train: {len(train_idx)}, val: {len(val_idx)}")

    # Feature diagnostics and linear baseline removed to quiet output

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

    print(f"DataLoaders ready (batch_size={batch_size}). Starting training...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_dim=X.shape[1])
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    # scheduler and early stopping
    # Some PyTorch builds don't accept the `verbose` kwarg; omit it for compatibility
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=3, factor=0.5)
    best_val_mse = float("inf")
    best_state: dict | None = None
    no_improve = 0

    use_amp = torch.cuda.is_available()
    # Create a GradScaler in a way that's compatible across torch versions.
    # Newer versions accept `device_type="cuda"` or a positional 'cuda' arg;
    # older versions use torch.cuda.amp.GradScaler(). Try several fallbacks.
    scaler = None
    if use_amp:
        try:
            scaler = torch.amp.GradScaler(device_type="cuda")
        except TypeError:
            try:
                scaler = torch.amp.GradScaler("cuda")
            except TypeError:
                try:
                    scaler = torch.cuda.amp.GradScaler()
                except Exception:
                    scaler = None
    model.train()
    epoch_stats: list[dict] = []
    train_start_time = time.perf_counter()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total = 0
        abs_err = 0.0
        epoch_start = time.perf_counter()
        num_batches = len(train_dl)
        progress_interval = max(1, num_batches // 10)
        batch_idx = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            if use_amp:
                # Use autocast in a version-compatible way; default to torch.amp.autocast
                try:
                    ac = torch.amp.autocast(device_type="cuda")
                except TypeError:
                    try:
                        ac = torch.amp.autocast("cuda")
                    except TypeError:
                        # fallback to the legacy cuda autocast
                        ac = torch.cuda.amp.autocast()

                with ac:
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
            batch_idx += 1
            if batch_idx % progress_interval == 0 or batch_idx == num_batches:
                pct = (batch_idx / num_batches) * 100.0
                print(f"Epoch {epoch}: batch {batch_idx}/{num_batches} ({pct:.0f}%) loss={loss.item():.4f}")

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

        epoch_time = time.perf_counter() - epoch_start
        epoch_stats.append({
            "epoch": epoch,
            "train_mse": float(train_mse),
            "train_rmse": float(train_rmse),
            "train_mae": float(train_mae),
            "val_mse": float(val_mse),
            "val_rmse": float(val_rmse),
            "val_mae": float(val_mae),
            "epoch_time_s": float(epoch_time),
        })

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

    training_total_time = time.perf_counter() - train_start_time

    return model, {
        "train_mse": float(train_mse),
        "train_rmse": float(train_rmse),
        "train_mae": float(train_mae),
        "val_mse": float(val_mse),
        "val_rmse": float(val_rmse),
        "val_mae": float(val_mae),
        "epoch_stats": epoch_stats,
        "training_total_time_s": float(training_total_time),
    }


def load_config(path: str | Path) -> dict:
    cp = configparser.ConfigParser()
    cp.read(path)

    cfg = {}
    cfg["epochs"] = cp.getint("training", "epochs", fallback=10)
    cfg["batch_size"] = cp.getint("training", "batch_size", fallback=64)
    cfg["lr"] = cp.getfloat("training", "lr", fallback=1e-3)
    cfg["weight_decay"] = cp.getfloat("training", "weight_decay", fallback=1e-5)
    cfg["early_stopping_patience"] = cp.getint("training", "early_stopping_patience", fallback=5)

    cfg["train_csv"] = cp.get("paths", "train_csv", fallback="training_data/train_merged.csv")
    cfg["input_csv"] = cp.get("paths", "input_csv", fallback="prediction_input/test_merged.csv")
    cfg["model_path"] = cp.get("paths", "model_path", fallback=str(MODEL_PATH))
    cfg["report_path"] = cp.get("paths", "report_path", fallback="prediction_output/training_report.txt")

    return cfg


def main():
    spark = get_spark()
    print("Spark session started")

    cfg = load_config("config.cfg")

    train_csv = cfg.get("train_csv")
    model_path = Path(cfg.get("model_path"))

    epochs = cfg.get("epochs")
    batch_size = cfg.get("batch_size")
    lr = cfg.get("lr")
    weight_decay = cfg.get("weight_decay")
    early_stopping_patience = cfg.get("early_stopping_patience")

    # measure preprocessing time
    print(f"Preprocessing input CSV: {train_csv}")
    pre_start = time.perf_counter()
    df = load_and_preprocess(spark, train_csv)
    pre_time = time.perf_counter() - pre_start
    try:
        print(f"Preprocessing completed: {len(df)} rows (took {pre_time:.2f}s)")
    except Exception:
        print(f"Preprocessing completed (took {pre_time:.2f}s)")

    # Compute rating distribution, mean and standard deviation from the preprocessed DataFrame
    try:
        total_rows = len(df)
        rating_series = df["label"].astype(float)
        vc = rating_series.value_counts().to_dict()
        # ensure integer keys 1..5 map to counts
        rating_counts = {i: int(vc.get(float(i), 0) or vc.get(i, 0)) for i in range(1, 6)}
        rating_mean = float(rating_series.mean())
        rating_sd = float(rating_series.std())
    except Exception:
        total_rows = len(df)
        rating_counts = {i: 0 for i in range(1, 6)}
        rating_mean = 0.0
        rating_sd = 0.0

    # Write a partial report containing rating distribution and stats so it
    # can be reviewed even if training is terminated later.
    try:
        rp = Path(cfg.get("report_path") or "prediction_output/training_report.txt")
        rp.parent.mkdir(parents=True, exist_ok=True)
        with rp.open("w", encoding="utf-8") as fh:
            fh.write("Partial training report - Rating distribution and statistics\n")
            fh.write("Checked file: {}\n\n".format(train_csv))
            fh.write("Rating distribution (count, percent):\n")
            denom = total_rows if total_rows > 0 else 1
            for r in range(1, 6):
                c = rating_counts.get(r, 0)
                pct = (c / denom) * 100.0
                fh.write(f"Rating {r}: {c} ({pct:.2f}%)\n")
            fh.write("\nRating statistics:\n")
            fh.write(f"Mean: {rating_mean:.4f}\n")
            fh.write(f"SD: {rating_sd:.4f}\n")
        print(f"Wrote partial training report to {rp}")
    except Exception as e:
        print(f"Failed to write partial report: {e}")

    model, results = train_model(
        df,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        model_path=model_path,
        weight_decay=weight_decay,
        early_stopping_patience=early_stopping_patience,
    )

    # Write a simple training report
    training_time = results.get("training_total_time_s", 0.0)
    epoch_stats = results.get("epoch_stats", [])
    offline_total = pre_time + float(training_time)

    report_lines = []
    report_lines.append(f"Preprocessing time (s): {pre_time:.6f}")
    report_lines.append(f"Total training time (s): {training_time:.6f}")
    report_lines.append(f"Offline total time (preprocessing + training) (s): {offline_total:.6f}")
    report_lines.append("")
    # Distribution
    report_lines.append("Rating distribution (count, percent):")
    denom = total_rows if total_rows > 0 else 1
    for r in range(1, 6):
        c = rating_counts.get(r, 0)
        pct = (c / denom) * 100.0
        report_lines.append(f"Rating {r}: {c} ({pct:.2f}%)")
    report_lines.append("")
    # Summary statistics
    report_lines.append("Rating statistics:")
    report_lines.append(f"Mean: {rating_mean:.4f}")
    report_lines.append(f"SD: {rating_sd:.4f}")
    report_lines.append("")
    report_lines.append("Per-epoch timing and metrics:")
    for es in epoch_stats:
        report_lines.append(
            f"Epoch {es['epoch']}: time={es['epoch_time_s']:.6f}s train_rmse={es['train_rmse']:.4f} val_rmse={es['val_rmse']:.4f}"
        )

    report_path = Path(cfg.get("report_path") or "training_report.txt")
    report_path.write_text("\n".join(report_lines))
    print(f"Training report written to {report_path}")


if __name__ == "__main__":
    main()
