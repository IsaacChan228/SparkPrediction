"""Training driver: Spark preprocessing + PyTorch MLP training.

Loads cleaned training data via a Spark session, performs light feature
engineering (numeric sanitization, optional precomputed embeddings), and
trains a small PyTorch MLP regressor for the ``rating`` target.

Features:
- Streaming-safe preprocessing via `load_and_preprocess()`.
- Heuristic driver memory estimation via `estimate_required_driver_memory()`.
- Trained artifacts saved to `Model/pytorch_mlp.pt` and a simple training
    report written to the configured report path.

"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple
import configparser
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from spark_prediction import (
    MODEL_PATH,
    get_spark,
    load_and_preprocess,
    TabularDataset,
    MLP,
)

# Enforce these text columns must be represented by embeddings
ALLOWED_BERT_COLS = ["comment", "title", "prod_title", "prod_features"]


def train_model(
    data: pd.DataFrame,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    model_path: Path = MODEL_PATH,
    weight_decay: float = 1e-6,
    early_stopping_patience: int = 5,
) -> Tuple[MLP, dict]:
    """Train an `MLP` on the provided tabular DataFrame and return the
    trained model and a summary dict with metrics and timing information.

    The function expects `data` to already contain a numeric feature frame
    and a `label` column. Precomputed embedding columns (e.g. "comment_emb_0")
    will be detected and used automatically.
    """
    print("Preparing training data...")
    # Only accept precomputed numeric embedding columns named '<col>_emb_<i>'
    bert_cols = ALLOWED_BERT_COLS
    # Base numeric feature columns expected for training
    feature_cols = ["votes", "purchased", "time"]

    # Detect numeric embedding columns stored in the dataframe
    emb_cols_detected: list[str] = []
    for c in bert_cols:
        prefix = f"{c}_emb_"
        matches = [col for col in data.columns if col.startswith(prefix)]
        if matches:
            try:
                matches = sorted(matches, key=lambda x: int(x.rsplit("_", 1)[1]))
            except Exception:
                matches = sorted(matches)
            emb_cols_detected.extend(matches)

    if emb_cols_detected:
        feature_cols.extend(emb_cols_detected)
    else:
        missing = [c for c in ALLOWED_BERT_COLS if not any(col.startswith(f"{c}_emb_") for col in data.columns)]
        raise ValueError(f"Missing required embeddings for: {missing}. Run preprocessing (data_merging) to add <col>_emb_<i> numeric columns.")

    # Include product-level features when present
    for pc in ["prod_price", "prod_rating_number", "prod_main_category", "prod_store"]:
        if pc in data.columns:
            feature_cols.append(pc)

    X = data[feature_cols].astype(np.float32).values
    y = data["label"].astype(np.float32).values

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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=3, factor=0.5)
    best_val_mse = float("inf")
    best_state: dict | None = None
    no_improve = 0

    use_amp = torch.cuda.is_available()
    # Create a GradScaler in a way that's compatible across torch versions.
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

    # Read commonly tuned training parameters and paths with sensible
    # fallbacks to allow running without a complete config file.
    cfg = {}
    cfg["epochs"] = cp.getint("training", "epochs", fallback=100)
    cfg["batch_size"] = cp.getint("training", "batch_size", fallback=64)
    cfg["lr"] = cp.getfloat("training", "lr", fallback=1e-3)
    cfg["weight_decay"] = cp.getfloat("training", "weight_decay", fallback=1e-6)
    cfg["early_stopping_patience"] = cp.getint("training", "early_stopping_patience", fallback=5)

    cfg["train_csv"] = cp.get("paths", "train_csv", fallback="training_data/train_merged.csv")
    cfg["input_csv"] = cp.get("paths", "input_csv", fallback="prediction_input/test_merged.csv")
    cfg["model_path"] = cp.get("paths", "model_path", fallback=str(MODEL_PATH))
    cfg["report_path"] = cp.get("paths", "report_path", fallback="prediction_output/training_report.txt")
    # feature flags
    cfg["bert_model"] = cp.get("features", "bert_model", fallback="all-MiniLM-L6-v2")

    return cfg


def _parse_mem_string(mem: str) -> int:
    """Parse human-friendly memory strings like '16g' or '512m' into bytes."""
    if not mem:
        return 0
    mem = str(mem).strip().lower()
    try:
        if mem.endswith("g"):
            return int(float(mem[:-1]) * 1024 ** 3)
        if mem.endswith("m"):
            return int(float(mem[:-1]) * 1024 ** 2)
        if mem.endswith("k"):
            return int(float(mem[:-1]) * 1024)
        return int(mem)
    except Exception:
        return 0


def estimate_required_driver_memory(csv_path: str | Path, sample_lines: int = 1000, expand_factor: float = 6.0) -> dict:
    """Estimate a recommended Spark driver heap (human-readable) based on CSV size.

    Heuristic: sample a number of rows to compute average bytes/row, estimate rows,
    multiply by an expansion factor (parsing + pandas/numpy in-memory) and the
    configured batch size used when streaming rows to pandas. Returns a dict
    containing `recommended` (e.g. '32g') and diagnostic fields.
    """
    import os
    try:
        csv_path = str(csv_path)
        file_size = os.path.getsize(csv_path)
        avg_line = None
        with open(csv_path, "rb") as fh:
            # skip header
            fh.readline()
            lengths = []
            for i in range(sample_lines):
                line = fh.readline()
                if not line:
                    break
                lengths.append(len(line))
        if lengths:
            avg_line = float(sum(lengths)) / len(lengths)
        else:
            # fallback to average considering whole file size and 1 line minimal
            avg_line = max(1.0, float(file_size))

        est_rows = int(file_size / max(1.0, avg_line))
        batch_size = int(os.environ.get("SPARK_TO_PANDAS_BATCH_SIZE", "500"))

        per_row_mem = avg_line * float(expand_factor)
        rows_in_batch = min(batch_size, max(1, est_rows))
        batch_mem = per_row_mem * rows_in_batch

        # Add headroom for pandas, Python and the JVM: 2GB base + 20% buffer
        total_required = batch_mem * 1.2 + (2 * 1024 ** 3)

        # Round to sensible GB multiples (min 4GB, round up to multiple of 4GB)
        gb = 1024 ** 3
        req_gb = int((total_required + gb - 1) // gb)
        if req_gb < 4:
            req_gb = 4
        elif req_gb % 4 != 0:
            req_gb = ((req_gb // 4) + 1) * 4

        recommended = f"{req_gb}g"
        return {
            "recommended": recommended,
            "estimated_rows": est_rows,
            "avg_line_bytes": avg_line,
            "batch_size": batch_size,
            "batch_mem_bytes": int(batch_mem),
        }
    except Exception as e:
        return {"error": str(e)}


def main():
    import os

    cfg = load_config("config.cfg")
    train_csv = cfg.get("train_csv")

    # Estimate required driver memory based on CSV size and batch settings
    try:
        est = estimate_required_driver_memory(train_csv)
        if "recommended" in est:
            print(
                f"Estimated required Spark driver memory: {est['recommended']} "
                f"(batch_size={est['batch_size']}, estimated_rows={est['estimated_rows']}, avg_line_bytes={est['avg_line_bytes']:.1f})"
            )
            existing = os.environ.get("SPARK_DRIVER_MEMORY")
            existing_bytes = _parse_mem_string(existing) if existing else 0
            rec_bytes = _parse_mem_string(est["recommended"])
            if rec_bytes > existing_bytes:
                print(f"Current SPARK_DRIVER_MEMORY={existing or 'unset'}; recommended={est['recommended']}")
                if os.environ.get("AUTO_SET_SPARK_DRIVER_MEMORY", "0") == "1":
                    os.environ["SPARK_DRIVER_MEMORY"] = est["recommended"]
                    # also set executor memory to the same recommended value
                    os.environ["SPARK_EXECUTOR_MEMORY"] = est["recommended"]
                    # set driver max result size to half of recommended driver memory (but at least 1g)
                    try:
                        rec_b = _parse_mem_string(est["recommended"])
                        half_b = max(1024 ** 3, rec_b // 2)
                        gb = 1024 ** 3
                        half_g = int((half_b + gb - 1) // gb)
                        os.environ["SPARK_DRIVER_MAX_RESULT_SIZE"] = f"{half_g}g"
                    except Exception:
                        pass
                    print("AUTO_SET_SPARK_DRIVER_MEMORY=1 -> SPARK_DRIVER_MEMORY, SPARK_EXECUTOR_MEMORY, and SPARK_DRIVER_MAX_RESULT_SIZE set to recommended values")
                else:
                    print("Tip: set SPARK_DRIVER_MEMORY to the recommended value to avoid Java heap OOMs.")
        else:
            print(f"Could not estimate required driver memory: {est.get('error')}")
    except Exception as e:
        print(f"Failed to compute memory estimate: {e}")

    spark = get_spark()
    print("Spark session started")

    model_path = Path(cfg.get("model_path"))

    epochs = cfg.get("epochs")
    batch_size = cfg.get("batch_size")
    lr = cfg.get("lr")
    weight_decay = cfg.get("weight_decay")
    early_stopping_patience = cfg.get("early_stopping_patience")

    # measure preprocessing time
    print(f"Preprocessing input CSV: {train_csv}")
    pre_start = time.perf_counter()
    # run preprocessing
    df = load_and_preprocess(spark, train_csv)
    # Reduce dataframe to only feature + label columns to minimize driver memory
    # Detect embedding columns and product feature cols, mirror train_model logic
    bert_cols = ALLOWED_BERT_COLS
    emb_columns: list[str] = []
    for c in bert_cols:
        prefix = f"{c}_emb_"
        matches = [col for col in df.columns if col.startswith(prefix)]
        if matches:
            try:
                matches = sorted(matches, key=lambda x: int(x.rsplit("_", 1)[1]))
            except Exception:
                matches = sorted(matches)
            emb_columns.extend(matches)

    feature_cols = ["votes", "purchased", "time"]
    for pc in ["prod_price", "prod_rating_number", "prod_main_category", "prod_store"]:
        if pc in df.columns:
            feature_cols.append(pc)
    feature_cols.extend(emb_columns)

    # Keep only features + label and downcast numeric dtypes to float32 to save memory
    keep_cols = [c for c in feature_cols if c in df.columns] + ["label"]
    try:
        df = df[keep_cols].astype(dtype="float32")
    except Exception:
        # fallback: leave as-is if casting fails
        pass

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
