"""Merge cleaned prediction or training data with product info.

This module left-joins a cleaned left-side CSV (prediction_input/test_clean.csv
by default, or training_data/train_clean.csv when `--train` is used)
with product_info/prodInfo_clean.csv on `parent_prod_id`.

Product columns are prefixed with `prod_`. The original product `id` field
from `prodInfo_clean.csv` is excluded from the merged output. If no matching
product is found for a left row, all `prod_` columns for that row are set to
the literal string "NA". When multiple product_info rows share the same
`parent_prod_id`, the first-seen product row is used.

This merge step also computes sentence-transformer embeddings using
`all-MiniLM-L6-v2` for text columns (title, comment, product title,
product features) and expands those embeddings into numeric columns named
like `<col>_emb_0`, `<col>_emb_1`, ... so downstream training and
prediction consume numeric features only.

API:
- `merge_train_with_prod(..., use_prediction: bool = True)` performs the
    merge, computes embeddings, and returns the output Path.

CLI:
- `python data_merging.py` (merge prediction input and compute embeddings)
- `python data_merging.py --train` (merge training input and compute embeddings)
"""

from __future__ import annotations

import csv
from pathlib import Path
import argparse
from typing import Dict, List, Optional
import numpy as np


DEFAULT_TRAIN_PRED = Path("prediction_input/test_clean.csv")
DEFAULT_TRAIN_TRAIN = Path("training_data/train_clean.csv")
DEFAULT_PROD = Path("product_info/prodInfo_clean.csv")
DEFAULT_OUT_PRED = Path("prediction_input/test_merged.csv")
DEFAULT_OUT_TRAIN = Path("training_data/train_merged.csv")
DEFAULT_EMB_DIM = 32


def merge_train_with_prod(
    train_path: Path | str | None = None,
    prod_path: Path | str = DEFAULT_PROD,
    output_path: Path | str | None = None,
    use_prediction: bool = True,
    bert_model_name: str = "all-MiniLM-L6-v2",
    bert_cols: Optional[List[str]] = None,
    batch_size: int = 256,
    target_emb_dim: int = DEFAULT_EMB_DIM,
) -> Path:
    # select defaults if explicit paths not provided
    if train_path is None:
        train_path = DEFAULT_TRAIN_PRED if use_prediction else DEFAULT_TRAIN_TRAIN
    if output_path is None:
        output_path = DEFAULT_OUT_PRED if use_prediction else DEFAULT_OUT_TRAIN

    train_path = Path(train_path)
    prod_path = Path(prod_path)
    output_path = Path(output_path)

    compressor_dir = Path("artifacts/emb_compressors")
    compressor_dir.mkdir(parents=True, exist_ok=True)

    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not prod_path.exists():
        raise FileNotFoundError(f"Product file not found: {prod_path}")

    # Load product info into a map keyed by parent_prod_id (keep first seen)
    prod_map: Dict[str, Dict[str, str]] = {}
    with prod_path.open(encoding="utf-8", newline="") as ph:
        prod_reader = csv.DictReader(ph)
        prod_fields = prod_reader.fieldnames or []
        for prow in prod_reader:
            key = prow.get("parent_prod_id")
            if key and key not in prod_map:
                prod_map[key] = dict(prow)

    # Read train file into memory so we can compute and attach embeddings
    with train_path.open(encoding="utf-8", newline="") as th:
        train_reader = csv.DictReader(th)
        train_fields = train_reader.fieldnames or []
        trows = [dict(r) for r in train_reader]

    # Prepare product output fields (prefix and skip parent_prod_id and prod id 'id' to avoid duplicate)
    prod_out_fields = [f"prod_{f}" for f in prod_fields if f and f not in {"parent_prod_id", "id"}]

    # Build merged rows in memory
    out_rows: List[Dict[str, str]] = []
    for trow in trows:
        prod_id = trow.get("prod_id")
        parent_key = trow.get("parent_prod_id")
        prow = {}
        if prod_id and prod_id in prod_map:
            prow = prod_map[prod_id]
        elif parent_key and parent_key in prod_map:
            prow = prod_map[parent_key]

        out_row = dict(trow)
        for f in prod_fields:
            if not f or f in {"parent_prod_id", "id"}:
                continue
            out_key = f"prod_{f}"
            if prow:
                out_row[out_key] = prow.get(f, "")
            else:
                out_row[out_key] = "NA"

        out_rows.append(out_row)

    # Compute sentence-transformer embeddings on the merged data and expand
    # them into numeric columns (embeddings are computed during merge and
    # written as numeric fields)
    emb_fieldnames: List[str] = []

    if bert_cols is None:
        bert_cols = ["title", "comment", "prod_title", "prod_features"]
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(bert_model_name)

        def _batch_encode_texts(texts: List[str], batch_sz: int) -> np.ndarray:
            chunks: List[np.ndarray] = []
            buf: List[str] = []
            processed = 0
            total = len(texts)
            for txt in texts:
                buf.append("") if txt is None else buf.append(str(txt))
                if len(buf) >= batch_sz:
                    emb = model.encode(buf, show_progress_bar=False, convert_to_numpy=True)
                    chunks.append(emb)
                    processed += len(buf)
                    print(f"    encoded {processed}/{total} rows (batch_size={batch_sz})", end="\r", flush=True)
                    buf = []
            if buf:
                emb = model.encode(buf, show_progress_bar=False, convert_to_numpy=True)
                chunks.append(emb)
                processed += len(buf)
                print(f"    encoded {processed}/{total} rows (final chunk)", end="\r", flush=True)
            if not chunks:
                return np.empty((0, 0))
            print("", flush=True)
            return np.vstack(chunks)

        def _fit_compressor(mat: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
            """Fit compressor on mat and return (mean, comp) where comp has shape (k, d)."""
            n, d = mat.shape
            mean = mat.mean(axis=0)
            Xc = mat - mean
            try:
                u, s, vt = np.linalg.svd(Xc, full_matrices=False)
                comp = vt[:k]
                return mean, comp
            except Exception:
                # deterministic random projection (fixed seed)
                rng = np.random.RandomState(0)
                proj = rng.randn(d, k) / np.sqrt(k)
                comp = proj.T
                return mean, comp

        def _apply_compressor(mat: np.ndarray, mean: np.ndarray, comp: np.ndarray) -> np.ndarray:
            if mat.size == 0:
                return mat
            Xc = mat - mean
            return Xc.dot(comp.T)

        for idx, col in enumerate(bert_cols):
            # Only compute embeddings for columns present in the merged rows
            if not out_rows or col not in out_rows[0]:
                continue
            texts = [r.get(col, "") or "" for r in out_rows]
            print(f"Starting encoding for column '{col}' ({idx+1}/{len(bert_cols)}), {len(texts)} rows")
            try:
                emb = _batch_encode_texts(texts, batch_size)
                if emb.size:
                    emb_dim = int(emb.shape[1])
                    compressor_path = compressor_dir / f"{col}_compressor.npz"

                    if not use_prediction:
                        # training run: fit compressor on training embeddings and save
                        mean, comp = _fit_compressor(emb, min(target_emb_dim, emb_dim))
                        emb_reduced = _apply_compressor(emb, mean, comp)
                        out_dim = emb_reduced.shape[1]
                        # persist compressor
                        try:
                            # persist as float32 to reduce disk and memory footprint
                            np.savez_compressed(compressor_path, mean=mean.astype(np.float32), comp=comp.astype(np.float32))
                        except Exception as ex:
                            print(f"WARNING: failed to save compressor for '{col}': {ex}")
                        print(f"INFO: fitted and saved compressor for '{col}' (orig_dim={emb_dim}, written_dim={out_dim})")
                    else:
                        # prediction run: load existing compressor and apply
                        if compressor_path.exists():
                            try:
                                with np.load(compressor_path) as data:
                                    mean = data['mean']
                                    comp = data['comp']
                                emb_reduced = _apply_compressor(emb, mean, comp)
                                out_dim = emb_reduced.shape[1]
                                print(f"INFO: loaded compressor for '{col}' and applied (orig_dim={emb_dim}, written_dim={out_dim})")
                            except Exception as ex:
                                print(f"WARNING: failed to load/apply compressor for '{col}': {ex}; falling back to fitting locally")
                                mean, comp = _fit_compressor(emb, min(target_emb_dim, emb_dim))
                                emb_reduced = _apply_compressor(emb, mean, comp)
                                out_dim = emb_reduced.shape[1]
                        else:
                            print(f"WARNING: compressor for '{col}' not found at {compressor_path}; fitting locally")
                            mean, comp = _fit_compressor(emb, min(target_emb_dim, emb_dim))
                            emb_reduced = _apply_compressor(emb, mean, comp)
                            out_dim = emb_reduced.shape[1]

                    col_names = [f"{col}_emb_{i}" for i in range(out_dim)]
                    emb_fieldnames.extend(col_names)
                    for i, r in enumerate(out_rows):
                        vec = emb_reduced[i] if i < emb_reduced.shape[0] else np.zeros(out_dim, dtype=float)
                        for j, val in enumerate(vec):
                            # round embedding numeric values to 6 decimal places to reduce CSV size
                            try:
                                r[col_names[j]] = float(round(float(val), 6))
                            except Exception:
                                r[col_names[j]] = float(val)
            except Exception as ex:
                print(f"WARNING: failed to compute embeddings for '{col}': {ex}")
    except Exception as ex:
        print(f"INFO: sentence-transformers not available or failed to import: {ex}")

    # Build a reduced output column list to avoid saving large unused text
    # Keep: id, core numeric features, a small set of product numeric fields,
    # any computed embedding columns, and rating (if present for training).
    core_keep = [c for c in ("id", "votes", "purchased", "time", "rating") if c in train_fields]
    prod_keep = [c for c in ("prod_price", "prod_rating_number", "prod_main_category", "prod_store") if c in prod_out_fields]
    out_fields = core_keep + prod_keep + emb_fieldnames

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {len(out_rows)} reduced merged rows to {output_path}")
    with output_path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=out_fields)
        writer.writeheader()
        for i, r in enumerate(out_rows):
            # Ensure missing fields are present before writing
            rowcopy = {k: r.get(k, "") for k in out_fields}
            writer.writerow(rowcopy)
            if (i + 1) % 1000 == 0:
                print(f"  wrote {i+1}/{len(out_rows)} rows", end='\r', flush=True)
        # final newline after progress
        print("", flush=True)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge cleaned data with product info")
    parser.add_argument("--train", dest="use_train", action="store_true", help="Merge training_data/train_clean.csv instead of prediction_input/test_clean.csv")
    args = parser.parse_args()

    use_prediction = not args.use_train
    out = merge_train_with_prod(use_prediction=use_prediction)
    print(f"Wrote merged data to {out}")


if __name__ == "__main__":
    main()