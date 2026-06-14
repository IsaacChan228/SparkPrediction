"""Merge cleaned prediction or training data with product info.

This module left-joins a cleaned left-side CSV (prediction_input/test_clean.csv
by default, or training_data/train_clean.csv when `--train` is used)
with product_info/prodInfo_clean.csv on `parent_prod_id`.

Product columns are prefixed with `prod_`. The original product `id` field
from `prodInfo_clean.csv` is excluded from the merged output. If no matching
product is found for a left row, all `prod_` columns for that row are set to
the literal string "NA". When multiple product_info rows share the same
`parent_prod_id`, the first-seen product row is used.

API:
- `merge_train_with_prod(..., use_prediction: bool = True)` performs the
    merge and returns the output Path.

CLI:
- `python data_merging.py` (default merges prediction_input/test_clean.csv)
- `python data_merging.py --train` (merge training_data/train_clean.csv)
"""

from __future__ import annotations

import csv
from pathlib import Path
import argparse
from typing import Dict


DEFAULT_TRAIN_PRED = Path("prediction_input/test_clean.csv")
DEFAULT_TRAIN_TRAIN = Path("training_data/train_clean.csv")
DEFAULT_PROD = Path("product_info/prodInfo_clean.csv")
DEFAULT_OUT_PRED = Path("prediction_input/test_merged.csv")
DEFAULT_OUT_TRAIN = Path("training_data/train_merged.csv")


def merge_train_with_prod(
    train_path: Path | str | None = None,
    prod_path: Path | str = DEFAULT_PROD,
    output_path: Path | str | None = None,
    use_prediction: bool = True,
) -> Path:
    # select defaults if explicit paths not provided
    if train_path is None:
        train_path = DEFAULT_TRAIN_PRED if use_prediction else DEFAULT_TRAIN_TRAIN
    if output_path is None:
        output_path = DEFAULT_OUT_PRED if use_prediction else DEFAULT_OUT_TRAIN

    train_path = Path(train_path)
    prod_path = Path(prod_path)
    output_path = Path(output_path)

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

    # Read train file and write merged output
    with train_path.open(encoding="utf-8", newline="") as th:
        train_reader = csv.DictReader(th)
        train_fields = train_reader.fieldnames or []

        # Prepare product output fields (prefix and skip parent_prod_id and prod id 'id' to avoid duplicate)
        prod_out_fields = [f"prod_{f}" for f in prod_fields if f and f not in {"parent_prod_id", "id"}]

        out_fields = list(train_fields) + prod_out_fields

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=out_fields)
            writer.writeheader()

            for trow in train_reader:
                key = trow.get("parent_prod_id")
                prow = prod_map.get(key, {})

                out_row = dict(trow)
                # inject prod fields (prefix keys)
                for f in prod_fields:
                    if not f or f in {"parent_prod_id", "id"}:
                        continue
                    out_key = f"prod_{f}"
                    if prow:
                        out_row[out_key] = prow.get(f, "")
                    else:
                        out_row[out_key] = "NA"

                writer.writerow(out_row)

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
