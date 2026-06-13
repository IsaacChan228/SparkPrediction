"""Product-info cleaning utilities.

This module reuses schema-loading and text sanitization helpers from
`traindatacleaning.py` to validate and clean
`product_info/prodInfo.csv` according to
`product_info/prod_info_format`.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import List, Sequence, Mapping

# reuse helpers from traindatacleaning
from traindatacleaning import (
    load_expected_fields,
    _sanitize_text_field,
    _PARENT_PROD_ID_RE,
    DEBUG_EXPORT_CORRUPTED,
)


DEFAULT_PROD_PATH = Path("product_info/prodInfo.csv")
DEFAULT_SCHEMA_PATH = Path("product_info/prod_info_format")
DEFAULT_CLEAN_PROD_PATH = Path("product_info/prodInfo_clean.csv")


@dataclass(frozen=True)
class ProdCleaningResult:
    input_rows: int
    clean_rows: int
    corrupted_rows: int
    output_path: Path


def _detect_corrupted_row_worker(args: tuple) -> List[str]:
    row, expected = args
    return detect_corrupted_row(row, expected)


def detect_corrupted_row(row: object, expected_fields: Sequence[str]) -> List[str]:
    problems: List[str] = []
    expected_count = len(expected_fields)

    if isinstance(row, Mapping):
        extra_keys = [k for k in row.keys() if k not in expected_fields and (k not in (None, "") or str(row.get(k)).strip() != "")]
        if extra_keys:
            problems.append("extra attribute")
            return problems
        row_values = [row.get(f) for f in expected_fields]
    else:
        row_values = list(row)
        if len(row_values) < expected_count:
            problems.append("missing attribute")
        if len(row_values) > expected_count:
            problems.append("extra attribute")
        if problems:
            return problems

    # basic emptiness check. Allow empty `price` and normalize it to "NA".
    for idx, (field_name, value) in enumerate(zip(expected_fields, row_values)):
        if field_name == "price":
            if value is None or str(value).strip() == "":
                try:
                    if isinstance(row, Mapping):
                        if isinstance(row, dict):
                            row[field_name] = "NA"
                    else:
                        row_values[idx] = "NA"
                except Exception:
                    pass
                # don't treat missing price as corruption
                continue

        if value is None or str(value).strip() == "":
            problems.append(f"{field_name} missing attribute")
            break
    if problems:
        return problems

    # id numeric check
    try:
        raw_id = row_values[0]
        int(str(raw_id).strip())
    except Exception:
        problems.append("id not integer")

    # parent_prod_id format
    parent = row_values[1]
    if not _PARENT_PROD_ID_RE.fullmatch(str(parent).strip()):
        problems.append("parent_prod_id malformed")

    # rating_number integer non-negative
    rating_num = row_values[7] if len(row_values) > 7 else None
    try:
        rv = int(str(rating_num).strip())
        if rv < 0:
            problems.append("rating_number negative")
    except Exception:
        problems.append("rating_number not integer")

    return problems


def clean_prod_csv(
    csv_path: Path | str = DEFAULT_PROD_PATH,
    schema_path: Path | str = DEFAULT_SCHEMA_PATH,
    output_path: Path | str = DEFAULT_CLEAN_PROD_PATH,
    max_workers: int | None = None,
) -> ProdCleaningResult:
    expected_fields = load_expected_fields(schema_path)
    csv_path = Path(csv_path)
    output_path = Path(output_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Product CSV not found: {csv_path}")

    text = csv_path.read_text(encoding="utf-8", errors="replace")
    parts = text.splitlines()
    # skip header
    rows = []
    for line in parts[1:]:
        if not line.strip():
            continue
        try:
            parsed = next(csv.reader([line]))
        except Exception:
            parsed = [c for c in line.split(",")]
        # pad
        while len(parsed) < len(expected_fields):
            parsed.append("")
        rows.append(parsed)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        problems_list = list(
            executor.map(_detect_corrupted_row_worker, zip(rows, [expected_fields] * len(rows)), chunksize=256)
        )

    input_rows = 0
    clean_rows = 0
    corrupted_rows = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    corrupted_handle = None
    corrupted_writer = None
    if DEBUG_EXPORT_CORRUPTED:
        corrupted_path = output_path.with_name(output_path.stem + "_corrupted.csv")
        corrupted_handle = corrupted_path.open("w", encoding="utf-8", newline="")

    with output_path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(expected_fields)
        if corrupted_handle is not None:
            corrupted_writer = csv.writer(corrupted_handle)
            corrupted_writer.writerow([*expected_fields, "corruption_reasons"])

        for row, problems in zip(rows, problems_list):
            input_rows += 1
            if problems:
                corrupted_rows += 1
                if corrupted_writer is not None:
                    corrupted_writer.writerow([*row, "; ".join(problems)])
                continue

            # sanitize text fields (title, features, store)
            # attempt to find indices
            try:
                title_idx = expected_fields.index("title")
            except ValueError:
                title_idx = None
            try:
                features_idx = expected_fields.index("features")
            except ValueError:
                features_idx = None
            try:
                store_idx = expected_fields.index("store")
            except ValueError:
                store_idx = None
            try:
                price_idx = expected_fields.index("price")
            except ValueError:
                price_idx = None

            if title_idx is not None and len(row) > title_idx:
                row[title_idx] = _sanitize_text_field(row[title_idx])
            if features_idx is not None and len(row) > features_idx:
                row[features_idx] = _sanitize_text_field(row[features_idx])
            if store_idx is not None and len(row) > store_idx:
                row[store_idx] = _sanitize_text_field(row[store_idx])

            # normalize empty price to "NA"
            if price_idx is not None and len(row) > price_idx:
                if row[price_idx] is None or str(row[price_idx]).strip() == "":
                    row[price_idx] = "NA"

            writer.writerow(row[: len(expected_fields)])
            clean_rows += 1

    if corrupted_handle is not None:
        corrupted_handle.close()

    return ProdCleaningResult(input_rows=input_rows, clean_rows=clean_rows, corrupted_rows=corrupted_rows, output_path=output_path)


def main() -> None:
    res = clean_prod_csv()
    print(f"Cleaned product data: {res.clean_rows} valid rows, {res.corrupted_rows} corrupted rows removed, output written to {res.output_path}")


if __name__ == "__main__":
    main()
