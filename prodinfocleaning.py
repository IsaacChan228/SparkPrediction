"""Product-info cleaning utilities.

Reads the canonical schema at ``product_info/prod_info_format.cfg`` and
validates the rows in ``product_info/prodInfo.csv``. The module focuses on
detecting malformed or missing semantic fields, normalizing benign issues,
and exporting a cleaned CSV suitable for downstream processing.

Validation highlights:
- Ensures row column count matches the schema.
- Detects missing values (but tolerates and normalizes some fields).
- Validates `id` is an integer and `parent_prod_id` follows the expected
    identifier pattern.
- Ensures `rating_number` is an integer; negative values are normalized to 0.

Normalization and sanitation:
- Empty `title`, `price`, `main_category`, and `store` fields are set to
    the placeholder "NA" rather than treated as corruption.
- Text fields are sanitized to remove undecodable/control characters.

Outputs:
- `clean_prod_csv()` writes the cleaned CSV to the supplied `output_path`.
- When `DEBUG_EXPORT_CORRUPTED` is True, corrupted rows are written to
    a sibling file with the suffix `_corrupted.csv` including reasons.
"""

from __future__ import annotations

import csv
import os
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
from traindatacleaning import _print_progress


DEFAULT_PROD_PATH = Path("product_info/prodInfo.csv")
DEFAULT_SCHEMA_PATH = Path("product_info/prod_info_format.cfg")
DEFAULT_CLEAN_PROD_PATH = Path("product_info/prodInfo_clean.csv")


@dataclass(frozen=True)
class ProdCleaningResult:
    """Summary of a product cleaning run."""

    input_rows: int
    clean_rows: int
    corrupted_rows: int
    output_path: Path


def _detect_corrupted_row_worker(args: tuple) -> List[str]:
    """Worker wrapper for parallel row validation.

    Accepts either a positional sequence (list of field values) or a mapping
    (dict from header name to value). This wrapper unifies the input for the
    downstream `detect_corrupted_row` function used by worker processes.
    """
    row, expected = args
    return detect_corrupted_row(row, expected)


def detect_corrupted_row(row: object, expected_fields: Sequence[str]) -> List[str]:
    """Return a list of corruption reasons for a parsed CSV row.

    The function accepts either a positional sequence (list/tuple of values)
    or a mapping-like object (for example a `csv.DictReader` row). Mapping
    inputs are checked for unexpected keys; sequence inputs are validated by
    length and positional semantics. Non-fatal issues are normalized in-place
    when possible so the parent process sees corrected values.
    """
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

    # basic emptiness check. Allow empty `title`, `price`, `main_category`,
    # and `store`, and normalize them to "NA".
    for idx, (field_name, value) in enumerate(zip(expected_fields, row_values)):
        if field_name in {"title", "price", "main_category", "store"}:
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

    # rating_number integer; negative values are normalized to 0
    try:
        rating_idx = expected_fields.index("rating_number")
    except ValueError:
        rating_idx = 7

    rating_num = row_values[rating_idx] if len(row_values) > rating_idx else None
    try:
        rv = int(str(rating_num).strip())
        if rv < 0:
            # normalize negative ratings to 0 (mutate mapping or positional)
            if isinstance(row, Mapping):
                try:
                    if isinstance(row, dict):
                        row[expected_fields[rating_idx]] = "0"
                except Exception:
                    pass
            else:
                row_values[rating_idx] = "0"
            rv = 0
    except Exception:
        problems.append("rating_number not integer")

    return problems


def clean_prod_csv(
    csv_path: Path | str = DEFAULT_PROD_PATH,
    schema_path: Path | str = DEFAULT_SCHEMA_PATH,
    output_path: Path | str = DEFAULT_CLEAN_PROD_PATH,
    max_workers: int | None = None,
) -> ProdCleaningResult:
    """Write a cleaned product CSV that excludes corrupted rows.

    Parses the product CSV into logical rows, pads short rows to the expected
    schema length, validates rows (optionally in parallel), sanitizes text
    fields, normalizes non-fatal issues, and writes a clean CSV ready for
    downstream consumption. Corrupted rows and reasons are exported when
    ``DEBUG_EXPORT_CORRUPTED`` is enabled.
    """
    expected_fields = load_expected_fields(schema_path)
    csv_path = Path(csv_path)
    output_path = Path(output_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Product CSV not found: {csv_path}")

    text = csv_path.read_text(encoding="utf-8", errors="replace")
    parts = text.splitlines()
    # Skip header and parse each non-empty input line. Show a single-line
    # progress update during parsing to give feedback on large files.
    rows = []
    total_lines = max(0, len(parts) - 1)
    for idx, line in enumerate(parts[1:], start=1):
        if idx % 50 == 0 or idx == total_lines:
            _print_progress(idx, total_lines, prefix="Parsing product rows")
        if not line.strip():
            continue
        try:
            parsed = next(csv.reader([line], delimiter=','))
        except Exception:
            parsed = [c for c in line.split(",")]
        # pad
        while len(parsed) < len(expected_fields):
            parsed.append("")
        rows.append(parsed)

    if not max_workers or max_workers <= 1:
        # Synchronous validation (single-process) to avoid process spawn
        # overhead for small files or when the caller explicitly requests it.
        problems_list = []
        total_rows = len(rows)
        for i, row in enumerate(rows, start=1):
            problems_list.append(detect_corrupted_row(row, expected_fields))
            if i % 50 == 0 or i == total_rows:
                _print_progress(i, total_rows, prefix="Validating product rows")
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            problems_list = []
            total_rows = len(rows)
            it = executor.map(_detect_corrupted_row_worker, zip(rows, [expected_fields] * len(rows)), chunksize=256)
            for i, p in enumerate(it, start=1):
                problems_list.append(p)
                if i % 50 == 0 or i == total_rows:
                    _print_progress(i, total_rows, prefix="Validating product rows")

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
        writer = csv.writer(out, delimiter=',')
        writer.writerow(expected_fields)
        if corrupted_handle is not None:
            corrupted_writer = csv.writer(corrupted_handle, delimiter=',')
            corrupted_writer.writerow([*expected_fields, "corruption_reasons"])

        total_rows = len(rows)
        for row, problems in zip(rows, problems_list):
            input_rows += 1
            if input_rows % 50 == 0 or input_rows == total_rows:
                _print_progress(input_rows, total_rows, prefix="Writing product output")
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

            # normalize empty title, price, main_category, and store to "NA"
            for field_name in ("title", "price", "main_category", "store"):
                try:
                    field_idx = expected_fields.index(field_name)
                except ValueError:
                    continue
                if len(row) > field_idx:
                    if row[field_idx] is None or str(row[field_idx]).strip() == "":
                        row[field_idx] = "NA"

            # Ensure rating_number is not negative — cap to 0 in the parent process
            try:
                rating_idx = expected_fields.index("rating_number")
            except ValueError:
                rating_idx = 7

            if len(row) > rating_idx:
                try:
                    rv = int(str(row[rating_idx]).strip())
                    if rv < 0:
                        row[rating_idx] = "0"
                except Exception:
                    # leave as-is; downstream validation already handled non-integers
                    pass

            writer.writerow(row[: len(expected_fields)])
            clean_rows += 1

    if corrupted_handle is not None:
        corrupted_handle.close()

    return ProdCleaningResult(input_rows=input_rows, clean_rows=clean_rows, corrupted_rows=corrupted_rows, output_path=output_path)


def main() -> None:
    # choose half of available CPU cores when possible, otherwise fallback to 1
    cpu = os.cpu_count()
    if cpu is None:
        workers = 1
    else:
        workers = max(1, cpu // 2)

    res = clean_prod_csv(max_workers=workers)
    print(f"Cleaned product data: {res.clean_rows} valid rows, {res.corrupted_rows} corrupted rows removed, output written to {res.output_path}")


if __name__ == "__main__":
    main()
