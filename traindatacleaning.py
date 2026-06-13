"""Training-data cleaning utilities.

This module reads the canonical schema from
``training_data/training_data_format`` and uses it to detect corrupted rows in
``train.csv``. A row is considered corrupted when it has:

* fewer attributes than the schema
* more attributes than the schema
* any empty attribute value

When duplicate ``id`` values appear, the first row keeps its original ``id``
and later rows are reassigned new numeric ``id`` values from the end of the
original id range so normal records keep their ids unchanged.

The main cleaning helper can be used before training to write a filtered CSV
containing only valid rows.
"""

from __future__ import annotations

import csv
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from itertools import repeat
from typing import Iterable, Iterator, List, Sequence, Tuple


DEFAULT_TRAIN_PATH = Path("training_data/train.csv")
DEFAULT_SCHEMA_PATH = Path("training_data/training_data_format")
DEFAULT_CLEAN_TRAIN_PATH = Path("training_data/train_clean.csv")
CSV_ENCODING = "latin1"

_FIELD_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class CleaningResult:
    """Summary of a cleaning run."""

    input_rows: int
    clean_rows: int
    corrupted_rows: int
    output_path: Path


def _detect_corrupted_row_worker(args: Tuple[Sequence[str], Sequence[str]]) -> List[str]:
    """Worker wrapper for parallel row validation."""

    row, expected_fields = args
    return detect_corrupted_row(row, expected_fields)


def load_expected_fields(schema_path: Path | str = DEFAULT_SCHEMA_PATH) -> List[str]:
    """Read the expected training columns from the schema file."""

    schema_path = Path(schema_path)
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    fields: List[str] = []
    with schema_path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or ":" not in line:
                continue

            field_name = line.split(":", 1)[0].strip()
            if _FIELD_NAME_RE.fullmatch(field_name):
                fields.append(field_name)

    if not fields:
        raise ValueError(f"No training fields found in schema file: {schema_path}")

    return fields


def detect_corrupted_row(row: Sequence[str], expected_fields: Sequence[str]) -> List[str]:
    """Return a list of corruption reasons for a parsed CSV row.

    The row is treated as corrupted when its attribute count does not match the
    schema or any attribute is blank.
    """

    problems: List[str] = []
    expected_count = len(expected_fields)

    if len(row) < expected_count:
        problems.append("missing attribute")
    if len(row) > expected_count:
        problems.append("extra attribute")

    if problems:
        return problems

    for field_name, value in zip(expected_fields, row):
        if value is None or str(value).strip() == "":
            problems.append(f"{field_name} missing attribute")
            break

    return problems


def iter_clean_rows(
    csv_path: Path | str = DEFAULT_TRAIN_PATH,
    schema_path: Path | str = DEFAULT_SCHEMA_PATH,
) -> Iterator[List[str]]:
    """Yield only rows that match the training schema exactly."""

    expected_fields = load_expected_fields(schema_path)
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {csv_path}")

    with csv_path.open(encoding=CSV_ENCODING, newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)  # ignore the header completely
        for row in reader:
            if not detect_corrupted_row(row, expected_fields):
                yield row


def clean_training_csv(
    csv_path: Path | str = DEFAULT_TRAIN_PATH,
    schema_path: Path | str = DEFAULT_SCHEMA_PATH,
    output_path: Path | str = DEFAULT_CLEAN_TRAIN_PATH,
    max_workers: int | None = None,
) -> CleaningResult:
    """Write a cleaned training CSV that excludes corrupted rows."""

    expected_fields = load_expected_fields(schema_path)
    csv_path = Path(csv_path)
    output_path = Path(output_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {csv_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_rows = 0
    clean_rows = 0
    corrupted_rows = 0

    with csv_path.open(encoding=CSV_ENCODING, newline="") as input_handle:
        reader = csv.reader(input_handle)
        next(reader, None)  # skip header; schema comes from training_data_format
        rows = list(reader)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        problems_list = list(
            executor.map(
                _detect_corrupted_row_worker,
                zip(rows, repeat(expected_fields)),
                chunksize=256,
            )
        )

    max_original_id = 0
    for row, problems in zip(rows, problems_list):
        if problems:
            continue
        try:
            row_id = int(row[0])
        except (ValueError, TypeError):
            continue
        if row_id > max_original_id:
            max_original_id = row_id

    next_new_id = max_original_id + 1
    seen_original_ids: set[int] = set()

    with output_path.open("w", encoding="utf-8", newline="") as output_handle:
        writer = csv.writer(output_handle)
        writer.writerow(expected_fields)

        for row, problems in zip(rows, problems_list):
            input_rows += 1
            if problems:
                corrupted_rows += 1
                continue

            try:
                row_id = int(row[0])
            except (ValueError, TypeError):
                corrupted_rows += 1
                continue

            if row_id in seen_original_ids:
                row_id = next_new_id
                next_new_id += 1

            seen_original_ids.add(int(row[0]))
            row = [str(row_id), *row[1:]]

            writer.writerow(row)
            clean_rows += 1

    return CleaningResult(
        input_rows=input_rows,
        clean_rows=clean_rows,
        corrupted_rows=corrupted_rows,
        output_path=output_path,
    )


def main() -> None:
    """CLI entry point for cleaning training data."""

    result = clean_training_csv()
    print(
        "Cleaned training data: "
        f"{result.clean_rows} valid rows, "
        f"{result.corrupted_rows} corrupted rows removed, "
        f"output written to {result.output_path}"
    )


if __name__ == "__main__":
    main()