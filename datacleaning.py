"""Utilities for detecting corrupted rows in the training CSV.

Functions expect a mapping-like row (e.g. csv.DictReader row, dict or pyspark Row
converted to dict). The main function `detect_corrupted_row` returns a list of
problem descriptions (empty list means the row appears valid).
"""
import re
import time
import csv
import os
from typing import Mapping, Iterable, List, Tuple


_USER_ID_RE = re.compile(r"^u_[A-Za-z0-9]{16}$")
_PROD_ID_RE = re.compile(r"^a_[A-Za-z0-9]{16}$")


def corrupted_data_handling(csv_path: str = "training_data/train.csv",
                            output_path: str = "prediction_output/corrupt_data.txt",
                            max_write: int = None) -> int:
    """Read a CSV file, detect corrupted rows, and write them to a text file.

    Returns the total number of rows detected as corrupted.

    Args:
        csv_path: Input CSV file path.
        output_path: Output text file path (will be overwritten).
        max_write: If specified, limit the number of corrupted rows written.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    written = 0
    total_corrupt = 0
    with open(csv_path, encoding="utf-8") as inf, open(output_path, "w", encoding="utf-8") as outf:
        reader = csv.DictReader(inf)
        for i, row in enumerate(reader):
            problems = detect_corrupted_row(row)
            if problems:
                total_corrupt += 1
                if max_write is None or written < max_write:
                    # human-readable line
                    preview = {
                        "id": row.get("id"),
                        "user_id": row.get("user_id"),
                        "prod_id": row.get("prod_id"),
                        "rating": row.get("rating"),
                        "votes": row.get("votes"),
                        "time": row.get("time"),
                        "purchased": row.get("purchased")
                    }
                    outf.write(f"Index: {i}; Id: {row.get('id')}; Problems: {', '.join(problems)}\n")
                    outf.write(f"Preview: {preview}\n")
                    outf.write("---\n")
                    written += 1
    return total_corrupt

def detect_corrupted_row(row: Mapping) -> List[str]:
    """Check a single row for invalid/corrupted fields.

    Validation is performed according to the README field descriptions.

    Returns a list of problem descriptions; empty list means the row appears valid.

    Args:
        row: A dict-like object supporting row.get('field').
    """
    problems: List[str] = []

    # If any cell in the row is missing or empty, mark the row corrupted.
    for key in row:
        val = row.get(key)
        if val is None or (isinstance(val, str) and val.strip() == ""):
            problems.append(f"{key} missing/empty")
    if problems:
        return problems

    # id: numeric
    _id = row.get("id")
    try:
        int(_id)
    except Exception:
        problems.append("id not integer")

    # user_id
    user_id = row.get("user_id")
    if not user_id or not _USER_ID_RE.match(user_id):
        problems.append("user_id malformed")

    # prod_id, parent_prod_id
    for fld in ("prod_id", "parent_prod_id"):
        v = row.get(fld)
        if not v or not _PROD_ID_RE.match(v):
            problems.append(f"{fld} malformed")

    # time: require a non-negative integer
    t = row.get("time")
    try:
        t_val = int(t)
        if t_val < 0:
            problems.append("time negative")
    except Exception:
        problems.append("time not integer")

    # votes: non-negative integer
    votes = row.get("votes")
    try:
        v = int(votes)
        if v < 0:
            problems.append("votes negative")
    except Exception:
        problems.append("votes not integer")

    # purchased: must be the exact strings "TRUE" or "FALSE"
    purchased = row.get("purchased")
    if not (isinstance(purchased, str) and purchased.strip() in ("TRUE", "FALSE")):
        problems.append("purchased not TRUE/FALSE")

    # rating: integer 1..5
    rating = row.get("rating")
    try:
        r = int(rating)
        if r < 1 or r > 5:
            problems.append("rating out of 1..5")
    except Exception:
        problems.append("rating not integer")

    # title/comment: ensure present (optional) and remove obviously binary / non-text
    title = row.get("title")
    comment = row.get("comment")
    for fld, val in (("title", title), ("comment", comment)):
        if val is None:
            # allow missing comments/titles but flag it
            problems.append(f"{fld} missing")
        else:
            # disallow control characters except common whitespace
            if any(ord(ch) < 9 for ch in str(val)):
                problems.append(f"{fld} contains control chars")

    return problems


def detect_corrupted_rows(rows: Iterable[Mapping]) -> List[Tuple[int, List[str]]]:
    """Check a sequence of rows and return a list of (index, problems).

    Index is the 0-based position in the input sequence.
    """
    out: List[Tuple[int, List[str]]] = []
    for i, row in enumerate(rows):
        problems = detect_corrupted_row(row)
        if problems:
            out.append((i, problems))
    return out


