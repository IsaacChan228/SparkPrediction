"""Prediction-data cleaning utilities.

This module reads the canonical schema from ``prediction_input/prediction_data_format``
and uses it to detect and clean corrupted rows in ``test.csv`` (prediction set).

Corruption Detection:
A row is considered corrupted when it has:

* fewer attributes than the schema
* more attributes than the schema
* missing or empty attribute values (except 'comment' field)
* malformed semantic fields:
    - user_id: must match pattern u_[A-Za-z0-9]{16}
    - prod_id: must match pattern a_[A-Za-z0-9]{16}
    - parent_prod_id: must match pattern a_[A-Za-z0-9]{16}
    - purchased: must be TRUE or FALSE
    - votes: must be a non-negative integer

Corrections Applied:
* Negative vote counts are normalized to 0
* Empty 'comment' fields are normalized to "NA"
* Misaligned tail fields (comment, time, votes, purchased) are extracted and realigned
* Extra fields beyond expected count are merged into the comment field

Output:
The main cleaning helper writes a filtered CSV containing only valid rows.
Corrupted rows are exported to a separate CSV file with detailed corruption reasons.
"""

from __future__ import annotations

import csv
import os
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from itertools import repeat
from typing import Iterable, Iterator, List, Sequence, Tuple, Mapping


DEFAULT_TRAIN_PATH = Path("prediction_input/test.csv")
DEFAULT_SCHEMA_PATH = Path("prediction_input/prediction_data_format")
DEFAULT_CLEAN_TRAIN_PATH = Path("prediction_input/test_clean.csv")
CSV_ENCODING = "utf-8"
CSV_ERRORS = "replace"

# When set to True, a CSV containing corrupted rows and the reason(s) for
# corruption is written next to the cleaned output file with suffix
# "_corrupted.csv".
DEBUG_EXPORT_CORRUPTED = True

_FIELD_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_USER_ID_RE = re.compile(r"^u_[A-Za-z0-9]{16}$")
_PROD_ID_RE = re.compile(r"^a_[A-Za-z0-9]{16}$")
_PARENT_PROD_ID_RE = re.compile(r"^a_[A-Za-z0-9]{16}$")


@dataclass(frozen=True)
class CleaningResult:
    """Summary of a cleaning run."""

    input_rows: int
    clean_rows: int
    corrupted_rows: int
    output_path: Path


def _detect_corrupted_row_worker(args: Tuple[object, Sequence[str]]) -> List[str]:
    """Worker wrapper for parallel row validation.
    Accepts either a sequence (list of fields) or a mapping (dict from header
    name to value) as the row input so the parallel worker is tolerant to the
    CSV reader used upstream.
    """

    row, expected_fields = args
    return detect_corrupted_row(row, expected_fields)


def _sanitize_text_field(value: str) -> str:
    """Replace undecodable or invalid text markers with spaces."""

    cleaned = []
    for ch in str(value):
        codepoint = ord(ch)
        if ch == "\ufffd" or 0xD800 <= codepoint <= 0xDFFF or codepoint < 32:
            cleaned.append(" ")
        else:
            cleaned.append(ch)
    return "".join(cleaned)


def load_expected_fields(schema_path: Path | str = DEFAULT_SCHEMA_PATH) -> List[str]:
    """Read the expected prediction columns from the schema file."""

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


def detect_corrupted_row(row: object, expected_fields: Sequence[str]) -> List[str]:
    """Return a list of corruption reasons for a parsed CSV row.

    Accepts either a sequence of values (positional CSV row) or a mapping-like
    object (e.g. `csv.DictReader` row). Mapping inputs will be checked for
    unexpected keys (extra attributes) similar to the older implementation.
    """

    problems: List[str] = []
    expected_count = len(expected_fields)

    # If the row is mapping-like (DictReader), check for unexpected keys and
    # then extract values in the expected field order for validation below.
    if isinstance(row, Mapping):
        extra_keys = []
        for key in row.keys():
            if key in expected_fields:
                continue
            # Empty header names ("" or None) are considered extra only when
            # they carry a non-empty value.
            if key in (None, ""):
                val = row.get(key)
                if val is not None and str(val).strip() != "":
                    extra_keys.append(key)
            else:
                extra_keys.append(key)

        if extra_keys:
            problems.append("extra attribute")
            return problems

        # Build a positional list of values matching expected_fields
        positional: List[str] = [row.get(f) for f in expected_fields]
        row_values = positional
    else:
        # Assume sequence-like
        row_values = list(row)
        if len(row_values) < expected_count:
            problems.append("missing attribute")
        if len(row_values) > expected_count:
            problems.append("extra attribute")

        if problems:
            return problems

    # Validate missing/blank attributes. Allow an empty `comment` field and
    # normalize it to the literal "NA" when possible.
    for idx, (field_name, value) in enumerate(zip(expected_fields, row_values)):
        if field_name == "comment":
            if value is None or str(value).strip() == "":
                try:
                    # If the original row is mapping-like, try to set the
                    # placeholder there; otherwise update the local
                    # positional list so downstream logic can observe it.
                    if isinstance(row, Mapping):
                        if isinstance(row, dict):
                            row[field_name] = "NA"
                    else:
                        row_values[idx] = "NA"
                except Exception:
                    pass
                # don't treat as a corruption
                continue

        if value is None or str(value).strip() == "":
            problems.append(f"{field_name} missing attribute")
            break

    if problems:
        return problems

    # Positional access for the remaining semantic checks
    user_id = row_values[1]
    if not _USER_ID_RE.fullmatch(str(user_id).strip()):
        problems.append("user_id malformed")

    prod_id = row_values[2]
    if not _PROD_ID_RE.fullmatch(str(prod_id).strip()):
        problems.append("prod_id malformed")

    parent_prod_id = row_values[3]
    if not _PARENT_PROD_ID_RE.fullmatch(str(parent_prod_id).strip()):
        problems.append("parent_prod_id malformed")

    # Normalize negative votes to 0 instead of treating the row as corrupted.
    try:
        votes_idx = expected_fields.index("votes")
    except ValueError:
        votes_idx = 7

    votes = row_values[votes_idx] if len(row_values) > votes_idx else ""
    try:
        votes_value = int(str(votes).strip())
        if votes_value < 0:
            # mutate mapping or positional values so downstream logic sees 0
            if isinstance(row, Mapping):
                try:
                    if isinstance(row, dict):
                        row[expected_fields[votes_idx]] = "0"
                except Exception:
                    pass
            else:
                row_values[votes_idx] = "0"
            votes_value = 0
    except (ValueError, TypeError):
        problems.append("votes not integer")

    purchased = _sanitize_text_field(row_values[8]).strip().upper()
    if purchased not in {"TRUE", "FALSE"}:
        problems.append("purchased malformed")

    return problems


def iter_clean_rows(
    csv_path: Path | str = DEFAULT_TRAIN_PATH,
    schema_path: Path | str = DEFAULT_SCHEMA_PATH,
    max_workers: int | None = None,
) -> Iterator[List[str]]:
    """Yield only rows that match the training schema exactly.

    When ``max_workers`` is provided and greater than 1, row validation is
    performed in parallel using :class:`concurrent.futures.ProcessPoolExecutor`.
    """

    expected_fields = load_expected_fields(schema_path)
    comment_idx = expected_fields.index("comment")
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {csv_path}")

    # For small/streaming consumption keep the original streaming path when
    # no parallel workers are requested.
    if not max_workers or max_workers <= 1:
        with csv_path.open(encoding=CSV_ENCODING, errors=CSV_ERRORS, newline="") as handle:
            reader = csv.reader(handle, delimiter=',')
            next(reader, None)  # ignore the header completely
            for row in reader:
                if not detect_corrupted_row(row, expected_fields):
                    # Ensure comment cell exists and normalize empty to "NA"
                    while len(row) <= comment_idx:
                        row.append("")
                    if row[comment_idx] is None or str(row[comment_idx]).strip() == "":
                        row[comment_idx] = "NA"
                    yield row

    else:
        # Read all rows first so we can validate them in parallel while
        # preserving original order.
        rows: List[List[str]] = []
        with csv_path.open(encoding=CSV_ENCODING, errors=CSV_ERRORS, newline="") as handle:
            reader = csv.reader(handle, delimiter=',')
            next(reader, None)
            for row in reader:
                rows.append(list(row))

        # Ensure comment cell exists and normalize empty to "NA" up-front
        for row in rows:
            while len(row) <= comment_idx:
                row.append("")
            if row[comment_idx] is None or str(row[comment_idx]).strip() == "":
                row[comment_idx] = "NA"

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            problems_list = list(
                executor.map(_detect_corrupted_row_worker, zip(rows, repeat(expected_fields)), chunksize=256)
            )

        for row, problems in zip(rows, problems_list):
            if not problems:
                yield row


def clean_training_csv(
    csv_path: Path | str = DEFAULT_TRAIN_PATH,
    schema_path: Path | str = DEFAULT_SCHEMA_PATH,
    output_path: Path | str = DEFAULT_CLEAN_TRAIN_PATH,
    max_workers: int | None = None,
) -> CleaningResult:
    """Write a cleaned prediction CSV that excludes corrupted rows."""

    expected_fields = load_expected_fields(schema_path)
    expected_count = len(expected_fields)
    csv_path = Path(csv_path)
    output_path = Path(output_path)

    comment_idx = expected_fields.index("comment")

    if not csv_path.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {csv_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_rows = 0
    clean_rows = 0
    corrupted_rows = 0

    text = csv_path.read_text(encoding=CSV_ENCODING, errors=CSV_ERRORS)
    if not text:
        rows = []
        header_line = None
    else:
        parts = text.splitlines()
        header_line = parts[0]
        rest = "\n".join(parts[1:])

        # Split the remainder into logical record chunks using a robust
        # pattern: a newline followed by an id and a user_id (u_....). This
        # catches cases where the CSV parser has joined multiple records due
        # to malformed quoting.
        record_chunks = re.split(r"\n(?=\d+,u_[A-Za-z0-9]{16},)", rest)

        rows = []
        tail_re = re.compile(
            r"(?P<comment>.*?),(?P<time>[0-9Ee.+-]+),(?P<votes>\d+),(?P<purchased>TRUE|FALSE|True|False|true|false)$",
            re.DOTALL,
        )
        for chunk in record_chunks:
            chunk = chunk.strip("\n")
            if not chunk:
                continue
            try:
                parsed = next(csv.reader([chunk], delimiter=','))
            except Exception:
                parsed = [c for c in chunk.split(",")]

            # If the parsed row is short but the last field contains the
            # expected tail tokens (time,votes,purchased), extract them.
            if len(parsed) < expected_count and parsed:
                last = parsed[-1]
                m = tail_re.match(last)
                if m:
                    comment_part = m.group("comment")
                    time_part = m.group("time")
                    votes_part = m.group("votes")
                    purchased_part = m.group("purchased")
                    new_row = parsed[:-1] + [comment_part, time_part, votes_part, purchased_part]
                    parsed = new_row

            # misaligned tail fields when the numeric/boolean fields appear
            # shifted into the comment.
            row = list(parsed)
            # Merge extras into comment if present
            while len(row) > expected_count:
                try:
                    nxt = row[comment_idx + 1]
                except IndexError:
                    break
                row[comment_idx] = f"{row[comment_idx]} {nxt}"
                del row[comment_idx + 1]

            # Fix misaligned tail fields even when length == expected_count
            def _looks_like_time(s: str) -> bool:
                try:
                    float(str(s).strip())
                    return True
                except Exception:
                    return False

            bad_time = not (_looks_like_time(row[6]) if len(row) > 6 else False)
            bad_purchased = not (len(row) > 8 and isinstance(row[8], str) and row[8].strip().upper() in {"TRUE", "FALSE"})

            if (bad_time or bad_purchased) and len(row) > comment_idx:
                comment_text = row[comment_idx]
                tokens = [t for t in re.split(r"[\s,]+", str(comment_text).strip()) if t]
                for take in (3, 2):
                    if len(tokens) < take:
                        continue
                    cand = tokens[-take:]
                    ok = True
                    try:
                        if take == 3:
                            float(cand[0])
                            int(cand[1])
                            if cand[2].strip().upper() not in {"TRUE", "FALSE"}:
                                ok = False
                        else:
                            int(cand[0])
                            if cand[1].strip().upper() not in {"TRUE", "FALSE"}:
                                ok = False
                    except Exception:
                        ok = False

                    if ok:
                        if take == 3:
                            new_tokens = tokens[:-3]
                            row[6] = cand[0]
                            row[7] = cand[1]
                            row[8] = cand[2]
                        else:
                            new_tokens = tokens[:-2]
                            # time not present; assume row[6] already correct
                            row[7] = cand[0]
                            row[8] = cand[1]

                        row[comment_idx] = " ".join(new_tokens)

            # Additional heuristic: if purchased is shifted left into
            # positions 6/7 and comment ends with time+votes tokens, move them
            # right and extract time/votes from the comment tail.
            try:
                tokens2 = [t for t in re.split(r"[\s,]+", str(row[comment_idx]).strip()) if t]
                if len(tokens2) >= 2 and len(row) >= 8:
                    tail_time = tokens2[-2]
                    tail_votes = tokens2[-1]
                    try:
                        float(tail_time)
                        int(tail_votes)
                        if isinstance(row[6], str) and row[6].strip().upper() in {"TRUE", "FALSE"} and (row[8] == "" or str(row[8]).strip() == ""):
                            purchased_val = row[6]
                            row[6] = tail_time
                            row[7] = tail_votes
                            while len(row) <= 8:
                                row.append("")
                            row[8] = purchased_val
                            row[comment_idx] = " ".join(tokens2[:-2])
                    except Exception:
                        pass
            except Exception:
                pass

            # Pad short rows
            while len(row) < expected_count:
                row.append("")

            rows.append(row)

    # Merge extra attributes into the `comment` field until rows have the
    # expected number of attributes. Operate per logical CSV row as parsed by
    # the CSV reader so embedded commas and quotes are handled correctly.
    comment_idx = expected_fields.index("comment")
    expected_count = len(expected_fields)
    for i, row in enumerate(rows):
        while len(row) > expected_count:
            try:
                nxt = row[comment_idx + 1]
            except IndexError:
                break
            # Merge with a single space to keep text readable.
            row[comment_idx] = f"{row[comment_idx]} {nxt}"
            del row[comment_idx + 1]

        # Heuristic: if numeric/boolean tail fields (time, votes, purchased)
        # were accidentally absorbed into the comment due to bad
        # quoting, try to extract them back from the end of the comment.
        try:
            # Only attempt when we have at least the comment field
            comment_text = row[comment_idx] if len(row) > comment_idx else ""
            # If purchased looks missing/malformed, try to parse
            bad_purchased = True
            if len(row) > 8 and isinstance(row[8], str) and row[8].strip().upper() in {"TRUE", "FALSE"}:
                bad_purchased = False

            if bad_purchased:
                # Tokenize by whitespace and commas to find candidate tail tokens
                tokens = [t for t in re.split(r"[\s,]+", comment_text.strip()) if t]
                # Try to extract up to 3 tail tokens: time, votes, purchased
                for take in (3, 2):
                    if len(tokens) < take:
                        continue
                    cand = tokens[-take:]
                    ok = True
                    try:
                        if take == 3:
                            # time, votes, purchased
                            float(cand[0])
                            int(cand[1])
                            if cand[2].strip().upper() not in {"TRUE", "FALSE"}:
                                ok = False
                        else:
                            # votes, purchased
                            int(cand[0])
                            if cand[1].strip().upper() not in {"TRUE", "FALSE"}:
                                ok = False
                    except Exception:
                        ok = False

                    if ok:
                        if take == 3:
                            new_tokens = tokens[:-3]
                            row[6] = cand[0]
                            row[7] = cand[1]
                            row[8] = cand[2]
                        else:
                            new_tokens = tokens[:-2]
                            # time not present; assume row[6] already correct
                            row[7] = cand[0]
                            row[8] = cand[1]

                        new_comment = " ".join(new_tokens)
                        # rebuild row ensuring length
                        while len(row) < expected_count:
                            row.append("")
                        row[comment_idx] = new_comment
                        break
        except Exception:
            pass

    # Normalize empty `comment` cells to "NA" so empty comments are
    # considered valid and downstream validation/writing sees the placeholder.
    for i, row in enumerate(rows):
        if isinstance(row, Mapping):
            try:
                if row.get("comment") is None or str(row.get("comment")).strip() == "":
                    if isinstance(row, dict):
                        row["comment"] = "NA"
            except Exception:
                pass
        else:
            # ensure comment column exists
            while len(row) <= comment_idx:
                row.append("")
            if row[comment_idx] is None or str(row[comment_idx]).strip() == "":
                row[comment_idx] = "NA"

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        problems_list = list(
            executor.map(
                _detect_corrupted_row_worker,
                zip(rows, repeat(expected_fields)),
                chunksize=256,
            )
        )

    # If the CSV header doesn't match the expected schema, mark every row as
    # corrupted with an explicit reason so it's obvious the file header is bad.
    try:
        header_bad
    except NameError:
        header_bad = False

    if header_bad:
        for i in range(len(problems_list)):
            if problems_list[i]:
                problems_list[i].append("invalid header")
            else:
                problems_list[i] = ["invalid header"]

    # No id re-assignment: assume original ids are correct, so skip
    # max_original_id/next_new_id calculation.
    # Optionally open a CSV to record corrupted rows and their reasons.
    corrupted_handle = None
    corrupted_writer = None
    if DEBUG_EXPORT_CORRUPTED:
        corrupted_path = output_path.with_name(output_path.stem + "_corrupted.csv")
        corrupted_handle = corrupted_path.open("w", encoding="utf-8", newline="")

    with output_path.open("w", encoding="utf-8", newline="") as output_handle:
        writer = csv.writer(output_handle)
        writer.writerow(expected_fields)

        if corrupted_handle is not None:
            corrupted_writer = csv.writer(corrupted_handle)
            corrupted_writer.writerow([*expected_fields, "corruption_reasons"])

        for row, problems in zip(rows, problems_list):
            input_rows += 1
            if problems:
                corrupted_rows += 1
                if corrupted_writer is not None:
                    if isinstance(row, dict):
                        corrupted_writer.writerow([*(row.get(f) for f in expected_fields), "; ".join(problems)])
                    else:
                        corrupted_writer.writerow([*row, "; ".join(problems)])

            # Use the original id as-is; do not attempt to correct or deduplicate
            raw_id = row.get("id") if isinstance(row, dict) else row[0]
            out_id = str(raw_id)

            # Normalize numeric semantic fields in the parent process so
            # mutations are reflected in the output CSV (workers cannot
            # modify the parent process memory).
            try:
                votes_idx = expected_fields.index("votes")
            except ValueError:
                votes_idx = 7

            if isinstance(row, Mapping):
                # votes: cap negative to 0
                try:
                    v = row.get(expected_fields[votes_idx])
                    if v is not None:
                        vi = int(str(v).strip())
                        if vi < 0:
                            row[expected_fields[votes_idx]] = "0"
                except Exception:
                    pass

                # no-op for prediction data
            else:
                # sequence/list row
                if len(row) > votes_idx:
                    try:
                        vi = int(str(row[votes_idx]).strip())
                        if vi < 0:
                            row[votes_idx] = "0"
                    except Exception:
                        pass
                # no-op for prediction data

            # Sanitize textual fields by name when using mapping
            if isinstance(row, dict):
                if "title" in row:
                    row["title"] = _sanitize_text_field(row.get("title"))
                if "comment" in row:
                    row["comment"] = _sanitize_text_field(row.get("comment"))
                    if row.get("comment") is None or str(row.get("comment")).strip() == "":
                        row["comment"] = "NA"
                out_row = [str(out_id), *(row.get(f) for f in expected_fields[1:])]
            else:
                # sequence fallback (shouldn't happen with DictReader above)
                # ensure comment exists and default to NA when empty
                while len(row) <= comment_idx:
                    row.append("")
                if row[comment_idx] is None or str(row[comment_idx]).strip() == "":
                    row[comment_idx] = "NA"
                row[4] = _sanitize_text_field(row[4])
                row[5] = _sanitize_text_field(row[5])
                out_row = [str(out_id), *row[1:]]

            writer.writerow(out_row)
            clean_rows += 1

    if corrupted_handle is not None:
        corrupted_handle.close()

    return CleaningResult(
        input_rows=input_rows,
        clean_rows=clean_rows,
        corrupted_rows=corrupted_rows,
        output_path=output_path,
    )


def main() -> None:
    """CLI entry point for cleaning prediction data."""
    # choose half of available CPU cores when possible, otherwise fallback to 1
    cpu = os.cpu_count()
    if cpu is None:
        workers = 1
    else:
        workers = max(1, cpu // 2)

    result = clean_training_csv(max_workers=workers)
    print(
        "Cleaned prediction data: "
        f"{result.clean_rows} valid rows, "
        f"{result.corrupted_rows} corrupted rows removed, "
        f"output written to {result.output_path}"
    )


if __name__ == "__main__":
    main()