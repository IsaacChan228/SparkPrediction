"""Training-data cleaning utilities.

This module reads the canonical schema from ``training_data/training_data_format`` 
and uses it to detect and clean corrupted rows in ``train.csv``.

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
  - rating: must be an integer between 1 and 5
  - votes: must be a non-negative integer

Corrections Applied:
* Negative vote counts are normalized to 0
* Empty 'comment' fields are normalized to "NA"
* Misaligned tail fields (comment, time, votes, purchased, rating) are extracted and realigned
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


DEFAULT_TRAIN_PATH = Path("training_data/train.csv")
DEFAULT_SCHEMA_PATH = Path("training_data/training_data_format")
DEFAULT_CLEAN_TRAIN_PATH = Path("training_data/train_clean.csv")
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

    rating = row_values[9]
    try:
        rating_value = int(str(rating).strip())
        if rating_value < 1 or rating_value > 5:
            problems.append("rating out of range")
    except (ValueError, TypeError):
        problems.append("rating not integer")

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
        raise FileNotFoundError(f"Training CSV not found: {csv_path}")

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
    """Write a cleaned training CSV that excludes corrupted rows."""

    expected_fields = load_expected_fields(schema_path)
    expected_count = len(expected_fields)
    csv_path = Path(csv_path)
    output_path = Path(output_path)

    comment_idx = expected_fields.index("comment")

    if not csv_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {csv_path}")

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
            r"(?P<comment>.*?),(?P<time>[0-9Ee.+-]+),(?P<votes>\d+),(?P<purchased>TRUE|FALSE|True|False|true|false),(?P<rating>\d+)$",
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
            # expected tail tokens (time,votes,purchased,rating), extract them.
            if len(parsed) < expected_count and parsed:
                last = parsed[-1]
                m = tail_re.match(last)
                if m:
                    comment_part = m.group("comment")
                    time_part = m.group("time")
                    votes_part = m.group("votes")
                    purchased_part = m.group("purchased")
                    rating_part = m.group("rating")
                    new_row = parsed[:-1] + [comment_part, time_part, votes_part, purchased_part, rating_part]
                    parsed = new_row

            # Ensure we have exactly expected_count or attempt to repair
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
            bad_rating = True
            if len(row) > 9:
                try:
                    rv = int(str(row[9]).strip())
                    if 1 <= rv <= 5:
                        bad_rating = False
                except Exception:
                    bad_rating = True

            if (bad_time or bad_purchased or bad_rating) and len(row) > comment_idx:
                comment_text = row[comment_idx]
                tokens = [t for t in re.split(r"[\s,]+", str(comment_text).strip()) if t]
                for take in (4, 3):
                    if len(tokens) < take:
                        continue
                    cand = tokens[-take:]
                    ok = True
                    try:
                        if take == 4:
                            float(cand[0])
                            int(cand[1])
                            if cand[2].strip().upper() not in {"TRUE", "FALSE"}:
                                ok = False
                            rint = int(cand[3])
                            if not (1 <= rint <= 5):
                                ok = False
                        else:
                            int(cand[0])
                            if cand[1].strip().upper() not in {"TRUE", "FALSE"}:
                                ok = False
                            rint = int(cand[2])
                            if not (1 <= rint <= 5):
                                ok = False
                    except Exception:
                        ok = False

                    if ok:
                        if take == 4:
                            new_tokens = tokens[:-4]
                            row[6] = cand[0]
                            row[7] = cand[1]
                            row[8] = cand[2]
                            row[9] = cand[3]
                        else:
                            new_tokens = tokens[:-3]
                            # time not present; assume row[6] already correct
                            row[7] = cand[0]
                            row[8] = cand[1]
                            row[9] = cand[2]

                        row[comment_idx] = " ".join(new_tokens)

            # Additional heuristic: if purchased/rating are shifted left into
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
                            rating_val = row[7] if len(row) > 7 else ""
                            row[6] = tail_time
                            row[7] = tail_votes
                            while len(row) <= 9:
                                row.append("")
                            row[8] = purchased_val
                            row[9] = rating_val
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

        # Heuristic: if numeric/boolean tail fields (time, votes, purchased,
        # rating) were accidentally absorbed into the comment due to bad
        # quoting, try to extract them back from the end of the comment.
        try:
            # Only attempt when we have at least the comment field
            comment_text = row[comment_idx] if len(row) > comment_idx else ""
            # If purchased or rating look missing/malformed, try to parse
            bad_purchased = True
            bad_rating = True
            if len(row) > 8 and isinstance(row[8], str) and row[8].strip().upper() in {"TRUE", "FALSE"}:
                bad_purchased = False
            if len(row) > 9:
                try:
                    rv = int(str(row[9]).strip())
                    if 1 <= rv <= 5:
                        bad_rating = False
                except Exception:
                    bad_rating = True

            if bad_purchased or bad_rating:
                # Tokenize by whitespace and commas to find candidate tail tokens
                tokens = [t for t in re.split(r"[\s,]+", comment_text.strip()) if t]
                # Try to extract up to 4 tail tokens: time, votes, purchased, rating
                for take in (4, 3):
                    if len(tokens) < take:
                        continue
                    cand = tokens[-take:]
                    # Map cand from left->right to time,votes,purchased,rating
                    # depending on take
                    ok = True
                    cand_time = None
                    cand_votes = None
                    cand_purchased = None
                    cand_rating = None
                    try:
                        if take == 4:
                            cand_time = cand[0]
                            cand_votes = cand[1]
                            cand_purchased = cand[2]
                            cand_rating = cand[3]
                        elif take == 3:
                            # maybe time merged; assume order votes,purchased,rating
                            cand_votes = cand[0]
                            cand_purchased = cand[1]
                            cand_rating = cand[2]

                        # validate
                        if cand_time is not None:
                            # allow scientific notation or integer
                            float(cand_time)
                        if cand_votes is not None:
                            int(cand_votes)
                        if cand_purchased is not None:
                            if str(cand_purchased).strip().upper() not in {"TRUE", "FALSE"}:
                                ok = False
                        if cand_rating is not None:
                            rint = int(cand_rating)
                            if not (1 <= rint <= 5):
                                ok = False
                    except Exception:
                        ok = False

                    if ok:
                        # Apply extracted values to row positions, trimming the
                        # comment accordingly.
                        if take == 4:
                            # remove last 4 tokens from comment_text
                            new_tokens = tokens[:-4]
                        else:
                            new_tokens = tokens[:-3]

                        new_comment = " ".join(new_tokens)
                        # rebuild row ensuring length
                        # ensure row has at least expected_count elements
                        while len(row) < expected_count:
                            row.append("")
                        row[comment_idx] = new_comment
                        if take == 4:
                            row[6] = cand_time
                            row[7] = cand_votes
                            row[8] = cand_purchased
                            row[9] = cand_rating
                        else:
                            # no time extracted; shift accordingly
                            row[7] = cand_votes
                            row[8] = cand_purchased
                            row[9] = cand_rating
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

    max_original_id = 0
    for row, problems in zip(rows, problems_list):
        if problems:
            continue
        try:
            raw_id = row.get("id") if isinstance(row, dict) else row[0]
            row_id = int(raw_id)
        except (ValueError, TypeError, AttributeError):
            continue
        if row_id > max_original_id:
            max_original_id = row_id

    next_new_id = max_original_id + 1
    seen_original_ids: set[int] = set()
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
                continue

            # Extract id for writing and duplicate detection
            try:
                raw_id = row.get("id") if isinstance(row, dict) else row[0]
                orig_id = int(raw_id)
            except (ValueError, TypeError, AttributeError):
                corrupted_rows += 1
                if corrupted_writer is not None:
                    # write a best-effort preview of the row
                    if isinstance(row, dict):
                        corrupted_writer.writerow([*(row.get(f) for f in expected_fields), "id not integer"])
                    else:
                        corrupted_writer.writerow([*row, "id not integer"])
                continue

            if orig_id in seen_original_ids:
                out_id = next_new_id
                next_new_id += 1
            else:
                out_id = orig_id

            seen_original_ids.add(orig_id)

            # Normalize numeric semantic fields in the parent process so
            # mutations are reflected in the output CSV (workers cannot
            # modify the parent process memory).
            try:
                votes_idx = expected_fields.index("votes")
            except ValueError:
                votes_idx = 7
            try:
                rating_idx = expected_fields.index("rating")
            except ValueError:
                rating_idx = 9

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

                # rating: cap to [1,5]
                try:
                    r = row.get(expected_fields[rating_idx])
                    if r is not None:
                        ri = int(str(r).strip())
                        if ri < 1:
                            row[expected_fields[rating_idx]] = "1"
                        elif ri > 5:
                            row[expected_fields[rating_idx]] = "5"
                except Exception:
                    pass
            else:
                # sequence/list row
                if len(row) > votes_idx:
                    try:
                        vi = int(str(row[votes_idx]).strip())
                        if vi < 0:
                            row[votes_idx] = "0"
                    except Exception:
                        pass
                if len(row) > rating_idx:
                    try:
                        ri = int(str(row[rating_idx]).strip())
                        if ri < 1:
                            row[rating_idx] = "1"
                        elif ri > 5:
                            row[rating_idx] = "5"
                    except Exception:
                        pass

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
    """CLI entry point for cleaning training data."""
    # choose half of available CPU cores when possible, otherwise fallback to 1
    cpu = os.cpu_count()
    if cpu is None:
        workers = 1
    else:
        workers = max(1, cpu // 2)

    result = clean_training_csv(max_workers=workers)
    print(
        "Cleaned training data: "
        f"{result.clean_rows} valid rows, "
        f"{result.corrupted_rows} corrupted rows removed, "
        f"output written to {result.output_path}"
    )


if __name__ == "__main__":
    main()