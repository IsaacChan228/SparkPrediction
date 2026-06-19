import csv
import sys
from pathlib import Path

INPUT = sys.argv[1] if len(sys.argv) > 1 else "training_data/train_merged.csv"
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else None
OUT = Path("artifacts/problematic_rows_column_mismatch.txt")
MAX_SAVE = 200

print(f"Checking CSV columns for: {INPUT}")
with open(INPUT, "r", encoding="utf-8", errors="replace", newline="") as f:
    reader = csv.reader(f, delimiter=",", quotechar='"', escapechar='\\')
    try:
        header = next(reader)
    except Exception as e:
        print("Failed to read header:", e)
        sys.exit(2)
    expected = len(header)
    print(f"Expected columns: {expected}")

    bad_count = 0
    saved = []
    idx = 1
    for row in reader:
        idx += 1
        if LIMIT and idx > LIMIT:
            break
        try:
            if len(row) != expected:
                bad_count += 1
                if len(saved) < MAX_SAVE:
                    saved.append((idx, len(row), row))
        except Exception as e:
            bad_count += 1
            if len(saved) < MAX_SAVE:
                saved.append((idx, -1, [f"PARSE_ERROR: {e}"]))

    print(f"Total rows checked: {idx}")
    print(f"Rows with column count != {expected}: {bad_count}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as out:
        out.write(f"Checked file: {INPUT}\n")
        out.write(f"Expected columns: {expected}\n")
        out.write(f"Total rows checked: {idx}\n")
        out.write(f"Bad rows: {bad_count}\n\n")
        for i, c, r in saved:
            out.write(f"ROW {i} (cols={c}): {r}\n")

print(f"Wrote first {len(saved)} problematic rows to {OUT}")
