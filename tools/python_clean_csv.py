import csv
from pathlib import Path

IN = Path('training_data/train_merged.csv')
OUT = Path('training_data/train_merged_pyclean.csv')
LOG = Path('artifacts/problematic_rows_python.txt')

if not IN.exists():
    print(f"Input file not found: {IN}")
    raise SystemExit(1)

with IN.open('r', encoding='utf-8', errors='replace', newline='') as inf:
    reader = csv.reader(inf, delimiter=',', quotechar='"', escapechar='\\', doublequote=True)
    header = next(reader)
    ncols = len(header)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)

    with OUT.open('w', encoding='utf-8', newline='') as outf, LOG.open('w', encoding='utf-8') as logf:
        writer = csv.writer(outf, delimiter=',', quotechar='"', escapechar='\\', doublequote=True, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(header)

        total = 0
        skipped = 0
        fixed = 0
        samples = 0
        for row in reader:
            total += 1
            if len(row) == ncols:
                writer.writerow(row)
            elif len(row) > ncols:
                # Merge extra columns into last column
                fixed_row = row[:ncols-1] + [','.join(row[ncols-1:])]
                writer.writerow(fixed_row)
                fixed += 1
                if samples < 20:
                    logf.write(f"FIXED: original_len={len(row)} fixed_len={len(fixed_row)} row={row}\n")
                    samples += 1
            else:
                # too few columns: skip and log
                skipped += 1
                if samples < 20:
                    logf.write(f"SKIPPED: original_len={len(row)} row={row}\n")
                    samples += 1

print(f"Processed {total} rows; fixed={fixed}, skipped={skipped}; output={OUT}; log={LOG}")
