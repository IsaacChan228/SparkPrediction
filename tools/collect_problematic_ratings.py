import re
import pandas as pd
from pathlib import Path

IN = Path('training_data/train_merged.csv')
OUT_DIR = Path('artifacts')
OUT_DIR.mkdir(exist_ok=True)
OUT = OUT_DIR / 'problematic_ratings.txt'

if not IN.exists():
    print(f"Input file {IN} not found")
    raise SystemExit(1)

pattern = re.compile(r'^[1-5](\.0+)?$')
found = 0
chunksize = 200000
with OUT.open('w', encoding='utf-8') as fh:
    for chunk in pd.read_csv(IN, dtype=str, chunksize=chunksize):
        if 'rating' not in chunk.columns:
            continue
        for _, row in chunk.iterrows():
            raw = row.get('rating')
            if pd.isna(raw) or str(raw).strip() == '':
                continue
            cleaned = re.sub(r'[^0-9.]', '', str(raw))
            if not pattern.match(cleaned):
                # write a simple representation of the row
                fh.write(str(row.to_dict()) + '\n')
                found += 1
                if found >= 10:
                    break
        if found >= 10:
            break
print(f"Wrote {found} problematic rows to {OUT}")
