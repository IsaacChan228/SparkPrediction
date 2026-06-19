import pandas as pd
from collections import Counter
from pathlib import Path

path = Path('training_data/train.csv')
if not path.exists():
    print(f"ERROR: {path} not found")
    raise SystemExit(1)

chunksize = 1000000
total = 0
nulls = 0
counter = Counter()
for chunk in pd.read_csv(path, usecols=['rating'], chunksize=chunksize):
    total += len(chunk)
    nulls += chunk['rating'].isna().sum()
    vals = chunk['rating'].dropna()
    # convert numeric to int-like strings if float like 4.0
    vals = vals.apply(lambda x: int(x) if (isinstance(x, float) and x.is_integer()) else x)
    counter.update(vals.astype(str).tolist())

print(f"Total rows: {total}")
print(f"Null rating count: {nulls}")
print("Rating distribution:")
for k, v in counter.most_common():
    print(f"  {k}: {v}")
