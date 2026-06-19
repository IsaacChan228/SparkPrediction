from pyspark.sql import SparkSession, functions as F
from pathlib import Path

spark = SparkSession.builder.appName("collect-problematic-ratings").getOrCreate()
IN = 'training_data/train_merged.csv'
OUT_DIR = Path('artifacts')
OUT_DIR.mkdir(exist_ok=True)
OUT = OUT_DIR / 'problematic_ratings.txt'

df = (
    spark.read.option('header', True)
    .option('nullValue', 'NA')
    .option('treatEmptyValuesAsNulls', 'true')
    .csv(IN)
)

rating_clean = F.when(
    F.regexp_replace(F.col('rating'), '[^0-9.]', '').rlike(r'^[1-5](\\.0+)?$'),
    F.regexp_replace(F.col('rating'), '[^0-9.]', '').cast('double').cast('int'),
).otherwise(F.lit(None))

prob = df.select(F.col('rating').alias('raw_rating'), F.col('comment'), F.col('votes'), F.col('time'), rating_clean.alias('clean_rating'))
prob = prob.where((F.col('raw_rating').isNotNull()) & (F.col('clean_rating').isNull())).limit(10)
rows = prob.collect()

with OUT.open('w', encoding='utf-8') as fh:
    for r in rows:
        # convert Row to dict-like string
        fh.write(str({k: r[k] for k in r.__fields__}) + '\n')

print(f'Wrote {len(rows)} rows to {OUT}')

spark.stop()
