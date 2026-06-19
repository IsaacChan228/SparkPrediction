from pyspark.sql import SparkSession, functions as F
from pathlib import Path

spark = SparkSession.builder.appName("inspect-problematic-row").getOrCreate()
IN = 'training_data/train_merged.csv'
OUT = Path('artifacts/problematic_row_inspect.txt')
OUT.parent.mkdir(exist_ok=True)

# read with strict CSV options
df = (
    spark.read.option('header', True)
    .option('nullValue', 'NA')
    .option('treatEmptyValuesAsNulls', 'true')
    .option('encoding', 'UTF-8')
    .option('sep', ',')
    .option('quote', '"')
    .option('escape', '\\')
    .option('multiLine', 'false')
    .csv(IN)
)

rating_clean = F.when(
    F.regexp_replace(F.col('rating'), '[^0-9.]', '').rlike(r'^[1-5](\.0+)?$'),
    F.regexp_replace(F.col('rating'), '[^0-9.]', '').cast('double'),
).otherwise(F.lit(None))

sel = df.where((F.col('rating').isNotNull()) & (rating_clean.isNull())).limit(1)
rows = sel.collect()

with OUT.open('w', encoding='utf-8') as fh:
    if not rows:
        fh.write('No problematic row found\n')
        print('No problematic row found')
    else:
        r = rows[0]
        d = {k: r[k] for k in r.__fields__}
        fh.write(str(d) + '\n')
        print(d)

spark.stop()
