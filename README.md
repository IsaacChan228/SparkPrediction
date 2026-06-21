	# SparkPrediction


py traindatacleaning.py

py prodinfocleaning.py

py predictiondatacleaning.py

py data_merging.py --train

py data_merging.py

py spark_training.py

py spark_prediction.py

these CSV columns are now pushed into training (in order):

votes (float, non-numeric stripped, missing → 0.0)
purchased (1.0 if "TRUE", else 0.0)
time (float, non-numeric stripped, missing → 0.0)
prod_price (float, non-numeric stripped, missing → 0.0)
prod_rating_number (float, non-numeric stripped, missing → 0.0)
prod_main_category (categorical → numeric codes, NaN → -1 code cast to float)
prod_store (categorical → numeric codes, NaN → -1 code cast to float)
embeddings: all numeric columns named <col>emb<i> in this bert order — comment, title, prod_title, prod_features (each group sorted by trailing index and appended)

Label (not a feature): rating → label (float).

To increase driver JVM memory or reduce shuffle load, set these beforehand in PowerShell:
$env:SPARK_DRIVER_MEMORY='6g'
$env:SPARK_DRIVER_MAX_RESULT_SIZE='3g'
$env:SPARK_SQL_SHUFFLE_PARTITIONS='8'
$env:SPARK_MASTER='local[*]'
$env:SPARK_TO_PANDAS_BATCH_SIZE='10000'