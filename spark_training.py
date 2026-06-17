from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
	Tokenizer,
	StopWordsRemover,
	HashingTF,
	IDF,
	StringIndexer,
	VectorAssembler,
)
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import PipelineModel
import argparse
import os


def build_and_train_mlp(spark: SparkSession, input_csv: str, model_out: str, num_features: int = 2000):
	df = (
		spark.read.option("header", True).option("inferSchema", True).csv(input_csv)
	)

	# Basic cleaning: drop rows where rating is null
	df = df.filter(col("rating").isNotNull())

	# Ensure columns used exist
	for c in ["title", "comment", "votes", "purchased", "rating"]:
		if c not in df.columns:
			raise ValueError(f"Expected column '{c}' in input data but it is missing")

	# Convert boolean-like `purchased` to indexed numeric
	purchased_indexer = StringIndexer(inputCol="purchased", outputCol="purchased_idx").setHandleInvalid("keep")

	# Label indexer: map ratings (1..5) to 0..4
	label_indexer = StringIndexer(inputCol="rating", outputCol="label").setHandleInvalid("keep")

	# Text processing for title
	title_tokenizer = Tokenizer(inputCol="title", outputCol="title_tokens")
	title_sw = StopWordsRemover(inputCol="title_tokens", outputCol="title_filtered")
	title_hash = HashingTF(inputCol="title_filtered", outputCol="title_tf", numFeatures=num_features)
	title_idf = IDF(inputCol="title_tf", outputCol="title_tfidf")

	# Text processing for comment
	comment_tokenizer = Tokenizer(inputCol="comment", outputCol="comment_tokens")
	comment_sw = StopWordsRemover(inputCol="comment_tokens", outputCol="comment_filtered")
	comment_hash = HashingTF(inputCol="comment_filtered", outputCol="comment_tf", numFeatures=num_features)
	comment_idf = IDF(inputCol="comment_tf", outputCol="comment_tfidf")

	# Assemble features: text tfidf vectors + numeric columns
	assembler = VectorAssembler(
		inputCols=["title_tfidf", "comment_tfidf", "votes", "purchased_idx"], outputCol="features"
	)

	# Define MLP layers. Input size = num_features (title) + num_features (comment) + 2 numeric
	input_size = num_features + num_features + 2
	layers = [input_size, 256, 64, 5]

	mlp = MultilayerPerceptronClassifier(layers=layers, labelCol="label", featuresCol="features", maxIter=100, seed=42)

	pipeline = Pipeline(
		stages=[
			purchased_indexer,
			label_indexer,
			title_tokenizer,
			title_sw,
			title_hash,
			title_idf,
			comment_tokenizer,
			comment_sw,
			comment_hash,
			comment_idf,
			assembler,
			mlp,
		]
	)

	# Split data
	train, test = df.randomSplit([0.8, 0.2], seed=42)

	model = pipeline.fit(train)

	# Evaluate
	preds = model.transform(test)
	evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
	acc = evaluator.evaluate(preds)

	print(f"Test accuracy: {acc:.4f}")

	# Save model
	os.makedirs(os.path.dirname(model_out), exist_ok=True)
	model.write().overwrite().save(model_out)
	print(f"Saved pipeline model to: {model_out}")

	return model, acc


def main():
	parser = argparse.ArgumentParser(description="Train a Spark MLP on review ratings")
	parser.add_argument("--input", "-i", required=False, help="Input CSV file path", default="training_data/train_clean.csv")
	parser.add_argument("--model-out", "-o", required=False, help="Output path for saved model", default="Model/mlp_pipeline")
	parser.add_argument("--num-features", "-f", required=False, type=int, default=2000, help="Number of hashing features for text")

	args = parser.parse_args()

	spark = SparkSession.builder.appName("SparkMLPTrainer").getOrCreate()
	try:
		build_and_train_mlp(spark, args.input, args.model_out, args.num_features)
	finally:
		spark.stop()


if __name__ == "__main__":
	main()

