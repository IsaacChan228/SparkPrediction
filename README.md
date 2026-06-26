# SparkPrediction

SparkPrediction is a Spark + PyTorch rating prediction pipeline. It cleans the raw CSVs, merges training and prediction data with product information, generates sentence-transformer embeddings, trains a tabular MLP regressor, and writes prediction outputs.

## Core Components

| Component | Role in the code | Model structure / output |
| --- | --- | --- |
| Spark | Cleans CSVs, merges training and prediction data, sanitizes numeric fields, and streams data into pandas batches. | Produces numeric tabular rows for training and inference. |
| BERT | `sentence-transformers` generates embeddings for `comment`, `title`, `prod_title`, and `prod_features` during the merge step. | Each text column becomes compressed embedding features named like `<column>_emb_<index>`. |
| PyTorch | Trains and serves the regressor used for rating prediction. | `MLP(input_dim, hidden=(4096, 2048, 2048, 1024, 512, 256, 128, 64))` with LayerNorm, SiLU, Dropout, and a final linear output layer. |

## Requirements

- Python 3.11
- Java JDK 17
- PySpark
- pandas, numpy
- torch, torchvision, torchaudio
- sentence-transformers

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Data Flow

The current workflow is:

1. Clean training data.
2. Clean product info.
3. Clean prediction input data.
4. Merge cleaned rows with product info.
5. Generate embeddings during the merge step.
6. Train the PyTorch model.
7. Run prediction.

## Run Order

```bash
py traindatacleaning.py
py prodinfocleaning.py
py predictiondatacleaning.py
py data_merging.py --train
py data_merging.py
py spark_training.py
py spark_prediction.py
```

## What Each Step Produces

- `traindatacleaning.py` writes `training_data/train_clean.csv` and, when enabled, `training_data/train_clean_corrupted.csv`.
- `prodinfocleaning.py` writes `product_info/prodInfo_clean.csv` and, when enabled, `product_info/prodInfo_clean_corrupted.csv`.
- `predictiondatacleaning.py` writes `prediction_input/test_clean.csv` and, when enabled, `prediction_input/test_clean_corrupted.csv`.
- `data_merging.py --train` writes `training_data/train_merged.csv` and saves embedding compressors in `artifacts/emb_compressors/`.
- `data_merging.py` writes `prediction_input/test_merged.csv` and reuses the saved compressors.
- `spark_training.py` writes `Model/pytorch_mlp.pt` and updates `prediction_output/training_report.txt`.
- `spark_prediction.py` writes `prediction_output/predictions_float.csv` and `prediction_output/predictions_int.csv`, then updates `prediction_output/training_report.txt` with inference and prediction analysis.

## Features Used for Training

The model trains on these numeric inputs, in this order:

- `votes` as a float, with non-numeric characters stripped and missing values mapped to `0.0`
- `purchased` as `1.0` when the source value is `TRUE`, otherwise `0.0`
- `time` as a float, with non-numeric characters stripped and missing values mapped to `0.0`
- `prod_price` as a float, with non-numeric characters stripped and missing values mapped to `0.0`
- `prod_rating_number` as a float, with non-numeric characters stripped and missing values mapped to `0.0`
- `prod_main_category` as categorical codes cast to float
- `prod_store` as categorical codes cast to float
- embedding columns named like `<column>_emb_<index>` for `comment`, `title`, `prod_title`, and `prod_features`, sorted by suffix index and appended in that order

The target label is `rating`, stored as `label` during training.

## Configuration

Runtime paths and training hyperparameters are read from `config.cfg`.

- `training.epochs`
- `training.batch_size`
- `training.lr`
- `training.weight_decay`
- `training.early_stopping_patience`
- `paths.train_csv`
- `paths.input_csv`
- `paths.model_path`
- `paths.report_path`

## Notes

- The merge step computes embeddings with `sentence-transformers` and reduces them through saved compressors in `artifacts/emb_compressors/`.
- `spark_prediction.py` caps prediction values to the `1` to `5` range before writing outputs.
- On Windows, Spark can use `C:/hadoop/bin/winutils.exe` if it is present.
