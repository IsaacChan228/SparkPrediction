"""Project entry point.

Running this module cleans ``training_data/train.csv`` by removing rows that do
not exactly match the schema described in
``training_data/training_data_format``.
"""

from traindatacleaning import clean_training_csv


def main() -> None:
    result = clean_training_csv()
    print(
        "Training data cleaned: "
        f"{result.clean_rows} rows kept, "
        f"{result.corrupted_rows} rows removed"
    )


if __name__ == "__main__":
    main()