import os
import sys
import pandas as pd
from dataclasses import dataclass
from datasets import load_dataset

from src.logger import logger
from src.exception import CustomException


@dataclass
class DataIngestionConfig:
    raw_data_path: str  = os.path.join("artifacts", "raw_train.csv")
    val_data_path: str  = os.path.join("artifacts", "raw_val.csv")
    dataset_name: str   = "jquigl/imdb-genres"


# Actual column names in jquigl/imdb-genres → our internal names
COLUMN_RENAME_MAP = {
    "movie title - year": "title",   # e.g. "Die Hard - 1988"
    "rating":             "ratings", # float, e.g. 8.2
}


class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        1. Pull jquigl/imdb-genres from Hugging Face
        2. Rename columns to internal standard names:
               'movie title - year' → 'title'
               'rating'             → 'ratings'
        3. Persist both splits as CSV to artifacts/
        Returns: (train_path, val_path)
        """
        logger.info("DataIngestion: starting")
        try:
            logger.info(f"Fetching dataset: {self.config.dataset_name}")
            dataset = load_dataset(self.config.dataset_name)

            train_df = pd.DataFrame(dataset["train"])
            val_df   = pd.DataFrame(dataset["validation"])

            logger.info(f"Raw columns : {list(train_df.columns)}")
            logger.info(f"Train shape : {train_df.shape}")
            logger.info(f"Val   shape : {val_df.shape}")

            # ── Normalise column names ─────────────────────────────
            for df in (train_df, val_df):
                df.rename(columns=COLUMN_RENAME_MAP, inplace=True)
                # Strip any leading/trailing whitespace from column names
                df.columns = df.columns.str.strip()

            logger.info(f"Normalised columns: {list(train_df.columns)}")

            # ── Persist CSVs ───────────────────────────────────────
            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)
            
            train_df.to_csv(self.config.raw_data_path, index=False)
            val_df.to_csv(self.config.val_data_path,   index=False)

            logger.info(f"Saved train → {self.config.raw_data_path}")
            logger.info(f"Saved val   → {self.config.val_data_path}")
            logger.info("DataIngestion: complete")

            return self.config.raw_data_path, self.config.val_data_path

        except Exception as e:
            raise CustomException(e, sys)