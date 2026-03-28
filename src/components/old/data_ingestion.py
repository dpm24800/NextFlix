import os
import sys
import pandas as pd
from dataclasses import dataclass
from datasets import load_dataset

from src.logger import logger
from src.exception import CustomException


@dataclass
class DataIngestionConfig:
    raw_data_path: str      = os.path.join("artifacts", "raw_train.csv")
    val_data_path: str      = os.path.join("artifacts", "raw_val.csv")
    dataset_name: str       = "jquigl/imdb-genres"


class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Pull the IMDB Genres dataset from Hugging Face and persist
        both train and validation splits as CSV files in artifacts/.
        Returns: (train_path, val_path)
        """
        logger.info("DataIngestion: starting")
        try:
            # ── Load from Hugging Face ─────────────────────────────
            logger.info(f"Fetching dataset: {self.config.dataset_name}")
            dataset = load_dataset(self.config.dataset_name)

            train_df = pd.DataFrame(dataset["train"])
            val_df   = pd.DataFrame(dataset["validation"])

            logger.info(f"Train shape   : {train_df.shape}")
            logger.info(f"Val shape     : {val_df.shape}")
            logger.info(f"Columns       : {list(train_df.columns)}")

            # ── Persist raw CSVs ───────────────────────────────────
            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)
            train_df.to_csv(self.config.raw_data_path, index=False)
            val_df.to_csv(self.config.val_data_path,   index=False)

            logger.info(f"Raw train saved → {self.config.raw_data_path}")
            logger.info(f"Raw val   saved → {self.config.val_data_path}")
            logger.info("DataIngestion: complete")

            return self.config.raw_data_path, self.config.val_data_path

        except Exception as e:
            raise CustomException(e, sys)
