import sys
from src.logger import logger
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainPipeline:
    def initiate_training(self):
        """
        Full training pipeline:
          DataIngestion → DataTransformation → ModelTrainer

        Artifacts saved:
          artifacts/raw_train.csv
          artifacts/raw_val.csv
          artifacts/processed_movies.pkl
          artifacts/tfidf_vectorizer.pkl
          artifacts/tfidf_matrix.pkl
          artifacts/recommender_model.pkl
        """
        try:
            logger.info("=" * 55)
            logger.info("  TRAIN PIPELINE: START")
            logger.info("=" * 55)

            # Step 1 — Data Ingestion
            logger.info("Step 1: Data Ingestion")
            ingestion        = DataIngestion()
            train_path, val_path = ingestion.initiate_data_ingestion()

            # Step 2 — Data Transformation
            logger.info("Step 2: Data Transformation")
            transformation   = DataTransformation()
            df, vectorizer, tfidf_matrix = transformation.initiate_data_transformation(train_path)

            # Step 3 — Model Training
            logger.info("Step 3: Model Training & Evaluation")
            trainer          = ModelTrainer()
            model            = trainer.initiate_model_training(df, vectorizer, tfidf_matrix)

            logger.info("TRAIN PIPELINE: COMPLETE")
            return model

        except Exception as e:
            raise CustomException(e, sys)
