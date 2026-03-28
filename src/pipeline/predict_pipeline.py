import os
import sys
from src.logger import logger
from src.exception import CustomException
from src.utils import load_object, artifacts_exist

# Paths to all required saved artifacts
MODEL_PATH = os.path.join("artifacts", "recommender_model.pkl")

REQUIRED_ARTIFACTS = [MODEL_PATH]


class PredictPipeline:
    """
    Loads the saved RecommenderModel from disk and provides
    recommendation methods for use by the Streamlit app.
    """

    def __init__(self):
        self._model = None

    def load_model(self):
        """Load model from disk (called once, then cached by Streamlit)."""
        try:
            if not artifacts_exist(REQUIRED_ARTIFACTS):
                raise FileNotFoundError(
                    "Trained artifacts not found. Please run: python train.py"
                )
            logger.info("PredictPipeline: loading model from artifacts/")
            self._model = load_object(MODEL_PATH)
            logger.info("PredictPipeline: model loaded successfully")
            return self._model
        except Exception as e:
            raise CustomException(e, sys)

    @property
    def model(self):
        if self._model is None:
            self.load_model()
        return self._model

    # ── Convenience wrappers (thin pass-throughs) ───────────────────────────

    def recommend_by_title(self, title: str, genre_filter: str = None, n: int = 5):
        logger.info(f"Predict: by_title | '{title}' | genre={genre_filter} | n={n}")
        return self.model.recommend_by_title(title, genre_filter, n)

    def recommend_by_description(self, description: str, genre_filter: str = None, n: int = 5):
        logger.info(f"Predict: by_description | genre={genre_filter} | n={n}")
        return self.model.recommend_by_description(description, genre_filter, n)

    def recommend_by_genre(self, genre: str, n: int = 5):
        logger.info(f"Predict: by_genre | '{genre}' | n={n}")
        return self.model.recommend_by_genre(genre, n)

    def all_genres(self):
        return self.model.all_genres()

    def title_suggestions(self, query: str, limit: int = 6):
        return self.model.title_suggestions(query, limit)

    def stats(self) -> dict:
        return self.model.stats()
