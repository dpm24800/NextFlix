import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity

from src.logger import logger
from src.exception import CustomException
from src.utils import save_object, intra_list_similarity


@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join("artifacts", "recommender_model.pkl")


class RecommenderModel:
    """
    Content-based movie recommender.
    Loaded from artifacts by the prediction pipeline — no re-training needed.
    """
    def __init__(self, df: pd.DataFrame, vectorizer, tfidf_matrix):
        self.df           = df.reset_index(drop=True)
        self.vectorizer   = vectorizer
        self.tfidf_matrix = tfidf_matrix
        self.title_index  = pd.Series(
            self.df.index, index=self.df["title"].str.lower()
        )

    # ── Public API ──────────────────────────────────────────────────────────

    def recommend_by_title(self, title: str, genre_filter: str = None, n: int = 5):
        """Top-N most similar movies to a given title."""
        key = title.lower().strip()
        if key not in self.title_index:
            logger.warning(f"Title not found: {title}")
            return pd.DataFrame()
        idx    = self.title_index[key]
        scores = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        scores[idx] = -1                                      # exclude self
        scores       = self._genre_mask(scores, genre_filter)
        return self._top_n(scores, n)

    def recommend_by_description(self, description: str, genre_filter: str = None, n: int = 5):
        """Top-N movies matching a free-text description."""
        from src.components.data_transformation import DataTransformation
        cleaned = DataTransformation.clean_text(description)
        if not cleaned:
            logger.warning("Empty cleaned description")
            return pd.DataFrame()
        vec    = self.vectorizer.transform([cleaned])
        scores = cosine_similarity(vec, self.tfidf_matrix).flatten()
        scores = self._genre_mask(scores, genre_filter)
        return self._top_n(scores, n)

    def recommend_by_genre(self, genre: str, n: int = 5):
        """Top-N highest-rated movies in a genre."""
        col = self._genre_col()
        if col is None:
            return pd.DataFrame()
        mask   = self.df[col].str.lower().str.contains(genre.lower(), na=False)
        subset = self.df[mask].copy()
        if subset.empty:
            return pd.DataFrame()
        if "ratings" in subset.columns:
            subset = subset.sort_values("ratings", ascending=False)
        subset["similarity_score"] = None
        return subset[self._output_cols()].head(n).reset_index(drop=True)

    def all_genres(self):
        col = self._genre_col()
        if col is None:
            return []
        return sorted(
            self.df[col].dropna()
            .str.split(r"[,|/]").explode()
            .str.strip().value_counts().index.tolist()
        )

    def title_suggestions(self, query: str, limit: int = 6):
        q = query.lower()
        return [t for t in self.df["title"].tolist() if q in t.lower()][:limit]

    def stats(self) -> dict:
        return {
            "total_movies"  : len(self.df),
            "tfidf_features": self.tfidf_matrix.shape[1],
            "total_genres"  : len(self.all_genres()),
            "sparsity_pct"  : round(
                100 * (1 - self.tfidf_matrix.nnz /
                       (self.tfidf_matrix.shape[0] * self.tfidf_matrix.shape[1])), 1
            ),
        }

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _genre_col(self):
        for c in ["expanded-genres", "genre", "genres"]:
            if c in self.df.columns:
                return c
        return None

    def _genre_mask(self, scores, genre_filter):
        if not genre_filter or genre_filter == "All":
            return scores
        col = self._genre_col()
        if col is None:
            return scores
        mask = self.df[col].str.lower().str.contains(genre_filter.lower(), na=False).values
        scores[~mask] = -1
        return scores

    def _top_n(self, scores, n):
        top    = scores.argsort()[::-1][:n]
        result = self.df.iloc[top][self._output_cols()].copy()
        result["similarity_score"] = scores[top].round(4)
        return result.reset_index(drop=True)

    def _output_cols(self):
        base  = ["title", "description"]
        extra = [c for c in ["ratings", "genre", "expanded-genres"] if c in self.df.columns]
        return base + extra


# ── ModelTrainer ─────────────────────────────────────────────────────────────

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_training(self, df, vectorizer, tfidf_matrix):
        """
        1. Instantiate RecommenderModel with pre-built TF-IDF artifacts
        2. Run evaluation (Intra-List Similarity)
        3. Persist the model object to disk
        Returns: RecommenderModel
        """
        logger.info("ModelTrainer: starting")
        try:
            model = RecommenderModel(df, vectorizer, tfidf_matrix)

            # ── Evaluation ──────────────────────────────────────────
            logger.info("Evaluating model (Intra-List Similarity)…")
            sample_titles = df["title"].sample(min(50, len(df)), random_state=42).tolist()
            ils_scores = []

            for title in sample_titles:
                recs = model.recommend_by_title(title, n=5)
                if recs.empty:
                    continue
                indices = [
                    model.title_index.get(t.lower())
                    for t in recs["title"]
                    if t.lower() in model.title_index
                ]
                score = intra_list_similarity(tfidf_matrix, [i for i in indices if i is not None])
                if score > 0:
                    ils_scores.append(score)

            mean_ils = float(np.mean(ils_scores)) if ils_scores else 0.0
            logger.info(f"Mean ILS (lower = more diverse): {mean_ils:.4f}")

            # ── Coverage metric ─────────────────────────────────────
            recommended_set = set()
            for title in sample_titles:
                recs = model.recommend_by_title(title, n=5)
                recommended_set.update(recs["title"].tolist())
            coverage = len(recommended_set) / len(df) * 100
            logger.info(f"Catalogue coverage : {coverage:.1f}%")

            # ── Save model ──────────────────────────────────────────
            save_object(self.config.model_path, model)
            logger.info(f"Model saved → {self.config.model_path}")
            logger.info("ModelTrainer: complete")

            print(f"\n{'='*55}")
            print("  MODEL TRAINING COMPLETE")
            print(f"{'='*55}")
            print(f"  Total movies indexed : {model.stats()['total_movies']:,}")
            print(f"  TF-IDF features      : {model.stats()['tfidf_features']:,}")
            print(f"  Mean ILS (diversity) : {mean_ils:.4f}")
            print(f"  Catalogue coverage   : {coverage:.1f}%")
            print(f"  Model saved to       : {self.config.model_path}")
            print(f"{'='*55}\n")

            return model

        except Exception as e:
            raise CustomException(e, sys)
