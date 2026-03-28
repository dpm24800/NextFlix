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
    Built once during training, saved to disk, loaded by PredictPipeline.
    """
    def __init__(self, df: pd.DataFrame, vectorizer, tfidf_matrix):
        self.df           = df.reset_index(drop=True)
        self.vectorizer   = vectorizer
        self.tfidf_matrix = tfidf_matrix

        # title column is guaranteed to exist (normalised by DataIngestion)
        # Keep only the first occurrence of each title to ensure unique index
        _titles = self.df["title"].astype(str).str.strip().str.lower()
        _mask   = ~_titles.duplicated(keep="first")
        self.title_index = pd.Series(
            self.df.index[_mask].tolist(),
            index=_titles[_mask].tolist()
        )

    # ── Public API ──────────────────────────────────────────────────────────

    def recommend_by_title(self, title: str, genre_filter: str = None, n: int = 5):
        """Top-N most similar movies to a given title."""
        key = str(title).strip().lower()

        # Try exact match first, then partial match (handles year suffix)
        def _safe_get(k):
            val = self.title_index[k]
            # guard against duplicate keys returning a Series
            if hasattr(val, "iloc"):
                val = val.iloc[0]
            return int(val)

        if key in self.title_index:
            idx = _safe_get(key)
        else:
            # Partial match: find titles that contain the query string
            matches = [t for t in self.title_index.index if key in t]
            if not matches:
                logger.warning(f"Title not found: {title}")
                return pd.DataFrame()
            idx = _safe_get(matches[0])

        scores       = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        scores[idx]  = -1
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
        mask   = self.df[col].astype(str).str.lower().str.contains(genre.lower(), na=False)
        subset = self.df[mask].copy()
        if subset.empty:
            return pd.DataFrame()
        rating_col = "ratings" if "ratings" in subset.columns else None
        if rating_col:
            subset = subset.sort_values(rating_col, ascending=False)
        subset["similarity_score"] = None
        return subset[self._output_cols()].head(n).reset_index(drop=True)

    def all_genres(self):
        col = self._genre_col()
        if col is None:
            return []
        return sorted(
            self.df[col].dropna()
            .astype(str).str.split(r"[,|/]").explode()
            .str.strip().value_counts().index.tolist()
        )

    def title_suggestions(self, query: str, limit: int = 6):
        q = query.lower()
        return [t for t in self.df["title"].astype(str).tolist() if q in t.lower()][:limit]

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
        mask = self.df[col].astype(str).str.lower().str.contains(genre_filter.lower(), na=False).values
        scores[~mask] = -1
        return scores

    def _top_n(self, scores, n):
        # Convert to plain Python ints to avoid numpy int64/iloc issues
        # on older pandas. Also skip masked scores (<= -1) and OOB indices.
        n_movies = len(self.df)
        top = [
            int(i) for i in scores.argsort()[::-1]
            if float(scores[i]) > -1 and int(i) < n_movies
        ][:n]

        if not top:
            return pd.DataFrame()

        result = self.df.iloc[top][self._output_cols()].copy()
        result["similarity_score"] = [round(float(scores[i]), 4) for i in top]
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
        1. Instantiate RecommenderModel
        2. Evaluate: Intra-List Similarity + Catalogue Coverage
        3. Persist model to disk
        Returns: RecommenderModel
        """
        logger.info("ModelTrainer: starting")
        try:
            model = RecommenderModel(df, vectorizer, tfidf_matrix)

            # ── Evaluation ──────────────────────────────────────────
            logger.info("Evaluating model…")
            sample_titles = df["title"].sample(min(50, len(df)), random_state=42).tolist()
            ils_scores    = []
            recommended   = set()

            for title in sample_titles:
                try:
                    recs = model.recommend_by_title(title, n=5)
                except Exception:
                    continue
                if recs.empty:
                    continue
                recommended.update(recs["title"].tolist())
                indices = []
                for t in recs["title"].astype(str):
                    key = t.strip().lower()
                    if key not in model.title_index:
                        continue
                    val = model.title_index[key]
                    # guard against Series (duplicate keys) — take first
                    if hasattr(val, "__iter__"):
                        val = list(val)[0]
                    indices.append(int(val))
                score   = intra_list_similarity(tfidf_matrix, indices)
                if score > 0:
                    ils_scores.append(score)

            mean_ils = float(np.mean(ils_scores)) if ils_scores else 0.0
            coverage = len(recommended) / len(df) * 100
            logger.info(f"Mean ILS (lower=more diverse) : {mean_ils:.4f}")
            logger.info(f"Catalogue coverage            : {coverage:.1f}%")

            # ── Save model ──────────────────────────────────────────
            save_object(self.config.model_path, model)
            logger.info(f"Model saved → {self.config.model_path}")

            print(f"\n{'='*55}")
            print("  MODEL TRAINING COMPLETE")
            print(f"{'='*55}")
            print(f"  Total movies indexed : {model.stats()['total_movies']:,}")
            print(f"  TF-IDF features      : {model.stats()['tfidf_features']:,}")
            print(f"  Mean ILS (diversity) : {mean_ils:.4f}  (lower = more diverse)")
            print(f"  Catalogue coverage   : {coverage:.1f}%")
            print(f"  Model saved to       : {self.config.model_path}")
            print(f"{'='*55}\n")

            logger.info("ModelTrainer: complete")
            return model

        except Exception as e:
            raise CustomException(e, sys)