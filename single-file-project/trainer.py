import os
import sys
import pickle
import logging
import re
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset
import nltk

# ── NLTK Setup ──────────────────────────────────────────────────────────────
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

STOP_WORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ── Logging Configuration ───────────────────────────────────────────────────
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log")
logging.basicConfig(
    filename=LOG_FILE,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("nextflix_trainer")

# ── Exception Handling ──────────────────────────────────────────────────────
def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    return f"Error in script [{file_name}] at line [{line_number}]: {str(error)}"

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)
        logger.error(self.error_message)
    def __str__(self):
        return self.error_message

# ── Utilities ───────────────────────────────────────────────────────────────
def save_object(file_path: str, obj) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
        logger.info(f"Object saved → {file_path}")
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path: str):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Artifact not found: {file_path}")
        with open(file_path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"Object loaded ← {file_path}")
        return obj
    except Exception as e:
        raise CustomException(e, sys)

def intra_list_similarity(tfidf_matrix, indices: list) -> float:
    indices = [int(i) for i in indices]
    if len(indices) < 2:
        return 0.0
    sub = tfidf_matrix[indices]
    sim = cosine_similarity(sub)
    np.fill_diagonal(sim, 0)
    return float(sim.sum() / (len(indices) * (len(indices) - 1)))

# ── Text Cleaning ───────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return " "
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [
        lemmatizer.lemmatize(w)
        for w in text.split()
        if w not in STOP_WORDS and len(w) > 2
    ]
    return " ".join(tokens)

# ── Data Ingestion ──────────────────────────────────────────────────────────
@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw_train.csv")
    val_data_path: str = os.path.join("artifacts", "raw_val.csv")
    dataset_name: str = "jquigl/imdb-genres"
    COLUMN_RENAME_MAP = {
        "movie title - year": "title",
        "rating": "ratings",
    }

class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logger.info("DataIngestion: starting ")
        try:
            logger.info(f"Fetching dataset: {self.config.dataset_name} ")
            dataset = load_dataset(self.config.dataset_name)
            train_df = pd.DataFrame(dataset["train"])
            val_df = pd.DataFrame(dataset["validation"])
            
            for df in (train_df, val_df):
                df.rename(columns=self.config.COLUMN_RENAME_MAP, inplace=True)
                df.columns = df.columns.str.strip()

            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)
            train_df.to_csv(self.config.raw_data_path, index=False)
            val_df.to_csv(self.config.val_data_path, index=False)
            
            logger.info("DataIngestion: complete ")
            return self.config.raw_data_path, self.config.val_data_path
        except Exception as e:
            raise CustomException(e, sys)

# ── Data Transformation ─────────────────────────────────────────────────────
@dataclass
class DataTransformationConfig:
    processed_data_path: str = os.path.join("artifacts", "processed_movies.pkl")
    vectorizer_path: str = os.path.join("artifacts", "tfidf_vectorizer.pkl")
    tfidf_matrix_path: str = os.path.join("artifacts", "tfidf_matrix.pkl")

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def _ensure_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        rename = {}
        if "movie title - year " in df.columns and "title " not in df.columns:
            rename["movie title - year "] = "title "
        if "rating " in df.columns and "ratings " not in df.columns:
            rename["rating "] = "ratings "
        if rename:
            df = df.rename(columns=rename)
        return df

    def initiate_data_transformation(self, train_path: str):
        logger.info("DataTransformation: starting ")
        try:
            df = pd.read_csv(train_path)
            df = self._ensure_columns(df)
            
            required = ["title", "description"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise ValueError(f"Required columns missing: {missing}")

            df = df.dropna(subset=["description"]).reset_index(drop=True)
            df = df.drop_duplicates(subset=["title", "description"], keep="first")
            df = df.drop_duplicates(subset=["title"], keep="first")
            df = df.reset_index(drop=True)

            logger.info("Cleaning description text… ")
            df["clean_description"] = df["description"].apply(clean_text)
            df = df[df["clean_description"].str.strip() != ""].reset_index(drop=True)

            logger.info("Fitting TF-IDF vectorizer… ")
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=20_000,
                sublinear_tf=True,
                min_df=2,
            )
            tfidf_matrix = vectorizer.fit_transform(df["clean_description"])

            save_object(self.config.processed_data_path, df)
            save_object(self.config.vectorizer_path, vectorizer)
            save_object(self.config.tfidf_matrix_path, tfidf_matrix)

            logger.info("DataTransformation: complete ")
            return df, vectorizer, tfidf_matrix
        except Exception as e:
            raise CustomException(e, sys)

# ── Model Definition (Must match app.py for pickle compatibility) ───────────
class RecommenderModel:
    def __init__(self, df: pd.DataFrame, vectorizer, tfidf_matrix):
        self.df = df.reset_index(drop=True)
        self.vectorizer = vectorizer
        self.tfidf_matrix = tfidf_matrix
        _titles = self.df["title"].astype(str).str.strip().str.lower()
        _mask = ~_titles.duplicated(keep="first")
        self.title_index = pd.Series(
            self.df.index[_mask].tolist(),
            index=_titles[_mask].tolist()
        )

    @staticmethod
    def _normalise(text: str) -> str:
        return str(text).strip().lower()

    @staticmethod
    def _name_only(full_title: str) -> str:
        t = str(full_title).strip().lower()
        return re.sub(r"\s*-\s*\d{4}\s*$", "", t).strip()

    def _safe_idx(self, k: str) -> int:
        val = self.title_index[k]
        if hasattr(val, "iloc"):
            val = val.iloc[0]
        return int(val)

    def find_title_index(self, query: str):
        q = self._normalise(query)
        if q in self.title_index:
            return self._safe_idx(q), q
        for full_key in self.title_index.index:
            if self._name_only(full_key) == q:
                return self._safe_idx(full_key), full_key
        query_words = q.split()
        for full_key in self.title_index.index:
            name = self._name_only(full_key)
            if all(w in name for w in query_words):
                return self._safe_idx(full_key), full_key
        for full_key in self.title_index.index:
            if q in full_key:
                return self._safe_idx(full_key), full_key
        return None, None

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
        base = ["title", "description"]
        extra = [c for c in ["ratings", "genre", "expanded-genres"] if c in self.df.columns]
        return base + extra

    def recommend_by_title(self, title: str, genre_filter: str = None, n: int = 5):
        idx, matched = self.find_title_index(title)
        if idx is None:
            return pd.DataFrame()
        scores = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        scores[idx] = -1
        scores = self._genre_mask(scores, genre_filter)
        return self._top_n(scores, n)

    def recommend_by_description(self, description: str, genre_filter: str = None, n: int = 5):
        # Use local clean_text function instead of importing DataTransformation
        cleaned = clean_text(description)
        if not cleaned:
            return pd.DataFrame()
        vec = self.vectorizer.transform([cleaned])
        scores = cosine_similarity(vec, self.tfidf_matrix).flatten()
        scores = self._genre_mask(scores, genre_filter)
        return self._top_n(scores, n)

    def recommend_by_genre(self, genre: str, n: int = 5):
        col = self._genre_col()
        if col is None:
            return pd.DataFrame()
        mask = self.df[col].astype(str).str.lower().str.contains(genre.lower(), na=False)
        subset = self.df[mask].copy()
        if subset.empty:
            return pd.DataFrame()
        subset = subset.drop_duplicates(subset=["title"], keep="first")
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

    def title_suggestions(self, query: str, limit: int = 8):
        q = self._normalise(query)
        if not q:
            return []
        starts, contains = [], []
        seen = set()
        for raw in self.df["title"].astype(str).tolist():
            if raw in seen:
                continue
            name = self._name_only(raw)
            if name.startswith(q):
                starts.append(raw)
                seen.add(raw)
            elif q in name:
                contains.append(raw)
                seen.add(raw)
        return (starts + contains)[:limit]

    def stats(self) -> dict:
        return {
            "total_movies": len(self.df),
            "tfidf_features": self.tfidf_matrix.shape[1],
            "total_genres": len(self.all_genres()),
            "sparsity_pct": round(
                100 * (1 - self.tfidf_matrix.nnz / (self.tfidf_matrix.shape[0] * self.tfidf_matrix.shape[1])), 1
            ),
        }

# ── Model Trainer ───────────────────────────────────────────────────────────
@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join("artifacts", "recommender_model.pkl")

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_training(self, df, vectorizer, tfidf_matrix):
        logger.info("ModelTrainer: starting ")
        try:
            model = RecommenderModel(df, vectorizer, tfidf_matrix)
            logger.info("Evaluating model… ")
            sample_titles = df["title"].sample(min(50, len(df)), random_state=42).tolist()
            ils_scores = []
            recommended = set()

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
                    if hasattr(val, "__iter__"):
                        val = list(val)[0]
                    indices.append(int(val))
                score = intra_list_similarity(tfidf_matrix, indices)
                if score > 0:
                    ils_scores.append(score)

            mean_ils = float(np.mean(ils_scores)) if ils_scores else 0.0
            coverage = len(recommended) / len(df) * 100
            logger.info(f"Mean ILS (lower=more diverse) : {mean_ils:.4f} ")
            logger.info(f"Catalogue coverage            : {coverage:.1f}% ")

            save_object(self.config.model_path, model)
            logger.info(f"Model saved → {self.config.model_path} ")
            print(f"\n{'='*55} ")
            print("  MODEL TRAINING COMPLETE ")
            print(f"{'='*55} ")
            print(f"  Total movies indexed : {model.stats()['total_movies']:,} ")
            print(f"  TF-IDF features      : {model.stats()['tfidf_features']:,} ")
            print(f"  Mean ILS (diversity) : {mean_ils:.4f}  (lower = more diverse) ")
            print(f"  Catalogue coverage   : {coverage:.1f}% ")
            print(f"  Model saved to       : {self.config.model_path} ")
            print(f"{'='*55}\n ")
            logger.info("ModelTrainer: complete ")
            return model
        except Exception as e:
            raise CustomException(e, sys)

# ── Train Pipeline ──────────────────────────────────────────────────────────
class TrainPipeline:
    def initiate_training(self):
        try:
            logger.info("=" * 55)
            logger.info("  TRAIN PIPELINE: START ")
            logger.info("=" * 55)

            logger.info("Step 1: Data Ingestion ")
            ingestion = DataIngestion()
            train_path, val_path = ingestion.initiate_data_ingestion()

            logger.info("Step 2: Data Transformation ")
            transformation = DataTransformation()
            df, vectorizer, tfidf_matrix = transformation.initiate_data_transformation(train_path)

            logger.info("Step 3: Model Training & Evaluation ")
            trainer = ModelTrainer()
            model = trainer.initiate_model_training(df, vectorizer, tfidf_matrix)

            logger.info("TRAIN PIPELINE: COMPLETE ")
            return model
        except Exception as e:
            raise CustomException(e, sys)

# ── Main Entry Point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  NextFlix — Movie Recommendation System")
    print("  Starting training pipeline…")
    print("=" * 55)
    pipeline = TrainPipeline()
    pipeline.initiate_training()
    print("\n Training complete!")
    print(" Run the app with:  streamlit run app.py\n")