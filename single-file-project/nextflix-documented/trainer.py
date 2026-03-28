# =============================================================================
# NextFlix — Movie Recommendation System (Training Pipeline)
# =============================================================================
# This module handles the complete training workflow for a content-based movie
# recommendation system using TF-IDF vectorization and cosine similarity.
# 
# Key Components:
# 1. Data Ingestion: Loads dataset from HuggingFace
# 2. Text Processing: Cleans and normalizes movie descriptions
# 3. Feature Extraction: Builds TF-IDF matrix for semantic similarity
# 4. RecommenderModel: Core class with multiple recommendation strategies
# 5. Evaluation: Measures diversity (ILS) and catalog coverage
# =============================================================================

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
# Download required NLTK resources if not already present
# - stopwords: Common words to filter out (the, is, at, etc.)
# - wordnet: Lexical database for lemmatization (reducing words to base form)
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
# Configure file-based logging with timestamps for debugging and audit trails
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
    """
    Extract detailed error information including file name and line number.
    This helps pinpoint exactly where an exception occurred in the pipeline.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    return f"Error in script [{file_name}] at line [{line_number}]: {str(error)}"

class CustomException(Exception):
    """
    Custom exception class that logs errors with full context before raising.
    Ensures all critical failures are captured in the log file for debugging.
    """
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)
        logger.error(self.error_message)  # Log the full error trace
    
    def __str__(self):
        return self.error_message

# ── Utilities ───────────────────────────────────────────────────────────────
def save_object(file_path: str, obj) -> None:
    """
    Serialize and save any Python object using pickle.
    Used for persisting trained models, vectorizers, and processed dataframes.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
        logger.info(f"Object saved → {file_path}")
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path: str):
    """
    Load a previously saved pickle object from disk.
    Includes validation to ensure the file exists before attempting load.
    """
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
    """
    Calculate Intra-List Similarity (ILS) for a set of recommended items.
    
    ILS measures how similar the items in a recommendation list are to each other.
    - Lower ILS = more diverse recommendations (desirable)
    - Higher ILS = redundant/similar recommendations
    
    Formula: Average pairwise cosine similarity excluding self-comparisons.
    """
    indices = [int(i) for i in indices]
    if len(indices) < 2:
        return 0.0  # Cannot compute similarity with < 2 items
    
    # Extract TF-IDF vectors for the recommended items
    sub = tfidf_matrix[indices]
    # Compute pairwise cosine similarity matrix
    sim = cosine_similarity(sub)
    # Zero out diagonal (item compared to itself = 1.0, which would skew average)
    np.fill_diagonal(sim, 0)
    # Return mean of all off-diagonal similarities
    return float(sim.sum() / (len(indices) * (len(indices) - 1)))

# ── Text Cleaning ───────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Preprocess movie description text for TF-IDF vectorization.
    
    Steps:
    1. Convert to lowercase for case-insensitive matching
    2. Remove non-alphabetic characters (punctuation, numbers, special chars)
    3. Collapse multiple whitespace into single spaces
    4. Tokenize, remove stopwords, and lemmatize remaining words
    5. Filter out very short tokens (length ≤ 2) to reduce noise
    
    Returns: Cleaned, space-separated string of meaningful tokens.
    """
    if not isinstance(text, str) or not text.strip():
        return " "
    
    # Normalize case
    text = text.lower()
    # Remove non-alphabetic characters (keep only letters and spaces)
    text = re.sub(r"[^a-z\s]", " ", text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    # Tokenize, filter stopwords, and lemmatize
    tokens = [
        lemmatizer.lemmatize(w)
        for w in text.split()
        if w not in STOP_WORDS and len(w) > 2  # Keep words with 3+ chars
    ]
    return " ".join(tokens)

# ── Data Ingestion ──────────────────────────────────────────────────────────
@dataclass
class DataIngestionConfig:
    """
    Configuration container for data ingestion paths and settings.
    Uses dataclass for clean, type-hinted configuration management.
    """
    raw_data_path: str = os.path.join("artifacts", "raw_train.csv")
    val_data_path: str = os.path.join("artifacts", "raw_val.csv")
    dataset_name: str = "jquigl/imdb-genres"  # HuggingFace dataset identifier
    # Map dataset column names to our internal schema for consistency
    COLUMN_RENAME_MAP = {
        "movie title - year": "title",
        "rating": "ratings",
    }

class DataIngestion:
    """
    Responsible for fetching raw data from external sources and saving locally.
    Acts as the first step in the ETL (Extract, Transform, Load) pipeline.
    """
    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Main execution method: downloads dataset, standardizes columns, saves CSVs.
        
        Returns:
            tuple: Paths to saved train and validation CSV files.
        """
        logger.info("DataIngestion: starting ")
        try:
            logger.info(f"Fetching dataset: {self.config.dataset_name} ")
            # Load dataset from HuggingFace Hub (train/validation splits)
            dataset = load_dataset(self.config.dataset_name)
            train_df = pd.DataFrame(dataset["train"])
            val_df = pd.DataFrame(dataset["validation"])
            
            # Standardize column names across both splits using our mapping
            for df in (train_df, val_df):
                df.rename(columns=self.config.COLUMN_RENAME_MAP, inplace=True)
                df.columns = df.columns.str.strip()  # Remove trailing spaces

            # Ensure output directory exists and save raw data for reproducibility
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
    """Configuration for processed data and model artifact paths."""
    processed_data_path: str = os.path.join("artifacts", "processed_movies.pkl")
    vectorizer_path: str = os.path.join("artifacts", "tfidf_vectorizer.pkl")
    tfidf_matrix_path: str = os.path.join("artifacts", "tfidf_matrix.pkl")

class DataTransformation:
    """
    Handles text preprocessing, feature extraction, and data sanitization.
    Transforms raw CSV into TF-IDF matrix ready for similarity computations.
    """
    def __init__(self):
        self.config = DataTransformationConfig()

    def _ensure_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fallback column renaming for edge cases where column names have trailing spaces.
        Defensive programming to handle inconsistent dataset schemas.
        """
        rename = {}
        if "movie title - year " in df.columns and "title " not in df.columns:
            rename["movie title - year "] = "title "
        if "rating " in df.columns and "ratings " not in df.columns:
            rename["rating "] = "ratings "
        if rename:
            df = df.rename(columns=rename)
        return df

    def initiate_data_transformation(self, train_path: str):
        """
        Main transformation workflow:
        1. Load and validate raw data
        2. Remove duplicates and missing values
        3. Apply text cleaning to descriptions
        4. Fit TF-IDF vectorizer and transform corpus
        5. Save all artifacts for model training
        
        Returns:
            tuple: (cleaned dataframe, fitted vectorizer, TF-IDF matrix)
        """
        logger.info("DataTransformation: starting ")
        try:
            df = pd.read_csv(train_path)
            df = self._ensure_columns(df)
            
            # Validate required columns exist for processing
            required = ["title", "description"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise ValueError(f"Required columns missing: {missing}")

            # Data quality checks: remove invalid/duplicate entries
            df = df.dropna(subset=["description"]).reset_index(drop=True)
            df = df.drop_duplicates(subset=["title", "description"], keep="first")
            df = df.drop_duplicates(subset=["title"], keep="first")  # Keep one per title
            df = df.reset_index(drop=True)

            # Apply text cleaning to prepare for vectorization
            logger.info("Cleaning description text… ")
            df["clean_description"] = df["description"].apply(clean_text)
            # Remove rows where cleaning resulted in empty text
            df = df[df["clean_description"].str.strip() != ""].reset_index(drop=True)

            # Configure and fit TF-IDF vectorizer
            # Key parameters:
            # - ngram_range=(1,2): Capture both unigrams and bigrams (e.g., "sci-fi")
            # - max_features=20000: Limit vocabulary size for efficiency
            # - sublinear_tf=True: Apply log scaling to term frequency (reduces bias toward long docs)
            # - min_df=2: Ignore terms appearing in only 1 document (noise reduction)
            logger.info("Fitting TF-IDF vectorizer… ")
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=20_000,
                sublinear_tf=True,
                min_df=2,
            )
            tfidf_matrix = vectorizer.fit_transform(df["clean_description"])

            # Persist all artifacts for later use in inference/training
            save_object(self.config.processed_data_path, df)
            save_object(self.config.vectorizer_path, vectorizer)
            save_object(self.config.tfidf_matrix_path, tfidf_matrix)

            logger.info("DataTransformation: complete ")
            return df, vectorizer, tfidf_matrix
        except Exception as e:
            raise CustomException(e, sys)

# ── Model Definition (Must match app.py for pickle compatibility) ───────────
class RecommenderModel:
    """
    Core recommendation engine supporting multiple query types:
    - recommend_by_title: Find similar movies to a given title
    - recommend_by_description: Match movies based on plot description
    - recommend_by_genre: Filter and rank movies by genre
    - title_suggestions: Autocomplete/search-as-you-type functionality
    
    Design Notes:
    - Uses cosine similarity on TF-IDF vectors for content-based filtering
    - Title matching is fuzzy to handle year suffixes and case variations
    - Genre filtering is applied as a post-processing mask on similarity scores
    """
    
    def __init__(self, df: pd.DataFrame, vectorizer, tfidf_matrix):
        """
        Initialize the recommender with preprocessed data and features.
        
        Args:
            df: Cleaned dataframe with movie metadata
            vectorizer: Fitted TfidfVectorizer for transforming new text
            tfidf_matrix: Sparse matrix of TF-IDF features for all movies
        """
        self.df = df.reset_index(drop=True)
        self.vectorizer = vectorizer
        self.tfidf_matrix = tfidf_matrix
        
        # Build efficient title->index lookup with deduplication
        # Handles case-insensitive matching and removes duplicate titles
        _titles = self.df["title"].astype(str).str.strip().str.lower()
        _mask = ~_titles.duplicated(keep="first")
        self.title_index = pd.Series(
            self.df.index[_mask].tolist(),
            index=_titles[_mask].tolist()
        )

    @staticmethod
    def _normalise(text: str) -> str:
        """Standardize text for matching: lowercase and strip whitespace."""
        return str(text).strip().lower()

    @staticmethod
    def _name_only(full_title: str) -> str:
        """
        Extract movie name by removing year suffix (e.g., "Movie (2023)" → "movie").
        Enables matching queries like "inception" to "Inception (2010)".
        """
        t = str(full_title).strip().lower()
        return re.sub(r"\s*-\s*\d{4}\s*$", "", t).strip()

    def _safe_idx(self, k: str) -> int:
        """
        Safely extract integer index from title_index lookup.
        Handles edge cases where pandas returns Series instead of scalar.
        """
        val = self.title_index[k]
        if hasattr(val, "iloc"):
            val = val.iloc[0]
        return int(val)

    def find_title_index(self, query: str):
        """
        Multi-strategy title matching with fallbacks:
        1. Exact match (normalized)
        2. Match after removing year suffix
        3. Match if all query words appear in title (bag-of-words)
        4. Substring match (query contained in title)
        
        Returns:
            tuple: (movie_index, matched_title) or (None, None) if not found
        """
        q = self._normalise(query)
        
        # Strategy 1: Exact normalized match
        if q in self.title_index:
            return self._safe_idx(q), q
        
        # Strategy 2: Match ignoring year suffix
        for full_key in self.title_index.index:
            if self._name_only(full_key) == q:
                return self._safe_idx(full_key), full_key
        
        # Strategy 3: All query words must appear in title (order-independent)
        query_words = q.split()
        for full_key in self.title_index.index:
            name = self._name_only(full_key)
            if all(w in name for w in query_words):
                return self._safe_idx(full_key), full_key
        
        # Strategy 4: Simple substring containment
        for full_key in self.title_index.index:
            if q in full_key:
                return self._safe_idx(full_key), full_key
        
        # All strategies failed
        return None, None

    def _genre_col(self):
        """
        Dynamically detect which column contains genre information.
        Supports multiple possible column names for flexibility.
        """
        for c in ["expanded-genres", "genre", "genres"]:
            if c in self.df.columns:
                return c
        return None

    def _genre_mask(self, scores, genre_filter):
        """
        Apply genre filtering by masking out non-matching movies.
        
        Mechanism: Set similarity score to -1 for movies not matching the genre.
        Since _top_n() filters out negative scores, this effectively excludes them.
        
        Args:
            scores: Array of similarity scores for all movies
            genre_filter: Genre string to filter by (or None for no filter)
        
        Returns:
            Modified scores array with non-matching genres masked
        """
        if not genre_filter or genre_filter == "All":
            return scores  # No filtering requested
        
        col = self._genre_col()
        if col is None:
            return scores  # No genre column available
        
        # Create boolean mask: True where genre matches (case-insensitive substring)
        mask = self.df[col].astype(str).str.lower().str.contains(genre_filter.lower(), na=False).values
        # Mask non-matching entries by setting score to -1 (will be filtered later)
        scores[~mask] = -1
        return scores

    def _top_n(self, scores, n):
        """
        Extract top-N recommendations from similarity scores.
        
        Logic:
        1. Sort scores in descending order
        2. Filter out masked items (score = -1) and invalid indices
        3. Return dataframe with top N results + similarity scores rounded to 4 decimals
        """
        n_movies = len(self.df)
        # Get indices sorted by score (highest first), with filtering
        top = [
            int(i) for i in scores.argsort()[::-1]
            if float(scores[i]) > -1 and int(i) < n_movies
        ][:n]
        
        if not top:
            return pd.DataFrame()  # No valid recommendations
        
        # Build result dataframe with selected columns
        result = self.df.iloc[top][self._output_cols()].copy()
        result["similarity_score"] = [round(float(scores[i]), 4) for i in top]
        return result.reset_index(drop=True)

    def _output_cols(self):
        """Define which columns to include in recommendation results."""
        base = ["title", "description"]
        # Include optional metadata columns if they exist in the dataframe
        extra = [c for c in ["ratings", "genre", "expanded-genres"] if c in self.df.columns]
        return base + extra

    def recommend_by_title(self, title: str, genre_filter: str = None, n: int = 5):
        """
        Content-based recommendation: find movies similar to a given title.
        
        Process:
        1. Resolve title to movie index using fuzzy matching
        2. Compute cosine similarity between target movie and all others
        3. Apply genre filter if specified
        4. Return top-N most similar movies
        
        Args:
            title: Movie title to find recommendations for
            genre_filter: Optional genre to restrict results
            n: Number of recommendations to return
        
        Returns:
            DataFrame with recommended movies and similarity scores
        """
        idx, matched = self.find_title_index(title)
        if idx is None:
            return pd.DataFrame()  # Title not found
        
        # Compute similarity between target movie and entire corpus
        scores = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        scores[idx] = -1  # Exclude the movie itself from recommendations
        scores = self._genre_mask(scores, genre_filter)  # Apply genre filter
        return self._top_n(scores, n)

    def recommend_by_description(self, description: str, genre_filter: str = None, n: int = 5):
        """
        Plot-based recommendation: find movies matching a plot description.
        
        Use case: "Find movies about time travel and romance"
        
        Process:
        1. Clean and vectorize the input description using the fitted TF-IDF vectorizer
        2. Compute similarity between query vector and all movie vectors
        3. Apply genre filtering and return top-N results
        
        Note: Uses the same clean_text() function as training for consistency.
        """
        cleaned = clean_text(description)
        if not cleaned:
            return pd.DataFrame()  # Empty query after cleaning
        
        # Transform query into TF-IDF space using the trained vectorizer
        vec = self.vectorizer.transform([cleaned])
        # Compute similarity between query and all movies
        scores = cosine_similarity(vec, self.tfidf_matrix).flatten()
        scores = self._genre_mask(scores, genre_filter)
        return self._top_n(scores, n)

    def recommend_by_genre(self, genre: str, n: int = 5):
        """
        Genre-based browsing: return top-rated movies in a specific genre.
        
        Unlike the other methods, this does NOT use similarity scoring.
        Instead, it filters by genre and sorts by rating (if available).
        
        Args:
            genre: Genre name to filter by (case-insensitive substring match)
            n: Number of results to return
        
        Returns:
            DataFrame of top movies in the genre, sorted by rating
        """
        col = self._genre_col()
        if col is None:
            return pd.DataFrame()
        
        # Filter movies containing the genre (case-insensitive substring match)
        mask = self.df[col].astype(str).str.lower().str.contains(genre.lower(), na=False)
        subset = self.df[mask].copy()
        
        if subset.empty:
            return pd.DataFrame()
        
        # Remove duplicate titles, keeping highest-rated version
        subset = subset.drop_duplicates(subset=["title"], keep="first")
        
        # Sort by rating if available, otherwise return in original order
        rating_col = "ratings" if "ratings" in subset.columns else None
        if rating_col:
            subset = subset.sort_values(rating_col, ascending=False)
        
        # Add placeholder similarity_score (not applicable for genre browsing)
        subset["similarity_score"] = None
        return subset[self._output_cols()].head(n).reset_index(drop=True)

    def all_genres(self):
        """
        Extract and return all unique genres from the dataset.
        
        Handles multiple delimiter formats (comma, pipe, slash) and
        returns sorted list for UI dropdowns or autocomplete.
        """
        col = self._genre_col()
        if col is None:
            return []
        
        # Split genre strings by common delimiters, explode to individual genres
        return sorted(
            self.df[col].dropna()
            .astype(str).str.split(r"[,|/]").explode()  # Split on , or | or /
            .str.strip().value_counts().index.tolist()   # Clean and count unique
        )

    def title_suggestions(self, query: str, limit: int = 8):
        """
        Autocomplete functionality for search-as-you-type UI.
        
        Prioritization:
        1. Titles that START with the query (more relevant)
        2. Titles that CONTAIN the query anywhere (fallback)
        
        Returns deduplicated list up to the specified limit.
        """
        q = self._normalise(query)
        if not q:
            return []
        
        starts, contains = [], []
        seen = set()  # Track seen titles to avoid duplicates
        
        for raw in self.df["title"].astype(str).tolist():
            if raw in seen:
                continue
            name = self._name_only(raw)  # Compare without year suffix
            
            if name.startswith(q):
                starts.append(raw)
                seen.add(raw)
            elif q in name:
                contains.append(raw)
                seen.add(raw)
        
        # Return starts first (more relevant), then contains, up to limit
        return (starts + contains)[:limit]

    def stats(self) -> dict:
        """
        Return diagnostic statistics about the trained model.
        Useful for logging, monitoring, and debugging.
        """
        return {
            "total_movies": len(self.df),
            "tfidf_features": self.tfidf_matrix.shape[1],  # Vocabulary size
            "total_genres": len(self.all_genres()),
            # Sparsity: percentage of zero values in TF-IDF matrix (typical for text data)
            "sparsity_pct": round(
                100 * (1 - self.tfidf_matrix.nnz / (self.tfidf_matrix.shape[0] * self.tfidf_matrix.shape[1])), 1
            ),
        }

# ── Model Trainer ───────────────────────────────────────────────────────────
@dataclass
class ModelTrainerConfig:
    """Configuration for model artifact storage path."""
    model_path: str = os.path.join("artifacts", "recommender_model.pkl")

class ModelTrainer:
    """
    Orchestrates model instantiation, evaluation, and persistence.
    Includes offline evaluation metrics to assess recommendation quality.
    """
    
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_training(self, df, vectorizer, tfidf_matrix):
        """
        Main training method: creates model, evaluates diversity/coverage, saves artifact.
        
        Evaluation Metrics:
        - Mean ILS (Intra-List Similarity): Measures recommendation diversity
          * Lower = more diverse recommendations (better)
          * Computed on sample of 50 random titles
        - Catalog Coverage: Percentage of unique movies ever recommended
          * Higher = better exploration of the catalog
        
        Returns:
            Trained RecommenderModel instance
        """
        logger.info("ModelTrainer: starting ")
        try:
            # Instantiate the recommender with preprocessed features
            model = RecommenderModel(df, vectorizer, tfidf_matrix)
            
            logger.info("Evaluating model… ")
            # Sample titles for offline evaluation (balance speed vs. representativeness)
            sample_titles = df["title"].sample(min(50, len(df)), random_state=42).tolist()
            ils_scores = []
            recommended = set()  # Track unique movies ever recommended

            for title in sample_titles:
                try:
                    recs = model.recommend_by_title(title, n=5)
                except Exception:
                    continue  # Skip titles that cause errors (robustness)
                
                if recs.empty:
                    continue
                
                # Track coverage: which movies appear in any recommendation list
                recommended.update(recs["title"].tolist())
                
                # Compute ILS for this recommendation list
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

            # Aggregate evaluation metrics
            mean_ils = float(np.mean(ils_scores)) if ils_scores else 0.0
            coverage = len(recommended) / len(df) * 100
            
            logger.info(f"Mean ILS (lower=more diverse) : {mean_ils:.4f} ")
            logger.info(f"Catalogue coverage            : {coverage:.1f}% ")

            # Persist the trained model for inference
            save_object(self.config.model_path, model)
            logger.info(f"Model saved → {self.config.model_path} ")
            
            # Console output for user feedback during training
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
    """
    High-level orchestrator that sequences all training steps.
    Implements the complete workflow: ingestion → transformation → training.
    """
    
    def initiate_training(self):
        """
        Execute the full training pipeline with error handling and logging.
        Each step's output becomes the next step's input (pipeline pattern).
        """
        try:
            logger.info("=" * 55)
            logger.info("  TRAIN PIPELINE: START ")
            logger.info("=" * 55)

            # Step 1: Fetch and save raw data
            logger.info("Step 1: Data Ingestion ")
            ingestion = DataIngestion()
            train_path, val_path = ingestion.initiate_data_ingestion()

            # Step 2: Clean text and build TF-IDF features
            logger.info("Step 2: Data Transformation ")
            transformation = DataTransformation()
            df, vectorizer, tfidf_matrix = transformation.initiate_data_transformation(train_path)

            # Step 3: Instantiate model, evaluate, and persist
            logger.info("Step 3: Model Training & Evaluation ")
            trainer = ModelTrainer()
            model = trainer.initiate_model_training(df, vectorizer, tfidf_matrix)

            logger.info("TRAIN PIPELINE: COMPLETE ")
            return model
        except Exception as e:
            raise CustomException(e, sys)

# ── Main Entry Point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Script entry point when run directly: python train.py
    
    Executes the full training pipeline and provides next-step instructions.
    """
    print("=" * 55)
    print("  NextFlix — Movie Recommendation System")
    print("  Starting training pipeline…")
    print("=" * 55)
    
    pipeline = TrainPipeline()
    pipeline.initiate_training()
    
    print("\n Training complete!")
    print(" Run the app with:  streamlit run app.py\n")