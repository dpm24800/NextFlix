import os
import sys
import re
import pandas as pd
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
nltk.download("stopwords", quiet=True)
nltk.download("wordnet",   quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from src.logger import logger
from src.exception import CustomException
from src.utils import save_object

STOP_WORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


@dataclass
class DataTransformationConfig:
    processed_data_path: str = os.path.join("artifacts", "processed_movies.pkl")
    vectorizer_path: str     = os.path.join("artifacts", "tfidf_vectorizer.pkl")
    tfidf_matrix_path: str   = os.path.join("artifacts", "tfidf_matrix.pkl")


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    # ── Text Cleaning ───────────────────────────────────────────────────────
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Full NLP preprocessing pipeline:
          1. Lowercase
          2. Remove punctuation / digits / special chars
          3. Remove stopwords
          4. Lemmatise tokens
        """
        if not isinstance(text, str) or not text.strip():
            return ""
        text = text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        tokens = [
            lemmatizer.lemmatize(w)
            for w in text.split()
            if w not in STOP_WORDS and len(w) > 2
        ]
        return " ".join(tokens)

    # ── Main Entry Point ────────────────────────────────────────────────────
    def initiate_data_transformation(self, train_path: str):
        """
        1. Load raw CSV
        2. Drop rows without descriptions
        3. Apply clean_text to description column
        4. Fit TF-IDF vectorizer on cleaned text
        5. Save: processed DataFrame, vectorizer, TF-IDF matrix
        Returns: (processed_df, vectorizer, tfidf_matrix)
        """
        logger.info("DataTransformation: starting")
        try:
            df = pd.read_csv(train_path)
            logger.info(f"Loaded raw data: {df.shape}")

            # ── Drop nulls ──────────────────────────────────────────
            initial_len = len(df)
            df = df.dropna(subset=["description"]).reset_index(drop=True)
            logger.info(f"Dropped {initial_len - len(df)} rows with null description")

            # ── Clean text ──────────────────────────────────────────
            logger.info("Cleaning description text…")
            df["clean_description"] = df["description"].apply(self.clean_text)
            df = df[df["clean_description"].str.strip() != ""].reset_index(drop=True)
            logger.info(f"After cleaning: {len(df)} movies retained")

            # ── TF-IDF Vectorisation ────────────────────────────────
            logger.info("Fitting TF-IDF vectorizer…")
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),   # unigrams + bigrams
                max_features=20_000,  # vocabulary cap
                sublinear_tf=True,    # log-normalise TF
                min_df=2,             # ignore very rare terms
            )
            tfidf_matrix = vectorizer.fit_transform(df["clean_description"])
            logger.info(f"TF-IDF matrix shape : {tfidf_matrix.shape}")
            sparsity = 100 * (1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1]))
            logger.info(f"Matrix sparsity     : {sparsity:.2f}%")

            # ── Save artifacts ──────────────────────────────────────
            save_object(self.config.processed_data_path, df)
            save_object(self.config.vectorizer_path,     vectorizer)
            save_object(self.config.tfidf_matrix_path,   tfidf_matrix)

            logger.info("DataTransformation: complete")
            return df, vectorizer, tfidf_matrix

        except Exception as e:
            raise CustomException(e, sys)
