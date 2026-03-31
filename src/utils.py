import os
import sys
import pickle
import numpy as np
from src.logger import logger
from src.exception import CustomException


def save_object(file_path: str, obj) -> None:
    # Serialize any Python object to disk using pickle
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
        logger.info(f"Object saved → {file_path}")
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str):
    # Deserialize a pickled object from disk
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Artifact not found: {file_path}")
        with open(file_path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"Object loaded ← {file_path}")
        return obj
    except Exception as e:
        raise CustomException(e, sys)


def artifacts_exist(paths: list) -> bool:
    # Return True only if every path in the list exists."""
    return all(os.path.exists(p) for p in paths)


def intra_list_similarity(tfidf_matrix, indices: list) -> float:
    """
    Intra-List Similarity (ILS):
    Average pairwise cosine similarity within a recommendation list.
    Lower value = more diverse recommendations.
    """
    from sklearn.metrics.pairwise import cosine_similarity
    indices = [int(i) for i in indices]   # ensure plain Python ints
    if len(indices) < 2:
        return 0.0
    sub = tfidf_matrix[indices]
    sim = cosine_similarity(sub)
    np.fill_diagonal(sim, 0)
    return float(sim.sum() / (len(indices) * (len(indices) - 1)))