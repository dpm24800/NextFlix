"""
train.py
─────────────────────────────────────────────────────────
Entry point for training the NextFlix recommendation model.

Usage:
    python train.py

What it does:
    1. Downloads jquigl/imdb-genres from Hugging Face
    2. Cleans & preprocesses movie descriptions
    3. Fits a TF-IDF vectorizer (bigrams, 20k features)
    4. Evaluates the model (ILS, coverage)
    5. Saves ALL artifacts to artifacts/ directory:
         - raw_train.csv / raw_val.csv
         - processed_movies.pkl
         - tfidf_vectorizer.pkl
         - tfidf_matrix.pkl
         - recommender_model.pkl

Once training is complete, run the app with:
    streamlit run app.py
"""

from src.pipeline.train_pipeline import TrainPipeline

if __name__ == "__main__":
    print("=" * 55)
    print("  NextFlix — Movie Recommendation System")
    print("  Starting training pipeline…")
    print("=" * 55)

    pipeline = TrainPipeline()
    pipeline.initiate_training()

    print("\n Training complete!")
    print(" Run the app with:  streamlit run app.py\n")
