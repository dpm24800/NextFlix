# NextFlix — Movie Recommendation System

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2+-orange?logo=scikitlearn)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

NextFlix is an **end-to-end content-based movie recommendation system** that **suggests your next movie instantly**, built for the Aakash Group AI/ML Internship Task. It uses **TF-IDF vectorization** and **cosine similarity** on movie plot descriptions from the IMDB Genres dataset to recommend similar movies.

## Deployment

**Live Web App:** [https://dpm24800-nextflix.streamlit.app/](https://dpm24800-nextflix.streamlit.app/)

## Video Demo

<!-- <video src="demo/nextflix-demo.mp4" controls></video> -->

<a href="https://youtu.be/L9iB59upObg" target="_blank">
  <img src="https://img.youtube.com/vi/L9iB59upObg/maxresdefault.jpg" alt="Watch the video" style="width:100%; max-width:800px;">
</a>

<!-- [![Watch the video](https://img.youtube.com/vi/L9iB59upObg/0.jpg)](https://www.youtube.com/watch?v=L9iB59upObg) -->

## Preview
> Run `python train.py` then <br>
`streamlit run app.py` and take a screenshot here.

## Project Structure
Unlike standard scripts, this project uses a modular component-based pipeline:

- **Data Ingestion:** Fetches `jquigl/imdb-genres` from Hugging Face and handles schema normalization.
- **Data Transformation:** An NLP pipeline using NLTK for lemmatization and Scikit-learn for TF-IDF (Bigrams, 20k features).
- **Model Trainer:** Computes similarity matrices and calculates **Intra-List Similarity (ILS)** for diversity.
- **Predict Pipeline:** A lightweight abstraction for the Streamlit UI to serve real-time requests.

## Folder Structure
```
NextFlix/
├── artifacts/                        # Saved model & data artifacts (auto-created)
│   ├── raw_train.csv                 # Raw downloaded training data
│   ├── raw_val.csv                   # Raw validation data
│   ├── processed_movies.pkl          # Cleaned + preprocessed DataFrame
│   ├── tfidf_vectorizer.pkl          # Fitted TF-IDF vectorizer
│   ├── tfidf_matrix.pkl              # Computed TF-IDF sparse matrix
│   └── recommender_model.pkl         # Full RecommenderModel object ← used by app
├── logs/                             # Execution timestamps and error traces
│
├── notebook/
│   └── eda_and_modeling.ipynb        # Exploratory analysis + modeling rationale
│
├── src/
│   ├── __init__.py
│   ├── logger.py                     # Custom logging utility
│   ├── exception.py                  # Custom exception handler
│   ├── utils.py                      # save_object / load_object / ILS metric
│   │
│   ├── components/                   # Ingestion, Transformation, Trainer logic
│   │   ├── __init__.py
│   │   ├── data_ingestion.py         # Downloads dataset from Hugging Face → CSV
│   │   ├── data_transformation.py    # Cleans text + fits TF-IDF → pkl artifacts
│   │   └── model_trainer.py          # Builds RecommenderModel + evaluates → pkl
│   │
│   └── pipeline/                     # Train & Predict orchestrators
│       ├── __init__.py
│       ├── train_pipeline.py         # Orchestrates ingestion → transform → train
│       └── predict_pipeline.py       # Loads saved model → exposes recommend API
│
├── app.py                            # Streamlit web interface (loads from artifacts/)
├── train.py                          # Pipeline entry point: run this to train & save model
├── setup.py                          # Package configuration
├── requirements.txt
├── .gitignore
└── README.md
```

## Dependencies

```
numpy           — Matrix operations
pandas          — Data manipulation
matplotlib      — Visualizations (notebook)
seaborn         — EDA plots (notebook)
datasets        — Hugging Face dataset loading
scikit-learn    — TF-IDF vectorizer, cosine similarity
nltk            — Stopwords, lemmatisation
streamlit       — Web interface
```

## Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/dpm24800/nextflix.git
cd nextflix
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

### Step 1 — Train & Save the Model
```bash
python train.py
```

This runs the full pipeline:
1. Downloads `jquigl/imdb-genres` from Hugging Face
2. Cleans and preprocesses all movie descriptions
3. Fits the TF-IDF vectorizer
4. Evaluates the model (ILS + coverage metrics)
5. Saves **all artifacts** to `artifacts/`

You only need to run this **once**. The app reuses the saved model.

### Step 2 — Launch the Streamlit App
```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.


## Approach

### Dataset
- **Source:** [`jquigl/imdb-genres`](https://huggingface.co/datasets/jquigl/imdb-genres) via Hugging Face
- **Train split** used for building the model; <br> **validation split** available for testing
- **Key fields:** `movie title - year`, `genre`, `expanded-genres`, `ratings`, `description` 


### 1. Preprocessing Pipeline
To transform raw movie descriptions into meaningful numerical data, a robust NLP pipeline has been implemented within `src/components/data_transformation.py`:

  * **Normalization:** Lowercasing and removing special characters/digits to reduce noise.
  * **Stopword Removal:** Filtering out common English words (e.g., "the", "is") using NLTK.
  * **Lemmatization:** Reducing words to their dictionary root (e.g., "running" → "run") using the **WordNet Lemmatizer** to ensure semantic consistency.

<!-- | Step | What it does | Why |
|------|-------------|-----|
| **Lowercase** | `"Action" → "action"` | Removes case duplicates |
| **Punctuation removal** | Strips `!`, `.`, `,`, digits | Reduces noise |
| **Stopword removal** | Drops `"the"`, `"is"`, `"a"` etc. | Keeps only meaningful tokens |
| **Lemmatisation** | `"running" → "run"`, `"heroes" → "hero"` | Reduces vocabulary size, improves matching | -->


### 2. Vectorization Choice: TF-IDF
**TF-IDF (Term Frequency-Inverse Document Frequency)** has been chosen over Bag-of-Words (BoW) for the following reasons:

  * **Importance Weighting:** TF-IDF penalizes frequently occurring words across the whole dataset (like "movie" or "story") and boosts unique keywords that define a specific film's plot.
  * **Context:** Used **n-grams (1, 2)** to capture phrases (e.g., "science fiction") rather than just individual words.


<!-- 
## Why TF-IDF over Bag-of-Words?

**Bag-of-Words** counts raw word frequencies — so words like `"film"`, `"story"`, or `"man"` which appear in nearly every movie description receive very high weights, drowning out the words that actually distinguish one film from another.

**TF-IDF** addresses this by multiplying term frequency by the *inverse document frequency* — automatically down-weighting ubiquitous terms and **up-weighting rare but discriminating terms** like `"heist"`, `"dystopia"`, or `"samurai"`. This makes cosine similarity comparisons between descriptions much more meaningful.

**Configuration chosen:**
```python
TfidfVectorizer(
    ngram_range=(1, 2),   # Unigrams + bigrams: captures "serial killer", "road trip"
    max_features=20_000,  # Vocabulary cap for memory efficiency
    sublinear_tf=True,    # Log-normalises term frequency: 1 + log(tf)
    min_df=2,             # Drops terms appearing in only one document
)
``` 
-->

### Recommendation Logic

The system calculates the **Cosine Similarity** between the user's input vector and the entire movie matrix. This measures the cosine of the angle between two vectors, providing a similarity score from 0 to 1.

```
User Input
    │
    ▼
clean_text()  ─── lowercase, remove punct, stopwords, lemmatise
    │
    ▼
TF-IDF Vector  ─── transform via saved vectorizer
    │
    ▼
Cosine Similarity  ─── against all movies in tfidf_matrix
    │
    ▼
Top-N Movies  ─── sorted by similarity score, optional genre filter
```

Three input modes are supported:

| Mode | Input | How it works |
|------|-------|--------------|
| **By Title** | Movie title in dataset | Retrieves that movie's vector, finds nearest neighbours |
| **By Description** | Free-text plot description | Vectorises query, finds closest movie vectors |
| **By Genre** | Genre name | Filters by genre column, sorts by rating |

## Evaluation/Performance Metrics
The model has been evaluated using three distinct perspectives:

1.  **Cosine Similarity Score:** Quantifies how closely the recommended movie's description matches the target.
2.  **Intra-List Similarity (ILS):** Measures the **diversity** of the top 5 recommendations. A lower ILS indicates a more diverse set of results, preventing the "echo chamber" effect.
3.  **Catalogue Coverage:** The percentage of movies in the dataset that the recommender is capable of suggesting, ensuring the model doesn't just stick to a few "popular" clusters.


<!-- - **Catalog Coverage:** Measures the % of movies the system is capable of recommending.
- **ILS Score:** Evaluates the diversity of the recommendation list (lower = more diverse). -->

<!-- | Metric | Description | Target |
|--------|-------------|--------|
| **Intra-List Similarity (ILS)** | Avg pairwise cosine similarity within Top-5 recommendations. Lower = more diverse, less redundant. | < 0.25 |
| **Avg Recommendation Score** | Mean cosine similarity of returned results to the query. Higher = more relevant. | > 0.20 |
| **Catalogue Coverage** | % of movies that ever appear in any recommendation list. Higher = less popularity bias. | > 50% | -->

### Why ILS?
Since there are no ground-truth user ratings available, ILS is a proxy for recommendation quality. A list of 5 near-identical movies (high ILS) is useless; a diverse list (low ILS) gives the user genuine options.

## Artifact Management

All serialized objects are stored in `artifacts/` using Python's `pickle`:

| File | Contents | Used by |
|------|----------|---------|
| `raw_train.csv` | Raw IMDB train data | Ingestion audit trail |
| `raw_val.csv` | Raw IMDB validation data | Ingestion audit trail |
| `processed_movies.pkl` | Cleaned DataFrame with `clean_description` | Model loading |
| `tfidf_vectorizer.pkl` | Fitted `TfidfVectorizer` | Query vectorisation |
| `tfidf_matrix.pkl` | Sparse TF-IDF matrix (movies × features) | Cosine similarity search |
| `recommender_model.pkl` | Full `RecommenderModel` object | Streamlit app |

The Streamlit app loads **only** `recommender_model.pkl` via `PredictPipeline` — no re-training, no re-downloading.

## Limitations & Future Improvements

### Current Limitations

1. **No semantic understanding** — TF-IDF treats `"car chase"` and `"automobile pursuit"` as completely different; it cannot infer meaning.
2. **Cold-start problem** — Movies with very short or generic descriptions produce poor-quality sparse vectors.
3. **No personalisation** — Purely content-based; ignores individual user history or preferences.
4. **O(n²) similarity** — Computing full cosine similarity is quadratic; impractical beyond ~100k movies without approximation.
5. **Description quality bias** — Movies with longer, richer plot summaries will dominate recommendations regardless of quality.

### Potential Improvements
<!-- ## How to Improve Accuracy & Scalability -->


| Improvement | Expected Benefit |
| :--- | :--- |
| **Sentence Transformers** | Uses BERT-style embeddings to understand deep semantic meaning. |
| **FAISS (Vector DB)** | Enables Approximate Nearest Neighbor search for sub-millisecond retrieval at scale. |
| **Hybrid Filtering** | Combining content-based filtering with Collaborative Filtering (User-item ratings). |
| **Metadata Fusion** | Including Director, Cast, and Keywords in the vectorization process. |


<!-- | Improvement | Expected Benefit |
|-------------|----------------|
| **Sentence Transformers** (SBERT, all-MiniLM-L6) | True semantic similarity; handles paraphrases |
| **BM25** instead of TF-IDF | Better ranking for short queries |
| **FAISS / Annoy** (Approximate Nearest Neighbour) | O(log n) retrieval; scales to millions of movies |
| **Collaborative filtering** (ALS, SVD) | Adds user-preference signal beyond content |
| **Hybrid model** (content + CF) | Best-of-both accuracy |
| **Metadata fusion** (cast, director, keywords) | Richer feature representation | -->

## Author
Dipak Pulami Magar

## License
MIT License — see [LICENSE](LICENSE) for details.
