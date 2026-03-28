# =============================================================================
# NextFlix — Movie Recommendation System (Streamlit Frontend)
# =============================================================================
# This module provides the interactive web interface for the movie recommender.
# Built with Streamlit, it offers three search modes:
# 
# 1. By Movie Title: Find films similar to a known title
# 2. By Description: Search using natural language plot descriptions  
# 3. By Genre: Browse top-rated movies within a specific genre
#
# Architecture:
# - Loads pre-trained RecommenderModel from artifacts/ (created by train.py)
# - Uses @st.cache_resource to avoid reloading model on every interaction
# - Renders results with custom CSS for cinematic dark theme
# - Displays diagnostic stats and match confidence scores
# =============================================================================

import pandas as pd
import streamlit as st
from src.pipeline.predict_pipeline import PredictPipeline
import os

# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────────────────
# Configure browser tab title, favicon, and layout before any content renders
st.set_page_config(
    page_title="NextFlix — Movie Recommender",  # Browser tab title
    page_icon="🎬",                              # Tab favicon (emoji)
    layout="wide",                               # Use full width for card grid layout
    initial_sidebar_state="expanded",            # Show filters/sidebar by default
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — Cinematic Dark Theme
# ─────────────────────────────────────────────────────────────────────────────
# Inject custom CSS for branded visual design.
# Key design choices:
# - Dark background (#0a0a0f) reduces eye strain, evokes cinema atmosphere
# - Gold/amber accent gradient (#f5c842 → #c0392b) suggests "premium" feel
# - Bebas Neue font for headings: bold, condensed, movie-poster aesthetic
# - DM Sans for body: highly readable at small sizes
# - Hover effects on cards provide interactive feedback
# - Badge components use semi-transparent backgrounds with colored borders
st.markdown("""
<style>
/* Import cinematic fonts from Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');

/* Global resets: ensure consistent dark theme across all Streamlit elements */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0a0f;
    color: #e8e0d4;  /* Soft off-white for readability */
}

.stApp { background-color: #0a0a0f; }  /* Main app container */

/* Hero title: large gradient text with movie-poster styling */
.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 5rem; letter-spacing: 0.08em; line-height: 1;
    background: linear-gradient(135deg, #f5c842 0%, #e8834a 50%, #c0392b 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}

/* Subtitle: small caps, muted color, wide letter-spacing for elegance */
.hero-sub {
    font-size: 1rem; color: #777;
    letter-spacing: 0.25em; text-transform: uppercase;
    margin-top: 0.2rem; margin-bottom: 1.5rem;
}

/* Section labels: gold left-border accent for visual hierarchy */
.section-label {
    font-family: 'Bebas Neue', sans-serif; font-size: 1.4rem;
    letter-spacing: 0.15em; color: #f5c842;
    border-left: 4px solid #f5c842; padding-left: 0.75rem;
    margin-bottom: 1rem;
}

/* Movie result cards: gradient background, subtle border, hover animation */
.movie-card {
    background: linear-gradient(135deg, #15151e, #1a1a26);
    border: 1px solid #2a2a3a; border-radius: 12px;
    padding: 1.2rem 1.4rem; margin-bottom: 0.9rem;
    position: relative; transition: border-color 0.2s, transform 0.2s;
}
.movie-card:hover { border-color: #f5c842; transform: translateX(4px); }

/* Rank number: large, subtle, positioned top-right of card */
.card-rank {
    font-family: 'Bebas Neue', sans-serif; font-size: 3rem;
    color: #2a2a3a; position: absolute; top: 0.5rem; right: 1rem; line-height: 1;
}

/* Card typography hierarchy */
.card-title  { font-size: 1.1rem; font-weight: 600; color: #f0e8d8; margin-bottom: 0.3rem; }
.card-desc   { font-size: 0.82rem; color: #777; line-height: 1.5; margin-bottom: 0.6rem; }
.card-meta   { display: flex; gap: 0.5rem; flex-wrap: wrap; }

/* Badge components: colored pills for score/rating/genre metadata */
.badge       { font-size: 0.72rem; font-weight: 600; padding: 0.2rem 0.6rem; border-radius: 999px; letter-spacing: 0.05em; }
.badge-score  { background:#f5c84220; color:#f5c842; border:1px solid #f5c84260; }
.badge-rating { background:#e8834a20; color:#e8834a; border:1px solid #e8834a60; }
.badge-genre  { background:#6c63ff20; color:#a89cff; border:1px solid #6c63ff50; }

/* Sidebar styling: slightly lighter dark, right border separator */
section[data-testid="stSidebar"] {
    background-color: #0f0f18; border-right: 1px solid #1e1e2e;
}
section[data-testid="stSidebar"] label {
    color: #888 !important; font-size: 0.78rem;
    letter-spacing: 0.1em; text-transform: uppercase;
}

/* Form inputs: dark background, gold focus ring for accessibility */
.stTextInput>div>input,
.stTextArea>div>textarea,
.stSelectbox>div>div {
    background-color: #15151e !important;
    border: 1px solid #2a2a3a !important;
    color: #e8e0d4 !important; border-radius: 8px !important;
}
.stTextInput>div>input:focus,
.stTextArea>div>textarea:focus {
    border-color: #f5c842 !important;
    box-shadow: 0 0 0 1px #f5c84240 !important;  /* Subtle gold glow */
}

/* Primary button: gradient background, bold condensed font */
.stButton>button {
    background: linear-gradient(135deg,#f5c842,#e8834a) !important;
    color: #0a0a0f !important;
    font-family: 'Bebas Neue', sans-serif !important;
    letter-spacing: 0.12em !important; font-size: 1.1rem !important;
    border: none !important; border-radius: 8px !important;
    padding: 0.6rem 2rem !important; width: 100%;
}
.stButton>button:hover { opacity: 0.88; }  /* Subtle dim on hover */

/* Stats boxes: compact metric display with gold accent values */
.stat-box     { background:#15151e; border:1px solid #2a2a3a; border-radius:10px; padding:1rem; text-align:center; }
.stat-value   { font-family:'Bebas Neue',sans-serif; font-size:2.2rem; color:#f5c842; line-height:1; }
.stat-label   { font-size:0.72rem; color:#555; text-transform:uppercase; letter-spacing:0.1em; margin-top:0.25rem; }

/* Decorative gold divider line for section separation */
.gold-divider{ height:2px; background:linear-gradient(90deg,#f5c842,transparent); border:none; margin:1.5rem 0; }

/* Info boxes: subtle gold-tinted background for helper text */
.info-box {
    background: #1a1a10; border: 1px solid #f5c84240;
    border-radius: 8px; padding: 1rem 1.2rem; margin-bottom: 1rem;
    font-size: 0.85rem; color: #b8a060;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Model Loading Helper
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline():
    """
    Load the trained PredictPipeline with resource caching.
    
    Why @st.cache_resource?
    - Streamlit re-runs the entire script on every user interaction
    - Without caching, the model would reload from disk on every click (slow!)
    - cache_resource persists the loaded model across reruns for the session
    - show_spinner=False hides the default spinner since we use custom loading UI
    
    Returns:
        PredictPipeline: Initialized pipeline with loaded recommender model
    """
    pipeline = PredictPipeline()
    pipeline.load_model()  # Loads recommender_model.pkl from artifacts/
    return pipeline

# ─────────────────────────────────────────────────────────────────────────────
# Hero Header
# ─────────────────────────────────────────────────────────────────────────────
# Render branded title and subtitle using custom CSS classes defined above
st.markdown('<div class="hero-title">NextFlix</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Content-Based Movie Recommendation System · TF-IDF + Cosine Similarity</div>', unsafe_allow_html=True)
st.markdown('<hr class="gold-divider">', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Check Artifacts & Load Model
# ─────────────────────────────────────────────────────────────────────────────
# Defensive check: ensure trained model exists before proceeding
# Prevents confusing runtime errors if user runs app.py before train.py
if not os.path.exists(os.path.join("artifacts", "recommender_model.pkl")):
    st.error("⚠️ No trained model found. Please run `python train.py` first to train and save the model.")
    st.stop()  # Halt execution — no point continuing without model

# Load model with user feedback spinner
# The spinner is shown while load_pipeline() executes (first load only, then cached)
with st.spinner("Loading trained model from artifacts/..."):
    pipeline = load_pipeline()

# Fetch model statistics for display in dashboard
# These stats help users understand the scale and characteristics of the system
s = pipeline.stats()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar: Search Mode & Filters
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Search mode selector: radio buttons with custom styling
    st.markdown('<div class="section-label">SEARCH MODE</div>', unsafe_allow_html=True)
    mode = st.radio(
        "",  # Empty label since we use custom CSS label above
        ["By Movie Title", "By Description", "By Genre"],
        label_visibility="collapsed",  # Hide Streamlit's default label
    )
    
    # Filter controls section
    st.markdown('<div class="section-label" style="margin-top:1.5rem">FILTERS</div>', unsafe_allow_html=True)
    n_results = st.slider("Results", 3, 10, 5)  # Min=3, Max=10, Default=5

    # Decorative separator
    st.markdown("---")
    
    # Model info panel: technical details for curious users
    # Uses HTML for fine-grained styling not possible with Streamlit primitives
    st.markdown(f"""
    <div style="font-size:0.75rem;color:#444;line-height:1.8">
    <b style="color:#666">Model</b><br>TF-IDF · bigrams · 20k features<br>Cosine Similarity<br><br>
    <b style="color:#666">Dataset</b><br><a href="https://huggingface.co/datasets/jquigl/imdb-genres" style="color:#444">jquigl/imdb-genres</a><br>on Hugging Face<br><br>
    <b style="color:#666">Status</b><br><span style="color:#4caf7a">● Model loaded from disk</span>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Stats Dashboard Row
# ─────────────────────────────────────────────────────────────────────────────
# Display key model metrics in a 4-column grid using custom stat-box styling
c1, c2, c3, c4 = st.columns(4)
stats_data = [
    (c1, f"{s['total_movies']:,}", "Movies Indexed"),      # Corpus size
    (c2, f"{s['tfidf_features']:,}", "TF-IDF Features"),   # Vocabulary dimensionality
    (c3, s['total_genres'], "Genres"),                      # Genre coverage
    (c4, f"{s['sparsity_pct']}%", "Matrix Sparsity")       # % of zero values in TF-IDF matrix
]

# Render each stat box with consistent formatting
for col, val, label in stats_data:
    with col:
        st.markdown(
            f'<div class="stat-box"><div class="stat-value">{val}</div>'
            f'<div class="stat-label">{label}</div></div>',
            unsafe_allow_html=True,
        )

st.markdown('<hr class="gold-divider">', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Card Renderer Function
# ─────────────────────────────────────────────────────────────────────────────
def render_card(row, rank):
    """
    Render a single movie recommendation as a styled HTML card.
    
    This function encapsulates all presentation logic for result items,
    keeping the main app flow clean and maintainable.
    
    Args:
        row: pandas Series containing movie data from recommendation results
        rank: Integer position (1-based) for display as large rank number
    
    Card Components:
    1. Rank badge (top-right, large subtle number)
    2. Title (cleaned, with year suffix removed if present)
    3. Description (truncated to 200 chars for readability)
    4. Metadata badges: similarity score, rating, primary genre
    """
    # Extract and clean title: remove year suffix if formatted as "Title - Year"
    raw_title = str(row.get("title", "Unknown"))
    title = raw_title.rsplit(" - ", 1)[0] if " - " in raw_title else raw_title
    
    # Truncate description for card layout (prevent overflow)
    desc = str(row.get("description", ""))
    desc = desc[:200] + "..." if len(desc) > 200 else desc
    
    # Extract optional metadata fields with null-safety checks
    score = row.get("similarity_score")
    rating = row.get("ratings", row.get("rating", None))  # Support both column names

    # Extract primary genre: try multiple possible column names, take first genre
    genre_val = None
    for gc in ["expanded-genres", "genre", "genres"]:
        if gc in row and pd.notna(row.get(gc)):
            genre_val = str(row[gc]).split(",")[0].strip()  # Take first if multiple
            break

    # Build badge HTML string conditionally based on available data
    badges = ""
    if score is not None and pd.notna(score):
        # Convert similarity (0-1) to percentage match badge
        badges += f'<span class="badge badge-score">⚡ {int(float(score)*100)}% match</span> '
    if rating is not None and pd.notna(rating):
        try:
            # Format rating with 1 decimal place, star prefix
            badges += f'<span class="badge badge-rating">★ {float(rating):.1f}</span> '
        except:
            pass  # Silently skip if rating format is unexpected
    if genre_val:
        badges += f'<span class="badge badge-genre">{genre_val}</span>'

    # Render complete card HTML with CSS classes for styling
    st.markdown(f"""
    <div class="movie-card">
        <div class="card-rank">{rank:02d}</div>
        <div class="card-title">{title}</div>
        <div class="card-desc">{desc}</div>
        <div class="card-meta">{badges}</div>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Search Panel Logic (Mode-Dependent UI)
# ─────────────────────────────────────────────────────────────────────────────
# Initialize result containers
results_df = None  # Will hold recommendation results DataFrame
error_msg = None   # Will hold user-facing error message if any

# ── MODE 1: Search by Movie Title ───────────────────────────────────────────
if mode == "By Movie Title":
    st.markdown('<div class="section-label">FIND BY TITLE</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Type any part of a movie title — case does not matter. Pick from suggestions then hit Search.</div>', unsafe_allow_html=True)

    # Free-text input for title search
    title_input = st.text_input("", placeholder="e.g. dark knight, inception...", label_visibility="collapsed", key="title_input")

    # Autocomplete logic: show suggestions only after 2+ characters typed
    selected_title = None
    if title_input and len(title_input) >= 2:
        # Fetch matching titles from model's title index (fuzzy matching)
        suggestions = pipeline.title_suggestions(title_input)
        if suggestions:
            # Dropdown with empty option to allow manual entry override
            selected_title = st.selectbox(
                "Select a match:",
                options=[""] + suggestions,
                format_func=lambda x: "— choose a title —" if x == "" else x,
                key="title_select",
            )
        else:
            st.caption("No matching titles found.")  # Gentle feedback, not error

    # Determine final query: use selected suggestion if provided, else raw input
    final_title = selected_title if selected_title else title_input
    
    # Dynamic button text: "RECOMMEND" if title confirmed, else "SEARCH"
    if st.button("RECOMMEND" if selected_title else "SEARCH"):
        if final_title.strip():
            with st.spinner("Computing cosine similarity..."):
                # Execute recommendation: find movies similar to queried title
                results_df = pipeline.recommend_by_title(final_title.strip(), n=n_results)
                if results_df.empty:
                    error_msg = f'"{final_title}" not found.'

# ── MODE 2: Search by Description ───────────────────────────────────────────
elif mode == "By Description":
    st.markdown('<div class="section-label">FIND BY VAGUE DESCRIPTION</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Describe a movie in your own words and NextFlix will find the closest matches.</div>', unsafe_allow_html=True)
    
    # Multi-line text area for plot-like descriptions
    desc_input = st.text_area("", height=110, label_visibility="collapsed", placeholder="e.g. a crew has to rob a casino...")
    
    if st.button("FIND MOVIES"):
        if desc_input.strip():
            with st.spinner("Analyzing description..."):
                # Transform description to TF-IDF, compute similarity to all movies
                results_df = pipeline.recommend_by_description(desc_input.strip(), n=n_results)
        else:
            error_msg = "Please enter a description."  # Validation feedback

# ── MODE 3: Browse by Genre ─────────────────────────────────────────────────
else:  # By Genre
    st.markdown('<div class="section-label">BROWSE BY GENRE</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Pick a genre to surface the highest-rated titles.</div>', unsafe_allow_html=True)
    
    # Dropdown populated with all unique genres from dataset
    genre_pick = st.selectbox("", pipeline.all_genres(), label_visibility="collapsed")
    
    if st.button("SHOW TOP MOVIES"):
        with st.spinner("Fetching movies..."):
            # Filter by genre, sort by rating (not similarity), return top N
            results_df = pipeline.recommend_by_genre(genre_pick, n=n_results)

# ─────────────────────────────────────────────────────────────────────────────
# Results Rendering & Display
# ─────────────────────────────────────────────────────────────────────────────
# Show error message if search failed (e.g., title not found)
if error_msg:
    st.error(f"⚠️  {error_msg}")

# Render results only if we have a non-empty DataFrame
if results_df is not None and not results_df.empty:
    st.markdown('<hr class="gold-divider">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-label">TOP {len(results_df)} RECOMMENDATIONS</div>', unsafe_allow_html=True)

    # Two-column layout: cards on left (wider), analytics panel on right
    col_cards, col_panel = st.columns([3, 2])

    # ── Left Column: Movie Cards ────────────────────────────────────────────
    with col_cards:
        for rank, (_, row) in enumerate(results_df.iterrows(), 1):
            render_card(row, rank)  # Reuse card renderer for consistency

    # ── Right Column: Analytics Panel ───────────────────────────────────────
    with col_panel:
        st.markdown("**Score breakdown**")
        
        # Dynamically determine which columns to display based on availability
        display_cols = ["title"]  # Title is always shown
        
        # Add similarity score if present and has non-null values
        if "similarity_score" in results_df.columns and results_df["similarity_score"].notna().any():
            display_cols.append("similarity_score")
            
        # Add rating column if available (support both naming conventions)
        if "ratings" in results_df.columns:
            display_cols.append("ratings")
        elif "rating" in results_df.columns:
            display_cols.append("rating")

        # Create display table with selected columns only
        tbl = results_df[display_cols].copy()
        
        # Format similarity scores as percentages for readability
        if "similarity_score" in tbl.columns:
            tbl["similarity_score"] = tbl["similarity_score"].apply(
                lambda x: f"{float(x)*100:.1f}%" if pd.notna(x) else "—"
            )
            
        # Clean up column names for UI: replace underscores, title case
        tbl.columns = [c.replace("_", " ").title() for c in tbl.columns]
        
        # Render interactive dataframe with auto-sizing and hidden index
        st.dataframe(tbl, use_container_width=True, hide_index=True)

        # Render bar chart of match scores (only if similarity data exists)
        if "similarity_score" in results_df.columns and results_df["similarity_score"].notna().any():
            st.markdown("**Match scores**")
            # Prepare chart data: truncate long titles, convert score to percentage
            chart = pd.DataFrame({
                "Movie": results_df["title"].str[:22] + "…",  # Truncate for chart labels
                "Score": (results_df["similarity_score"].astype(float) * 100).round(1),
            })
            # Render horizontal bar chart with gold accent color
            st.bar_chart(chart.set_index("Movie"), color="#f5c842")