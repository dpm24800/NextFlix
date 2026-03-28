import pandas as pd
import streamlit as st
from src.pipeline.predict_pipeline import PredictPipeline
import os

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NextFlix — Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — cinematic dark theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0a0f;
    color: #e8e0d4;
}

.stApp { background-color: #0a0a0f; }

.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 5rem; letter-spacing: 0.08em; line-height: 1;
    background: linear-gradient(135deg, #f5c842 0%, #e8834a 50%, #c0392b 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.hero-sub {
    font-size: 1rem; color: #777;
    letter-spacing: 0.25em; text-transform: uppercase;
    margin-top: 0.2rem; margin-bottom: 1.5rem;
}
.section-label {
    font-family: 'Bebas Neue', sans-serif; font-size: 1.4rem;
    letter-spacing: 0.15em; color: #f5c842;
    border-left: 4px solid #f5c842; padding-left: 0.75rem;
    margin-bottom: 1rem;
}
.movie-card {
    background: linear-gradient(135deg, #15151e, #1a1a26);
    border: 1px solid #2a2a3a; border-radius: 12px;
    padding: 1.2rem 1.4rem; margin-bottom: 0.9rem;
    position: relative; transition: border-color 0.2s, transform 0.2s;
}
.movie-card:hover { border-color: #f5c842; transform: translateX(4px); }
.card-rank {
    font-family: 'Bebas Neue', sans-serif; font-size: 3rem;
    color: #2a2a3a; position: absolute; top: 0.5rem; right: 1rem; line-height: 1;
}
.card-title  { font-size: 1.1rem; font-weight: 600; color: #f0e8d8; margin-bottom: 0.3rem; }
.card-desc   { font-size: 0.82rem; color: #777; line-height: 1.5; margin-bottom: 0.6rem; }
.card-meta   { display: flex; gap: 0.5rem; flex-wrap: wrap; }
.badge       { font-size: 0.72rem; font-weight: 600; padding: 0.2rem 0.6rem; border-radius: 999px; letter-spacing: 0.05em; }
.badge-score  { background:#f5c84220; color:#f5c842; border:1px solid #f5c84260; }
.badge-rating { background:#e8834a20; color:#e8834a; border:1px solid #e8834a60; }
.badge-genre  { background:#6c63ff20; color:#a89cff; border:1px solid #6c63ff50; }

section[data-testid="stSidebar"] {
    background-color: #0f0f18; border-right: 1px solid #1e1e2e;
}
section[data-testid="stSidebar"] label {
    color: #888 !important; font-size: 0.78rem;
    letter-spacing: 0.1em; text-transform: uppercase;
}

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
    box-shadow: 0 0 0 1px #f5c84240 !important;
}

.stButton>button {
    background: linear-gradient(135deg,#f5c842,#e8834a) !important;
    color: #0a0a0f !important;
    font-family: 'Bebas Neue', sans-serif !important;
    letter-spacing: 0.12em !important; font-size: 1.1rem !important;
    border: none !important; border-radius: 8px !important;
    padding: 0.6rem 2rem !important; width: 100%;
}
.stButton>button:hover { opacity: 0.88; }

.stat-box     { background:#15151e; border:1px solid #2a2a3a; border-radius:10px; padding:1rem; text-align:center; }
.stat-value   { font-family:'Bebas Neue',sans-serif; font-size:2.2rem; color:#f5c842; line-height:1; }
.stat-label   { font-size:0.72rem; color:#555; text-transform:uppercase; letter-spacing:0.1em; margin-top:0.25rem; }
.gold-divider{ height:2px; background:linear-gradient(90deg,#f5c842,transparent); border:none; margin:1.5rem 0; }

.info-box {
    background: #1a1a10; border: 1px solid #f5c84240;
    border-radius: 8px; padding: 1rem 1.2rem; margin-bottom: 1rem;
    font-size: 0.85rem; color: #b8a060;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Load model (cached — runs once per session)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline():
    pipeline = PredictPipeline()
    pipeline.load_model()
    return pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">NextFlix</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Content-Based Movie Recommendation System · TF-IDF + Cosine Similarity</div>', unsafe_allow_html=True)
st.markdown('<hr class="gold-divider">', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────────────────────
if not os.path.exists(os.path.join("artifacts", "recommender_model.pkl")):
    st.error("⚠️  No trained model found. Please run `python train.py` first to train and save the model.")
    st.stop()

with st.spinner("Loading trained model from artifacts/…"):
    pipeline = load_pipeline()

s = pipeline.stats()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-label">SEARCH MODE</div>', unsafe_allow_html=True)
    mode = st.radio(
        "", ["By Movie Title", "By Description", "By Genre"],
        label_visibility="collapsed",
    )
    st.markdown('<div class="section-label" style="margin-top:1.5rem">FILTERS</div>', unsafe_allow_html=True)
    all_genres   = ["All"] + pipeline.all_genres()
    n_results    = st.slider("Results", 3, 10, 5)

    st.markdown("---")
    st.markdown(f"""
    <div style="font-size:0.75rem;color:#444;line-height:1.8">
    <b style="color:#666">Model</b><br>TF-IDF · bigrams · 20k features<br>Cosine Similarity<br><br>
    <b style="color:#666">Dataset</b><br><a href="https://huggingface.co/datasets/jquigl/imdb-genres" style="color:#444">jquigl/imdb-genres</a><br>on Hugging Face<br><br>
    <b style="color:#666">Status</b><br><span style="color:#4caf7a">● Model loaded from disk</span>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Stats row
# ─────────────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
for col, val, label in zip(
    [c1, c2, c3, c4],
    [f"{s['total_movies']:,}", f"{s['tfidf_features']:,}", s['total_genres'], f"{s['sparsity_pct']}%"],
    ["Movies Indexed", "TF-IDF Features", "Genres", "Matrix Sparsity"],
):
    with col:
        st.markdown(
            f'<div class="stat-box"><div class="stat-value">{val}</div>'
            f'<div class="stat-label">{label}</div></div>',
            unsafe_allow_html=True,
        )

st.markdown('<hr class="gold-divider">', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Card renderer
# ─────────────────────────────────────────────────────────────────────────────
def render_card(row, rank):
    raw_title = str(row.get("title", "Unknown"))
    title  = raw_title.rsplit(" - ", 1)[0] if " - " in raw_title else raw_title
    desc   = str(row.get("description", ""))
    desc   = desc[:200] + "…" if len(desc) > 200 else desc
    score  = row.get("similarity_score")
    rating = row.get("ratings", row.get("rating", None))

    genre_val = None
    for gc in ["expanded-genres", "genre", "genres"]:
        if gc in row and pd.notna(row.get(gc)):
            genre_val = str(row[gc]).split(",")[0].strip()
            break

    badges = ""
    if score is not None and pd.notna(score):
        badges += f'<span class="badge badge-score">⚡ {int(float(score)*100)}% match</span> '
    if rating is not None and pd.notna(rating):
        try:
            badges += f'<span class="badge badge-rating">★ {float(rating):.1f}</span> '
        except (ValueError, TypeError):
            pass
    if genre_val:
        badges += f'<span class="badge badge-genre">{genre_val}</span>'

    st.markdown(f"""
    <div class="movie-card">
        <div class="card-rank">{rank:02d}</div>
        <div class="card-title">{title}</div>
        <div class="card-desc">{desc}</div>
        <div class="card-meta">{badges}</div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Search panel
# ─────────────────────────────────────────────────────────────────────────────
genre_filter = "All"
gf           = None
results_df = None
error_msg  = None

if mode == "By Movie Title":
    st.markdown('<div class="section-label">FIND BY TITLE</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Type any part of a movie title — case does not matter. Pick from suggestions then hit Search.</div>', unsafe_allow_html=True)

    title_input = st.text_input("", placeholder="e.g. dark knight, inception, para…", label_visibility="collapsed", key="title_input")

    selected_title = None
    if title_input and len(title_input) >= 2:
        suggestions = pipeline.title_suggestions(title_input)
        if suggestions:
            import re as _re
            def _display(raw):
                m = _re.search(r"-\s*(\d{4})\s*$", raw.strip())
                name = _re.sub(r"\s*-\s*\d{4}\s*$", "", raw.strip())
                return f"{name} ({m.group(1)})" if m else raw.strip()

            raw_options = [""] + suggestions
            selected_title = st.selectbox(
                "Select a match:",
                options=raw_options,
                format_func=lambda x: "— choose a title —" if x == "" else x,
                key="title_select",
            )
        else:
            st.caption("No matching titles found — try a different spelling.")

    final_title = selected_title if selected_title else title_input
    btn_label = "RECOMMEND" if selected_title else "SEARCH"
    search = st.button(btn_label)

    if search and final_title and final_title.strip():
        with st.spinner("Computing cosine similarity…"):
            results_df = pipeline.recommend_by_title(final_title.strip(), genre_filter=gf, n=n_results)
            if results_df.empty:
                error_msg = f'"{final_title}" not found — try picking from the suggestions dropdown above.'

elif mode == "By Description":
    st.markdown('<div class="section-label">FIND BY VAGUE DESCRIPTION</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Describe a movie in your own words — even vaguely — and NextFlix will find the closest matches.</div>', unsafe_allow_html=True)
    desc_input = st.text_area(
        "", height=110, label_visibility="collapsed",
        placeholder="e.g. a heist where the crew has to rob a casino in Las Vegas with an elaborate plan…",
    )
    if st.button("FIND MOVIES"):
        if desc_input.strip():
            with st.spinner("Analysing your description…"):
                results_df = pipeline.recommend_by_description(desc_input.strip(), genre_filter=gf, n=n_results)
                if results_df.empty:
                    error_msg = "No matches found. Try adding more plot details."
        else:
            error_msg = "Please enter a description first."

else:
    st.markdown('<div class="section-label">BROWSE BY GENRE</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Pick a genre and NextFlix will surface the highest-rated titles in that category.</div>', unsafe_allow_html=True)
    genre_pick = st.selectbox("", [g for g in all_genres if g != "All"], label_visibility="collapsed")
    if st.button("SHOW TOP MOVIES"):
        with st.spinner("Fetching top-rated movies…"):
            results_df = pipeline.recommend_by_genre(genre_pick, n=n_results)
            if results_df.empty:
                error_msg = f'No movies found for "{genre_pick}".'


# ─────────────────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────────────────
if error_msg:
    st.error(f"⚠️  {error_msg}")

if results_df is not None and not results_df.empty:
    st.markdown('<hr class="gold-divider">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-label">TOP {len(results_df)} RECOMMENDATIONS</div>', unsafe_allow_html=True)

    col_cards, col_panel = st.columns([3, 2])

    with col_cards:
        for rank, (_, row) in enumerate(results_df.iterrows(), 1):
            render_card(row, rank)

    with col_panel:
        st.markdown("**Score breakdown**")
        display_cols = ["title"]
        if "similarity_score" in results_df.columns and results_df["similarity_score"].notna().any():
            display_cols.append("similarity_score")
        if "ratings" in results_df.columns:
            display_cols.append("ratings")

        tbl = results_df[display_cols].copy()
        if "similarity_score" in tbl.columns:
            tbl["similarity_score"] = tbl["similarity_score"].apply(
                lambda x: f"{float(x)*100:.1f}%" if pd.notna(x) else "—"
            )
        tbl.columns = [c.replace("_", " ").title() for c in tbl.columns]
        st.dataframe(tbl, width='stretch', hide_index=True)

        if "similarity_score" in results_df.columns and results_df["similarity_score"].notna().any():
            st.markdown("**Match scores**")
            chart = pd.DataFrame({
                "Movie": results_df["title"].str[:22] + "…",
                "Score": (results_df["similarity_score"].astype(float) * 100).round(1),
            })
            st.bar_chart(chart.set_index("Movie"), color="#f5c842")