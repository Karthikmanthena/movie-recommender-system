import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# CONFIG
# =========================

# Sample MovieLens movies.csv online (movieId, title, genres)
SAMPLE_DATA_URL = "https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/movielens_data/movies.csv"

REQUIRED_COLUMNS = ["movieId", "title", "genres"]


# =========================
# HELPER FUNCTIONS
# =========================

@st.cache_data(show_spinner=False)
def load_sample_data():
    df = pd.read_csv(SAMPLE_DATA_URL)
    return df


def validate_dataset(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return False, f"Missing columns: {', '.join(missing)}"
    return True, ""


@st.cache_data(show_spinner=False)
def build_similarity_matrix(movies_df):
    # Fill missing genres
    movies_df = movies_df.copy()
    movies_df["genres"] = movies_df["genres"].fillna("")

    # TF-IDF on genres
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies_df["genres"])

    # Cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Reset index so row index matches cosine_sim index
    movies_df = movies_df.reset_index(drop=True)

    # Mapping: title -> index
    indices = pd.Series(movies_df.index, index=movies_df["title"]).drop_duplicates()

    return movies_df, cosine_sim, indices


def get_recommendations(title, movies_df, cosine_sim, indices, n_recs=10):
    if title not in indices:
        return pd.DataFrame()

    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Skip first (itself)
    sim_scores = sim_scores[1 : n_recs + 1]

    movie_indices = [i[0] for i in sim_scores]
    scores = [round(i[1], 4) for i in sim_scores]

    result = movies_df.loc[movie_indices, ["movieId", "title", "genres"]].copy()
    result["similarity_score"] = scores
    return result


# =========================
# STREAMLIT UI
# =========================

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="centered",
)

st.title("🎬 Movie Recommendation System")
st.write(
    """
    A simple **content-based** movie recommender using `genres` and TF-IDF.  
    You can **upload your own dataset** or use the **sample MovieLens dataset**.
    """
)

# ---- Choose data source ----
st.sidebar.header("1️⃣ Choose Dataset Source")

data_source = st.sidebar.radio(
    "Select dataset option:",
    ("Use sample MovieLens dataset (online)", "Upload your own CSV"),
)

df_movies = None
data_status_message = ""

if data_source == "Use sample MovieLens dataset (online)":
    st.sidebar.success("Using online MovieLens movies.csv")
    try:
        df_movies = load_sample_data()
        ok, msg = validate_dataset(df_movies)
        if not ok:
            st.error("Sample dataset is invalid: " + msg)
            df_movies = None
    except Exception as e:
        st.error(f"Error loading sample dataset: {e}")

else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file",
        type=["csv"],
        help="CSV must contain columns: movieId, title, genres",
    )

    if uploaded_file is not None:
        try:
            df_movies = pd.read_csv(uploaded_file)
            ok, msg = validate_dataset(df_movies)
            if not ok:
                st.error("Uploaded dataset is invalid: " + msg)
                df_movies = None
            else:
                st.sidebar.success("Uploaded dataset loaded successfully ✅")
        except Exception as e:
            st.error(f"Error reading uploaded CSV: {e}")
    else:
        st.info("👈 Upload a CSV in the sidebar to continue.")

# ---- If we have data, show some info ----
if df_movies is not None:
    st.subheader("📊 Dataset Preview")
    st.write(f"Total movies: **{len(df_movies)}**")
    st.dataframe(df_movies.head())

    st.markdown("---")

    # ---- Build similarity matrix ----
    with st.spinner("Building similarity matrix..."):
        movies_clean, cosine_sim, indices = build_similarity_matrix(df_movies)

    st.subheader("🎥 Get Recommendations")

    # Movie selection
    titles_sorted = movies_clean["title"].dropna().unique()
    if len(titles_sorted) == 0:
        st.error("No titles found in dataset.")
    else:
        selected_title = st.selectbox(
            "Choose a movie title:",
            options=sorted(titles_sorted),
        )

        n_recs = st.slider(
            "Number of recommendations:",
            min_value=3,
            max_value=20,
            value=10,
            step=1,
        )

        if st.button("Recommend"):
            recs = get_recommendations(
                selected_title, movies_clean, cosine_sim, indices, n_recs
            )

            if recs.empty:
                st.warning("No recommendations found.")
            else:
                st.write(f"Top **{len(recs)}** movies similar to **{selected_title}**:")
                st.dataframe(recs)

else:
    st.warning("No valid dataset loaded yet.")
