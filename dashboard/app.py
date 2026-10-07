from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pymongo import MongoClient
from scipy.sparse import csr_matrix, hstack, load_npz
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
RECOMMENDER_FILES = {
    "vectorizer": MODELS_DIR / "tfidf_vectorizer.pkl",
    "matrix": MODELS_DIR / "movie_tfidf_matrix.npz",
    "movies": MODELS_DIR / "movies_for_search.pkl",
}
CLASSIFIER_CANDIDATES = (
    MODELS_DIR / "airflow" / "best_model.pkl",
    MODELS_DIR / "best_model.pkl",
)
NUMERIC_FEATURES = (
    "runtime",
    "budget",
    "revenue",
    "popularity",
    "vote_average",
    "release_year",
    "release_month",
    "release_decade",
    "num_genres",
    "num_Keywords",
    "vote_count",
)


st.set_page_config(
    page_title="CineMind | Movie Intelligence",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --ink:#f3f2ed; --muted:#9da5b4; --panel:#151a24; --line:#27303d; --lime:#c8f169; --blue:#75a7ff; }
    html, body, [class*="css"] { font-family:Inter,ui-sans-serif,system-ui,sans-serif; }
    .stApp { background:radial-gradient(ellipse at 86% -10%,#27394b 0,transparent 36%),#0c1017; color:var(--ink); }
    [data-testid="stSidebar"] { background:#111720; border-right:1px solid var(--line); }
    [data-testid="stSidebar"] * { color:var(--ink); }
    h1,h2,h3 { font-family:ui-sans-serif,system-ui,sans-serif !important; letter-spacing:-.035em; }
    h1 { font-size:2.6rem !important; }
    .hero { padding:1.3rem 0 .9rem; }
    .eyebrow { color:var(--lime); font:500 .72rem ui-monospace,monospace; text-transform:uppercase; letter-spacing:.18em; }
    .hero-copy { color:var(--muted); font-size:1rem; max-width:760px; }
    .metric-card { background:linear-gradient(140deg,#1a2230,#141922); border:1px solid var(--line); border-radius:16px; padding:18px 20px; min-height:105px; }
    .metric-label { color:var(--muted); font:500 .68rem ui-monospace,monospace; letter-spacing:.1em; text-transform:uppercase; }
    .metric-value { color:var(--ink); font:600 1.8rem ui-sans-serif,system-ui,sans-serif; margin-top:8px; }
    .metric-note { color:var(--lime); font-size:.75rem; margin-top:2px; }
    .section-note { color:var(--muted); margin-top:-.55rem; margin-bottom:1.1rem; font-size:.88rem; }
    .pill { display:inline-block; border:1px solid #405036; background:#1c281a; color:var(--lime); border-radius:999px; padding:4px 10px; font:500 .7rem 'DM Mono',monospace; }
    div[data-testid="stTabs"] button { font-family:ui-monospace,monospace; }
    div[data-testid="stMetric"] { background:var(--panel); border:1px solid var(--line); padding:15px 17px; border-radius:14px; }
    div[data-testid="stMetricLabel"] { color:var(--muted); }
    div[data-testid="stMetricValue"] { font-family:'Space Grotesk',sans-serif; }
    div[data-testid="stDataFrame"], div[data-testid="stTable"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
    .stButton button, .stDownloadButton button { border-radius:10px; font-weight:600; }
    .stButton button[kind="primary"] { background:var(--lime); border-color:var(--lime); color:#11170d; }
    hr { border-color:var(--line); }
    </style>
    """,
    unsafe_allow_html=True,
)


def _as_items(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    if isinstance(value, dict):
        value = value.get("genres", value.get("keywords", [value]))
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith(("[", "{")):
            try:
                return _as_items(json.loads(text))
            except json.JSONDecodeError:
                pass
        value = re.split(r"[,;|]", text)
    if not isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
        value = [value]

    result: list[str] = []
    for item in value:
        if isinstance(item, dict):
            item = item.get("name", "")
        if item is None:
            continue
        text = str(item).strip()
        if text and text.lower() not in {"nan", "none"} and text not in result:
            result.append(text)
    return result


def _format_items(value: Any) -> str:
    return ", ".join(_as_items(value)) or "Not listed"


def prepare_movies(frame: pd.DataFrame) -> pd.DataFrame:
    movies = frame.copy()
    if "_id" in movies:
        movies = movies.drop(columns="_id")

    for column in ("title", "overview", "original_language"):
        if column not in movies:
            movies[column] = ""
        movies[column] = movies[column].fillna("").astype(str)
    for column in ("genres", "keywords"):
        if column not in movies:
            movies[column] = [[] for _ in range(len(movies))]
        movies[column] = movies[column].apply(_as_items)

    if "release_date" not in movies:
        movies["release_date"] = pd.NaT
    movies["release_date"] = pd.to_datetime(movies["release_date"], errors="coerce")

    for column in NUMERIC_FEATURES:
        if column not in movies:
            movies[column] = np.nan
        movies[column] = pd.to_numeric(movies[column], errors="coerce")
    if "release_year" not in movies or movies["release_year"].isna().all():
        movies["release_year"] = movies["release_date"].dt.year
    else:
        movies["release_year"] = movies["release_year"].fillna(
            movies["release_date"].dt.year
        )
    movies["release_month"] = movies["release_month"].fillna(
        movies["release_date"].dt.month
    )
    movies["release_decade"] = movies["release_decade"].fillna(
        (movies["release_year"] // 10) * 10
    )
    movies["decade_str"] = movies["release_decade"].apply(
        lambda year: f"{int(year)}s" if pd.notna(year) else "Unknown"
    )
    movies["num_genres"] = movies["genres"].apply(len)
    movies["num_Keywords"] = movies["keywords"].apply(len)
    movies["log_popularity"] = np.log1p(movies["popularity"].clip(lower=0))
    if "high_engagement" not in movies:
        threshold = movies["vote_count"].quantile(0.75)
        movies["high_engagement"] = (movies["vote_count"] >= threshold).astype(int)
    movies["high_engagement"] = pd.to_numeric(
        movies["high_engagement"], errors="coerce"
    ).fillna(0).astype(int)
    return movies.reset_index(drop=True)


@st.cache_data(ttl=300, show_spinner="Loading the movie catalog…")
def load_catalog() -> tuple[pd.DataFrame, str, str | None]:
    mongo_error: str | None = None
    uri = os.getenv("MONGODB_URI", "").strip()
    if uri:
        client: MongoClient | None = None
        try:
            client = MongoClient(
                uri,
                serverSelectionTimeoutMS=2500,
                connectTimeoutMS=2500,
            )
            client.admin.command("ping")
            database = client[os.getenv("MONGO_DB_NAME", "cin-mind")]
            collection = database[os.getenv("MONGO_COLLECTION", "cleaned_data")]
            documents = list(collection.find({}, {"_id": 0}))
            if documents:
                return prepare_movies(pd.DataFrame(documents)), "MongoDB", None
            mongo_error = "MongoDB is reachable, but the movie collection is empty."
        except Exception as exc:
            mongo_error = f"MongoDB unavailable ({type(exc).__name__})."
        finally:
            if client is not None:
                client.close()
    else:
        mongo_error = "MONGODB_URI is not configured."

    candidates = (
        (DATA_DIR / "processed" / "movies_with_features.pkl", "featured local dataset"),
        (DATA_DIR / "processed" / "movies_clean.pkl", "clean local dataset"),
        (DATA_DIR / "raw" / "all_movies.json", "raw local dataset"),
    )
    for path, label in candidates:
        if not path.is_file():
            continue
        try:
            if path.suffix == ".pkl":
                frame = pd.read_pickle(path)
            else:
                with path.open(encoding="utf-8") as source:
                    frame = pd.DataFrame(json.load(source)["movies"])
            if not frame.empty:
                return prepare_movies(frame), label, mongo_error
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            mongo_error = f"{mongo_error} Local fallback failed ({type(exc).__name__})."
    return pd.DataFrame(), "unavailable", mongo_error


@st.cache_resource(show_spinner="Loading classifier…")
def load_classifier(path_string: str) -> dict[str, Any]:
    bundle = joblib.load(path_string)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError("Classifier artifact must contain a 'model' entry.")
    if not bundle.get("numeric_features") or not bundle.get("categorical_features"):
        raise ValueError("Classifier artifact is missing its feature schema.")
    return bundle


def classifier_frame(movies: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    required = list(bundle["numeric_features"]) + list(bundle["categorical_features"])
    features = movies.copy()
    for column in required:
        if column not in features:
            features[column] = np.nan if column in bundle["numeric_features"] else ""
    for column in bundle["categorical_features"]:
        if column in {"genres", "keywords"}:
            features[column] = features[column].apply(_format_items)
        else:
            features[column] = features[column].fillna("").astype(str)
    return features[required]


@st.cache_data(show_spinner="Scoring the catalog…")
def score_catalog(
    movies: pd.DataFrame, classifier_path: str
) -> tuple[np.ndarray, np.ndarray]:
    bundle = load_classifier(classifier_path)
    features = classifier_frame(movies, bundle)
    model = bundle["model"]
    labels = np.asarray(model.predict(features))
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(features))
        classes = list(getattr(model, "classes_", [0, 1]))
        positive_index = classes.index(1) if 1 in classes else -1
        positive_scores = probabilities[:, positive_index]
    elif hasattr(model, "decision_function"):
        margins = np.asarray(model.decision_function(features)).ravel()
        positive_scores = 1 / (1 + np.exp(-np.clip(margins, -40, 40)))
    else:
        positive_scores = labels.astype(float)
    return labels, positive_scores


@st.cache_resource(show_spinner="Loading content recommendation artifacts…")
def load_recommender() -> tuple[Any, Any, pd.DataFrame]:
    missing = [str(path) for path in RECOMMENDER_FILES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing recommendation artifacts: " + ", ".join(missing))
    vectorizer = joblib.load(RECOMMENDER_FILES["vectorizer"])
    matrix = load_npz(RECOMMENDER_FILES["matrix"])
    movies = joblib.load(RECOMMENDER_FILES["movies"])
    if matrix.shape[0] != len(movies):
        raise ValueError(
            "Recommendation matrix and movie catalog have different row counts."
        )
    return vectorizer, matrix, prepare_movies(movies)


def recommend(
    vectorizer: Any,
    matrix: Any,
    movies: pd.DataFrame,
    query: str,
    limit: int,
    seed_index: int | None = None,
) -> pd.DataFrame:
    if seed_index is None:
        cleaned_query = re.sub(r"[^\w\s]", " ", query.lower())
        query_vector = vectorizer.transform([cleaned_query])
        scores = cosine_similarity(query_vector, matrix).ravel()
    else:
        scores = cosine_similarity(matrix.getrow(seed_index), matrix).ravel()
        scores[seed_index] = 0
    indices = np.argsort(scores)[::-1]
    indices = [i for i in indices if scores[i] > 0][:limit]
    if not indices:
        return pd.DataFrame()
    result = movies.iloc[indices].copy()
    result["similarity"] = scores[indices]
    return result


@st.cache_data(show_spinner="Grouping movies into clusters…")
def cluster_catalog(movies: pd.DataFrame, cluster_count: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(movies) < cluster_count:
        raise ValueError("Choose no more clusters than there are movies.")
    numeric_columns = [
        column
        for column in (
            "runtime",
            "budget",
            "revenue",
            "popularity",
            "vote_average",
            "release_year",
            "vote_count",
            "num_genres",
            "num_Keywords",
        )
        if column in movies and movies[column].notna().any()
    ]
    if not numeric_columns:
        raise ValueError("There are not enough numeric movie fields to cluster.")
    numeric = movies[numeric_columns].replace([np.inf, -np.inf], np.nan)
    numeric = numeric.fillna(numeric.median()).fillna(0)
    scaled = StandardScaler().fit_transform(numeric)

    text = (
        movies["overview"].fillna("").astype(str)
        + " "
        + movies["genres"].apply(_format_items)
        + " "
        + movies["keywords"].apply(_format_items)
    )
    try:
        text_matrix = TfidfVectorizer(
            max_features=1500, stop_words="english", min_df=1
        ).fit_transform(text)
        features = hstack([csr_matrix(scaled * 0.35), text_matrix], format="csr")
    except ValueError as exc:
        if "empty vocabulary" not in str(exc).lower():
            raise
        features = csr_matrix(scaled)

    labels = KMeans(
        n_clusters=cluster_count, random_state=42, n_init=10
    ).fit_predict(features)
    clustered = movies.copy()
    clustered["cluster"] = labels
    if min(features.shape) > 1:
        coordinates = TruncatedSVD(n_components=2, random_state=42).fit_transform(
            features
        )
        clustered["map_x"] = coordinates[:, 0]
        clustered["map_y"] = coordinates[:, 1]
    else:
        clustered["map_x"] = scaled[:, 0]
        clustered["map_y"] = scaled[:, 1] if scaled.shape[1] > 1 else 0
    summary = (
        clustered.groupby("cluster")
        .agg(
            movie_count=("title", "size"),
            mean_rating=("vote_average", "mean"),
            mean_popularity=("popularity", "mean"),
            mean_runtime=("runtime", "mean"),
        )
        .round(2)
    )
    return clustered, summary


def _metric(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _movie_table(frame: pd.DataFrame, limit: int = 100) -> pd.DataFrame:
    columns = [
        column
        for column in (
            "title",
            "release_year",
            "genres",
            "vote_average",
            "vote_count",
            "popularity",
            "overview",
        )
        if column in frame
    ]
    display = frame[columns].head(limit).copy()
    if "genres" in display:
        display["genres"] = display["genres"].apply(_format_items)
    if "overview" in display:
        display["overview"] = display["overview"].str.slice(0, 240)
    return display


def main() -> None:
    st.sidebar.markdown('<div class="eyebrow">CineMind / Analytics</div>', unsafe_allow_html=True)
    st.sidebar.title("Movie intelligence")
    if st.sidebar.button("↻ Refresh catalog", width="stretch"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    movies, data_source, load_warning = load_catalog()
    if movies.empty:
        st.title("CineMind")
        st.error("The movie catalog could not be loaded.")
        if load_warning:
            st.caption(load_warning)
        st.info(
            "Start MongoDB and populate the `cleaned_data` collection, or provide "
            "`data/processed/movies_clean.pkl` as a local fallback."
        )
        st.stop()

    if load_warning:
        st.sidebar.warning(f"Using {data_source}; {load_warning}")
    else:
        st.sidebar.success(f"Catalog source: {data_source}")

    years = movies["release_year"].dropna().astype(int)
    if years.empty:
        year_bounds = (1900, 2030)
    else:
        year_bounds = (int(years.min()), int(years.max()))
    if year_bounds[0] == year_bounds[1]:
        year_bounds = (year_bounds[0] - 1, year_bounds[1] + 1)

    genres = sorted({genre for values in movies["genres"] for genre in values})
    st.sidebar.markdown("---")
    st.sidebar.subheader("Catalog filters")
    selected_genres = st.sidebar.multiselect("Genres", genres, placeholder="All genres")
    year_range = st.sidebar.slider(
        "Release years",
        min_value=year_bounds[0],
        max_value=year_bounds[1],
        value=year_bounds,
    )
    min_rating = st.sidebar.slider("Minimum rating", 0.0, 10.0, 0.0, 0.5)

    filtered = movies[
        movies["release_year"].between(year_range[0], year_range[1], inclusive="both")
        & movies["vote_average"].fillna(0).ge(min_rating)
    ].copy()
    if selected_genres:
        filtered = filtered[
            filtered["genres"].apply(
                lambda values: any(genre in values for genre in selected_genres)
            )
        ]

    st.markdown('<div class="hero">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Film data • machine learning • discovery</div>', unsafe_allow_html=True)
    st.title("The movie intelligence desk")
    st.markdown(
        "Explore the catalog, estimate audience engagement, discover movie clusters, "
        "and find your next watch.",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
    st.caption(
        f"Showing {len(filtered):,} of {len(movies):,} titles · catalog loaded from {data_source}"
    )

    overview_tab, classify_tab, clusters_tab, recommend_tab = st.tabs(
        ["◈  Overview", "◎  Classification", "✳  Clusters", "⌕  Recommendations"]
    )

    with overview_tab:
        if filtered.empty:
            st.warning("No movies match these filters. Widen the filters in the sidebar.")
        else:
            rated = filtered["vote_average"].dropna()
            high_engagement = filtered["high_engagement"].mean() * 100
            metric_cols = st.columns(4)
            metrics = (
                ("TITLES", f"{len(filtered):,}", "in current view"),
                ("AVG. RATING", f"{rated.mean():.1f}" if not rated.empty else "—", "out of 10"),
                (
                    "HIGH ENGAGEMENT",
                    f"{high_engagement:.0f}%",
                    "catalog target class",
                ),
                (
                    "RELEASE SPAN",
                    f"{int(filtered['release_year'].min())}–{int(filtered['release_year'].max())}"
                    if filtered["release_year"].notna().any()
                    else "—",
                    "years represented",
                ),
            )
            for column, (label, value, note) in zip(metric_cols, metrics):
                with column:
                    _metric(label, value, note)

            left, right = st.columns(2)
            with left:
                st.subheader("The catalog, by year")
                release_counts = (
                    filtered.dropna(subset=["release_year"])
                    .groupby(filtered["release_year"].dropna().astype(int))
                    .size()
                    .rename("titles")
                )
                if not release_counts.empty:
                    st.area_chart(release_counts, color="#c8f169", height=280)
                else:
                    st.info("Release dates are not available for this selection.")
            with right:
                st.subheader("Most common genres")
                genre_counts = (
                    filtered[["genres"]]
                    .explode("genres")
                    .dropna()
                    .groupby("genres")
                    .size()
                    .sort_values(ascending=False)
                    .head(10)
                    .sort_values()
                    .rename("titles")
                )
                if not genre_counts.empty:
                    st.bar_chart(genre_counts, color="#75a7ff", height=280)
                else:
                    st.info("Genre information is not available.")

            left, right = st.columns([1, 1])
            with left:
                st.subheader("Rating distribution")
                rating_counts = (
                    pd.to_numeric(filtered["vote_average"], errors="coerce")
                    .dropna()
                    .round()
                    .value_counts()
                    .sort_index()
                    .rename("titles")
                )
                if not rating_counts.empty:
                    st.bar_chart(rating_counts, color="#d69cff", height=260)
            with right:
                st.subheader("Popularity meets rating")
                scatter = filtered[["popularity", "vote_average"]].dropna()
                if not scatter.empty:
                    st.scatter_chart(
                        scatter,
                        x="popularity",
                        y="vote_average",
                        height=260,
                        color="#c8f169",
                    )
                else:
                    st.info("Popularity and rating values are not available.")

            st.subheader("Movie catalog")
            st.dataframe(_movie_table(filtered), width="stretch", hide_index=True)
            st.download_button(
                "Download filtered catalog (CSV)",
                filtered.assign(
                    genres=filtered["genres"].apply(_format_items),
                    keywords=filtered["keywords"].apply(_format_items),
                ).to_csv(index=False).encode("utf-8"),
                file_name="cinemind_filtered_movies.csv",
                mime="text/csv",
            )

    with classify_tab:
        st.subheader("Audience engagement classifier")
        st.markdown(
            '<p class="section-note">Estimate whether a title is likely to land in the '
            'catalog’s high-engagement segment. This is a model score, not a guarantee '
            'of audience response.</p>',
            unsafe_allow_html=True,
        )
        model_path = next((path for path in CLASSIFIER_CANDIDATES if path.is_file()), None)
        if model_path is None:
            st.warning("No classifier artifact found in `models/`.")
        else:
            try:
                bundle = load_classifier(str(model_path))
                labels, scores = score_catalog(movies, str(model_path))
                choices = list(range(len(movies)))
                selected = st.selectbox(
                    "Choose a movie to score",
                    choices,
                    format_func=lambda index: (
                        f"{movies.iloc[index]['title']} "
                        f"({int(movies.iloc[index]['release_year']) if pd.notna(movies.iloc[index]['release_year']) else 'year unknown'})"
                    ),
                    key="classifier_movie",
                )
                selected_movie = movies.iloc[selected]
                probability = float(scores[selected])
                predicted_high = int(labels[selected]) == 1
                left, right = st.columns([1, 1])
                with left:
                    st.markdown(
                        f"### {selected_movie['title']}  "
                        f"<span class='pill'>{'HIGH ENGAGEMENT' if predicted_high else 'STANDARD ENGAGEMENT'}</span>",
                        unsafe_allow_html=True,
                    )
                    st.write(selected_movie["overview"] or "No plot summary available.")
                    st.caption(
                        f"{_format_items(selected_movie['genres'])} · "
                        f"{int(selected_movie['release_year']) if pd.notna(selected_movie['release_year']) else 'Year unknown'} · "
                        f"Rating {selected_movie['vote_average']:.1f}/10"
                        if pd.notna(selected_movie["vote_average"])
                        else _format_items(selected_movie["genres"])
                    )
                with right:
                    st.metric("High-engagement score", f"{probability:.1%}")
                    st.progress(min(max(probability, 0.0), 1.0))
                    st.caption(f"Model: {bundle.get('model_name', 'Saved classifier')}")
                scored_summary = pd.Series(labels).value_counts().sort_index()
                scored_summary.index = [
                    "Standard engagement" if int(label) == 0 else "High engagement"
                    for label in scored_summary.index
                ]
                st.subheader("Model view across the full catalog")
                st.bar_chart(scored_summary, color="#c8f169", height=220)
                st.caption(
                    f"Classifier artifact: `{model_path.relative_to(ROOT).as_posix()}`. "
                    "Displayed scores are model outputs and do not replace validation metrics."
                )
            except Exception as exc:
                st.error(f"Could not run the classifier ({type(exc).__name__}).")

    with clusters_tab:
        st.subheader("Movie neighborhoods")
        st.markdown(
            '<p class="section-note">Discover groups formed from film attributes, genres, '
            'keywords and plot summaries. Clusters are calculated from the current catalog '
            'when requested; they are not a saved classification model.</p>',
            unsafe_allow_html=True,
        )
        max_clusters = min(10, max(2, len(movies) - 1))
        cluster_count = st.slider(
            "Number of clusters",
            min_value=2,
            max_value=max_clusters,
            value=min(5, max_clusters),
            key="cluster_count",
        )
        if st.button("Build movie clusters", type="primary"):
            st.session_state["cluster_request"] = cluster_count
        if "cluster_request" in st.session_state:
            try:
                clustered, summary = cluster_catalog(
                    movies, int(st.session_state["cluster_request"])
                )
                left, right = st.columns([1, 1])
                with left:
                    st.subheader("Cluster sizes")
                    st.bar_chart(summary["movie_count"], color="#75a7ff", height=280)
                with right:
                    st.subheader("Cluster profiles")
                    st.dataframe(summary, width="stretch")
                st.subheader("Catalog map")
                st.scatter_chart(
                    clustered,
                    x="map_x",
                    y="map_y",
                    color="cluster",
                    height=460,
                )
                st.subheader("Representative titles")
                representatives = (
                    clustered.sort_values("vote_average", ascending=False)
                    .groupby("cluster", group_keys=False)
                    .head(5)
                    .sort_values(["cluster", "vote_average"], ascending=[True, False])
                )
                display = _movie_table(representatives)
                display.insert(
                    0,
                    "cluster",
                    representatives["cluster"].astype(str).to_numpy(),
                )
                st.dataframe(display, width="stretch", hide_index=True)
            except Exception as exc:
                st.error(f"Could not calculate clusters ({type(exc).__name__}).")
                st.caption(str(exc))

    with recommend_tab:
        st.subheader("Find your next film")
        st.markdown(
            '<p class="section-note">Recommendations use TF-IDF similarity over movie '
            'overviews. Pick a seed movie or describe the story and mood you want.</p>',
            unsafe_allow_html=True,
        )
        try:
            vectorizer, matrix, search_movies = load_recommender()
            st.caption(f"Recommendation index: {len(search_movies):,} movies")
            mode = st.radio(
                "Recommendation style",
                ["Similar to a movie", "Match a plot or mood"],
                horizontal=True,
            )
            seed_index: int | None = None
            query = ""
            if mode == "Similar to a movie":
                options = list(range(len(search_movies)))
                seed_index = st.selectbox(
                    "Choose a seed movie",
                    options,
                    format_func=lambda index: (
                        f"{search_movies.iloc[index]['title']} "
                        f"({int(search_movies.iloc[index]['release_year']) if pd.notna(search_movies.iloc[index]['release_year']) else 'year unknown'})"
                    ),
                    key="recommend_seed",
                )
                query = str(search_movies.iloc[seed_index]["title"])
            else:
                query = st.text_area(
                    "What would you like to watch?",
                    placeholder="A tense mystery in a small town, with a clever twist…",
                    height=100,
                )
            count = st.slider("How many recommendations?", 3, 15, 8, key="recommend_count")
            if st.button("Recommend movies", type="primary"):
                if mode == "Match a plot or mood" and len(query.strip()) < 3:
                    st.warning("Enter a little more about the kind of film you want.")
                else:
                    results = recommend(
                        vectorizer, matrix, search_movies, query, count, seed_index
                    )
                    if results.empty:
                        st.warning(
                            "No close matches found. Try a different movie or add more "
                            "specific plot keywords."
                        )
                    else:
                        for rank, (_, movie) in enumerate(results.iterrows(), start=1):
                            title_col, score_col = st.columns([5, 1])
                            with title_col:
                                year = (
                                    int(movie["release_year"])
                                    if pd.notna(movie["release_year"])
                                    else "Year unknown"
                                )
                                st.markdown(f"**{rank:02d} · {movie['title']}** · {year}")
                                st.caption(
                                    f"{_format_items(movie['genres'])} · "
                                    f"Rating {movie['vote_average']:.1f}/10"
                                    if pd.notna(movie["vote_average"])
                                    else _format_items(movie["genres"])
                                )
                                st.write(movie["overview"] or "No plot summary available.")
                            with score_col:
                                st.metric("Match", f"{movie['similarity']:.0%}")
                            st.divider()
        except Exception as exc:
            st.error(f"Recommendation artifacts could not be loaded ({type(exc).__name__}).")
            st.caption(
                "Expected files: `models/tfidf_vectorizer.pkl`, "
                "`models/movie_tfidf_matrix.npz`, and `models/movies_for_search.pkl`."
            )

    st.markdown("---")
    st.caption("CineMind · Movie discovery powered by catalog analytics and machine learning")


if __name__ == "__main__":
    main()
