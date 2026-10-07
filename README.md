# CineMind-Intelligent-Movie-Analytics-Classification-Recommendation
🎬 CineMind — An end-to-end movie intelligence platform using TMDB data, Machine Learning, NLP, TF-IDF, classification, clustering, and content-based recommendation, with automated pipelines via Airflow and a Streamlit dashboard.

## Streamlit dashboard

The dashboard includes catalog analytics and visualizations, movie-level audience-engagement predictions, on-demand clustering, and TF-IDF content recommendations.

Start the dashboard and MongoDB from the repository root:

```powershell
docker compose --env-file .env up -d --build mongo app
```

Open [http://localhost:8502](http://localhost:8502). To run Streamlit directly in a local Python environment instead:

```powershell
streamlit run dashboard/app.py
```

The app reads movies from the `cleaned_data` MongoDB collection and falls back to the processed or raw files in `data/`. Classification requires a saved `models/best_model.pkl` artifact. Content recommendations require `models/tfidf_vectorizer.pkl`, `models/movie_tfidf_matrix.npz`, and `models/movies_for_search.pkl`. Clusters are calculated on demand from the current catalog.
