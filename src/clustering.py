from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from feature_eng import load_data, load_features
from collections import defaultdict
import joblib


data = load_data()
df = load_features(data)

X = df.drop(columns=['vote_count', 'high_engagement'])
Y = df['high_engagement']
X['genres'] = X['genres'].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))
X['keywords'] = X['keywords'].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))

num_cols = X.select_dtypes(include="number").columns.tolist()
num_cols = [c for c in num_cols if c != 'movie_id']

num_pipeline = Pipeline(steps=[
  ('imputer', SimpleImputer(strategy='median')),
  ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer([
  ("num", num_pipeline, num_cols),
  ("overview", TfidfVectorizer(
    stop_words="english",
    max_features=1000,
    ngram_range=(2, 3)
  ), "overview")
])



features = preprocessor.fit_transform(X)

k_values = range(2, 12)
silhouette_scores = []

for k in k_values:
  candidate_model = KMeans(n_clusters=k, random_state=42, n_init="auto")
  candidate_clusters = candidate_model.fit_predict(features)
  silhouette_scores.append(silhouette_score(features, candidate_clusters))

best_k = k_values[np.argmax(silhouette_scores)]

plt.figure(figsize=(8, 4))
plt.plot(list(k_values), silhouette_scores, marker="o")
plt.xticks(list(k_values))
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette score")
plt.title("Silhouette score by number of clusters")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Best k: {best_k} (silhouette score: {max(silhouette_scores):.4f})")

final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
clusters = final_kmeans.fit_predict(features)
df['clusters'] = clusters

print("Final silhouette score:", silhouette_score(features, clusters))
print(df[["title", "genres", "clusters"]].head(40))


cluster_sizes = df.groupby("clusters").size().rename("movie_count")
cluster_percentages = (
    df["clusters"].value_counts(normalize=True).mul(100).rename("percentage")
)

cluster_summary = pd.concat([cluster_sizes, cluster_percentages], axis=1)
print("\n--- Cluster sizes ---")
print(cluster_summary.sort_index())


profile_columns = [
    "budget",
    "revenue",
    "popularity",
    "runtime",
    "release_year",
    "vote_average",
    "num_genres",
    "num_Keywords",
]

cluster_averages = df.groupby("clusters")[profile_columns].mean().round(2)

print("\n--- Cluster averages ---")
print(cluster_averages)


genres_by_cluster = (
    df[["clusters", "genres"]]
    .explode("genres")
    .dropna()
    .groupby(["clusters", "genres"])
    .size()
    .reset_index(name="movie_count")
)

top_genres = (
    genres_by_cluster
    .sort_values(["clusters", "movie_count"], ascending=[True, False])
    .groupby("clusters")
    .head(5)
)

print("\n--- Top 5 genres per cluster ---")
print(top_genres)

keywords_by_cluster = (
    df[["clusters", "keywords"]]
    .explode("keywords")
    .dropna()
    .groupby(["clusters", "keywords"])
    .size()
    .reset_index(name="movie_count")
)

top_keywords = (
    keywords_by_cluster
    .sort_values(["clusters", "movie_count"], ascending=[True, False])
    .groupby("clusters")
    .head(5)
)

print("\n--- Top 5 keywords per cluster ---")
print(top_keywords)

distances = final_kmeans.transform(features)

df["distance_to_centroid"] = distances[
    np.arange(len(df)),
    df["clusters"].to_numpy()
]

representative_movies = (
    df.sort_values(["clusters", "distance_to_centroid"])
    .groupby("clusters")
    .head(5)[["clusters", "title", "genres", "distance_to_centroid"]]
)

print("\n--- Five representative movies per cluster ---")
print(representative_movies)

catalog_average = df[profile_columns].mean()

difference_from_catalog = (
    cluster_averages - catalog_average
).round(2)

percentage_difference = (
    (cluster_averages / catalog_average - 1) * 100
).round(1)

print("\n--- Difference from catalog average ---")
print(difference_from_catalog)

print("\n--- Percentage difference from catalog average ---")
print(percentage_difference)


# Reduce the full feature matrix to two PCA-like components
pca = TruncatedSVD(n_components=2, random_state=42)
features_2d = pca.fit_transform(features)

# Reduce K-Means centroids into the same two-dimensional space
centers_2d = pca.transform(final_kmeans.cluster_centers_)

explained_variance = pca.explained_variance_ratio_ * 100

# Create a plotting DataFrame
plot_df = pd.DataFrame({
    "PC1": features_2d[:, 0],
    "PC2": features_2d[:, 1],
    "cluster": clusters
})

plt.figure(figsize=(12, 8))

sns.scatterplot(
    data=plot_df,
    x="PC1",
    y="PC2",
    hue="cluster",
    palette="tab10",
    alpha=0.65,
    s=35
)

# Show each cluster center
plt.scatter(
    centers_2d[:, 0],
    centers_2d[:, 1],
    c="black",
    marker="X",
    s=250,
    edgecolors="white",
    linewidths=1.5,
    label="Cluster centers"
)

# Add a readable label at each center
for cluster_id, (x, y) in enumerate(centers_2d):
    plt.annotate(
        f"Cluster {cluster_id}",
        (x, y),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold"
    )

plt.xlabel(f"Principal Component 1 ({explained_variance[0]:.1f}% variance)")
plt.ylabel(f"Principal Component 2 ({explained_variance[1]:.1f}% variance)")
plt.title("Movie Clusters — Two-Dimensional PCA Projection")
plt.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()