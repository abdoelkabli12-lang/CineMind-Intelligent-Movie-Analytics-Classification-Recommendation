# Standard library
from pathlib import Path
from typing import Any

from scipy.sparse import save_npz, load_npz

# Data manipulation
import numpy as np
import pandas as pd

# Machine learning and similarity
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

import sys


from nlp import clean_text


user_query = input("enter a keyword: \n")

cleaned_query = clean_text([user_query])[0]

vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
movie_matrix = load_npz("models/movie_tfidf_matrix.npz")
movies = joblib.load("models/movies_for_search.pkl")

query_vector = vectorizer.transform([cleaned_query])

if query_vector.nnz == 0:
  print("No usable matching words in the vectorizer vocabulary")
  sys.exit()

similarities = cosine_similarity(query_vector, movie_matrix)

if similarities.mean() == 0:
  print("Try different plot keywords.")
  sys.exit()

flattened_similarities = similarities.ravel()

ranked_indices = np.argsort(flattened_similarities)[::-1]

top_indices = [
    index for index in ranked_indices
    if flattened_similarities[index] > 0
][:10]

top_movies = movies.iloc[top_indices].copy()
top_movies["similarity_score"] = flattened_similarities[top_indices]

print(top_movies[["title", "genres", "release_year"]])


