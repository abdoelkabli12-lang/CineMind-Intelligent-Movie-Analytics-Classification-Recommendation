import re
import string
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.sparse import save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

from feature_eng import load_data, load_features

pd.set_option('display.max_columns', None)


def clean_text(values):
    res = []

    for i in values:
        if not isinstance(i, str):
            res.append('')
            continue

        i = i.lower()
        i = i.translate(str.maketrans('', '', string.punctuation))
        i = re.sub(r' {2,}', ' ', i)
        res.append(i)

    return res


def tf_idf(df):
    vectorizer = TfidfVectorizer(max_df=0.9, ngram_range=(1,2), stop_words='english', max_features=2500)
    tfidf_matrix = vectorizer.fit_transform(df)
    return vectorizer, tfidf_matrix

def avg_weights(vector,mat):
    avg_weights = np.array(mat.mean(axis=0)).ravel()
    feature_names = vector.get_feature_names_out()
    result = pd.DataFrame({'term': feature_names, 'avg_tfidf': avg_weights})
    result = result.sort_values('avg_tfidf', ascending=False)
    return result




if __name__ == "__main__":
    data = load_data()
    df = load_features(data)
    df_clean = pd.DataFrame({
    'title': clean_text(df['title']),
    'overview': clean_text(df['overview'])
})
    
    valid_overviews = df_clean["overview"].str.strip().ne("")
    movies_for_search = df.loc[valid_overviews].copy()
    overviews_for_search = df_clean.loc[valid_overviews, "overview"].tolist()

    print(f"Excluded empty overviews: {(~valid_overviews).sum()}")

    vectorizer, tfidf_matrix = tf_idf(overviews_for_search)

    print(avg_weights(vectorizer, tfidf_matrix))

    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
    save_npz("models/movie_tfidf_matrix.npz", tfidf_matrix)
    joblib.dump(movies_for_search, "models/movies_for_search.pkl")