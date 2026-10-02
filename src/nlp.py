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

data = load_data()
df = load_features(data)


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


df_clean = pd.DataFrame({
    'overview': clean_text(df['overview']),
    'title': clean_text(df['title'])
})
def tf_idf(df):
    vectorizer = TfidfVectorizer(max_df=1, ngram_range=(1,2), stop_words='english', vocabulary=500)
    tfidf_matrix = vectorizer.fit_transform(df)
    return tfidf_matrix

mat = tf_idf(df_clean['overview'].tolist())
print(mat)

