import pandas as pd
import numpy as np
import json
from connection_db import mycol


with open('data/raw/all_movies.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

data = pd.DataFrame(raw_data['movies'])


def health_check(df):
    print("=== Health Check ===")
    print(f"Shape: {df.shape}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nDuplicate titles: {df['title'].duplicated().sum()}")

    expected = {'movie_id', 'title', 'overview', 'release_date', 'runtime',
                'original_language', 'genres', 'keywords', 'budget', 'revenue',
                'popularity', 'vote_average', 'vote_count'}
    missing = expected - set(df.columns)
    if missing:
        print(f"Missing columns: {missing}")
    else:
        print("All expected columns present.")
    return df


def clean_data(df):
    if 'genres' in df.columns:
        df['genres'] = df['genres'].apply(
            lambda x: [g['name'] for g in x] if isinstance(x, list) else []
        )
    if 'keywords' in df.columns:
        df['keywords'] = df['keywords'].apply(
            lambda x: [k['name'] for k in x.get('keywords', [])] if isinstance(x, dict) else []
        )
    df = df.drop_duplicates(subset=['movie_id'])
    df = df.reset_index(drop=True)
    return df


def date_time(df):
    df['release_date'] = pd.to_datetime(df['release_date'])
    return df


def handle_missing_vals(df):
    if df.isnull().sum().any():
        df['runtime'] = df['runtime'].fillna(df['runtime'].median())
        df['budget'] = df['budget'].fillna(df['budget'].median())
        df['revenue'] = df['revenue'].fillna(df['revenue'].median())
    return df


def standard_num_vals(df):
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def handle_incons(df):
    bad_runtime = (df['runtime'] <= 0) | (df['runtime'] > 300)
    if bad_runtime.any():
        print(f"  Dropping {bad_runtime.sum()} movies with invalid runtime")
        df = df[~bad_runtime]

    df['vote_average'] = df['vote_average'].clip(0, 10)
    df['budget'] = df['budget'].clip(lower=0)
    df['revenue'] = df['revenue'].clip(lower=0)
    return df


def cols_groupping(df):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    text_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    list_cols = df.select_dtypes(include='object').columns.tolist()
    cols_groups = {
        'num_cols': numeric_cols,
        'text_cols': text_cols,
        'list_cols': list_cols
    }
    return cols_groups


# --- Pipeline ---
health_check(data)
cleaned = clean_data(data)
cleaned = date_time(cleaned)
cleaned = handle_missing_vals(cleaned)
cleaned = standard_num_vals(cleaned)
cleaned = handle_incons(cleaned)

print(f"\nAfter cleaning: {cleaned.shape[0]} rows, {cleaned.shape[1]} cols")
print(cleaned.head())
print(cols_groupping(cleaned))

# Write to MongoDB
mycol.delete_many(filter={})
records = cleaned.to_dict(orient='records')
mycol.insert_many(records)
print(f"Inserted {len(records)} documents into MongoDB")   