import pandas as pd
import numpy as np
import json
from datatest import validate
from connection_db import mycol
import pymongo


# Load the JSON file manually, then extract the movies list
with open('data/raw/all_movies.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)


# Extract just the movies list
data = pd.DataFrame(raw_data['movies'])

# print(f"Loaded {len(data)} movies")
# print(data.columns)
# print(data.dtypes)


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
    
    if 'keywords' in df.columns:
        df['genres'] = df['genres'].apply(
            lambda x: [g['name'] for g in x] if isinstance(x, list) else []
        )
        
        df['keywords'] = df['keywords'].apply(
            lambda x: [k['name'] for k in x.get('keywords', [])] if isinstance(x, dict) else []
        )
        
        df = df.drop_duplicates(subset=['movie_id'])
        
        df['genres'] = df['genres'].apply(lambda x: ', '.join(x) if x else '')
        df['keywords'] = df['keywords'].apply(lambda x: ', '.join(x) if x else '')
        df = df.drop_duplicates()
        df = df.reset_index(drop=True)
        
    return df


def date_time(df):
    df['release_date'] = pd.to_datetime(df['release_date'])
    df['release_year'] = df['release_date'].dt.year
    df['release_month'] = df['release_date'].dt.month
    return df

def handle_missing_vals(df):
    if df.isnull().sum().any():
        df['runtime'] = df['runtime'].fillna(df['runtime'].median())
        
def standard_num_vals(df):
    numeric_cols = df.select_dtypes(include=np.number)
    for col in numeric_cols.columns:
            pd.to_numeric(df[col])
    return numeric_cols

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
    text_cols = df.select_dtypes(include='str').columns.tolist()
    list_cols = df.select_dtypes(include='object').columns.tolist()
    cols_groups = {
        'num_cols' : numeric_cols,
        'text_cols' : text_cols,
        'list_cols': list_cols
    }
    return cols_groups

def get_df(df):
    return df

health_check(data)
cleaned = clean_data(data)
date_time(cleaned)
print(f"\nAfter cleaning: {cleaned.shape[0]} rows, {cleaned.shape[1]} cols")
print(cleaned.head())
print(standard_num_vals(cleaned))
print(cols_groupping(cleaned))
print(handle_incons(cleaned))


mycol.delete_many(filter={})

records = cleaned.to_dict(orient='records')
x = mycol.insert_many(records)
print(f"Inserted {len(records)} documents into MongoDB")


