import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from connection_db import mycol
from math  import log1p
pd.set_option('display.max_columns', None)

def load_data():
  data_raw = mycol.find({})
  df = pd.DataFrame(data_raw)
  if '_id' in df.columns:
    df = df.drop(columns='_id')
  print(f"Loaded {len(df)} movies from MongoDB")
  return df


def date_features(df):
    df['release_year'] = df['release_date'].dt.year
    df['release_month'] = df['release_date'].dt.month
    df['release_decade'] = (df['release_year'] // 10) * 10
    df['decade_str'] = df['release_decade'].astype(str) + 's'
    return df

def num_features(df):
  while True:
    df['num_genres'] = df['genres'].apply(len)
    df['num_Keywords'] = df['keywords'].apply(len)
    return df

def cat_features(df):
  labels_rt = ['low', 'medium', 'long', 'very_long']
  bins_rt = [60, 90, 120, 180, 200]
  df['duration_cat'] = pd.cut(df['runtime'], bins=bins_rt, labels=labels_rt, include_lowest=True)
  
  labels_bu = ['low', 'medium', 'high', 'very high']
  bins_bu = [10000, 10000000, 40000000, 80000000, 400000000]
  df['budget_cat'] = pd.cut(df['budget'], bins=bins_bu, labels=labels_bu, include_lowest=True)
  return df

def log_feature(df):
  df['log_popularity'] = np.log1p(df['popularity'])
  return df

def save_features(df):
  date_features(df)
  num_features(df)
  cat_features(df)
  log_feature(df)
  
  return df
df = load_data()
save_features(df)
print (df.columns)