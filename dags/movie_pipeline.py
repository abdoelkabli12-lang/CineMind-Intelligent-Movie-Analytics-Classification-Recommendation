# dags/movie_pipeline.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import runpy
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

default_args = {
    'owner': 'mel',
    'depends_on_past': False,
    'email_on_failure': True,
    'email': ['mel@example.com'],
    'retries': 2,
    'retry_delay': timedelta(minutes=2),
}

dag = DAG(
    'movie_intelligence_pipeline',
    default_args=default_args,
    description='Complete Movie Intelligence pipeline: extraction → ML → recommendation',
    schedule='@daily',
    start_date=datetime(2026, 9, 28),
    catchup=False,
    tags=['movie', 'tmdb', 'machine-learning', 'film'],

)
# Task 1: TMDB Extraction
def extract_task():
    from src.extraction import extract_films
    extract_films(
        min_id=1,
        max_id=5000,
        output_file='/opt/airflow/data/raw/all_movies.json',
    )
    print('Extraction completed')

task_extract = PythonOperator(
    task_id='extract_tmdb',
    python_callable=extract_task,
    dag=dag,
)

# Task 2: Cleaning
def clean_task():
    from src.cleaning import clean_file
    df = clean_file(
        '/opt/airflow/data/raw/all_movies.json',
        '/opt/airflow/data/processed/movies_clean.csv',
    )
    print(f'Cleaning completed: {len(df)} movies')
    df.to_pickle('/opt/airflow/data/processed/movies_clean.pkl')

task_clean = PythonOperator(
    task_id='clean_data',
    python_callable=clean_task,
    dag=dag,
)

# Task 3: Feature Engineering
def feature_eng_task():
    from src.feature_eng import load_features
    df = pd.read_pickle('/opt/airflow/data/processed/movies_clean.pkl')
    df = load_features(df)
    df.to_pickle('/opt/airflow/data/processed/movies_with_features.pkl')
    print(f'Feature engineering completed: {len(df)} movies, {len(df.columns)} columns')

task_feature_eng = PythonOperator(
    task_id='feature_engineering',
    python_callable=feature_eng_task,
    dag=dag,
)

# Task 4: MongoDB
def mongo_task():
    from connection_db import mycol
    df = pd.read_pickle('/opt/airflow/data/processed/movies_with_features.pkl')
    records = df.to_dict(orient='records')
    if not records:
        raise ValueError('No featured movie records available to store in MongoDB')
    mycol.delete_many({})
    mycol.insert_many(records)
    print(f'MongoDB: {mycol.count_documents({})} movies stored')

task_mongo = PythonOperator(
    task_id='store_mongodb',
    python_callable=mongo_task,
    dag=dag,
)

# Task 5: ML Classification
def ml_classification_task():
    runpy.run_module('src.classification', run_name='__main__')
    print('Classification completed. Model: /opt/airflow/models/airflow/best_model.pkl')

task_ml_class = PythonOperator(
    task_id='ml_classification',
    python_callable=ml_classification_task,
    dag=dag,
)

# Task 6: ML Clustering
def ml_clustering_task():
    runpy.run_module('src.clustering', run_name='__main__')
    print('Clustering completed')

task_ml_cluster = PythonOperator(
    task_id='ml_clustering',
    python_callable=ml_clustering_task,
    dag=dag,
)

# Define dependencies
task_extract >> task_clean >> task_feature_eng >> task_mongo
task_mongo >> task_ml_class
task_mongo >> task_ml_cluster