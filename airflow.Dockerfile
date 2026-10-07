FROM apache/airflow:3.3.1

RUN pip install --no-cache-dir \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.3.1/constraints-3.13.txt" \
    "matplotlib>=3.7.0" \
    "numpy>=1.24.0" \
    "pandas>=2.0.0" \
    "pymongo>=4.0.0" \
    "python-dotenv>=1.0.0" \
    "requests>=2.28.0" \
    "scikit-learn>=1.3.0" \
    "scipy>=1.10.0" \
    "seaborn>=0.12.0" \
    "joblib>=1.2.0"
