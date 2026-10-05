import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from feature_eng import load_data, load_features

data = load_data()
df = load_features(data)

X = df.drop(columns=['vote_count', 'high_engagement'])
Y = df['high_engagement']
X['genres'] = X['genres'].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))
X['keywords'] = X['keywords'].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))   

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include='object').columns.tolist()

num_cols = [c for c in num_cols if c != 'movie_id']
cat_cols = [c for c in cat_cols if c not in ('title', 'overview')]


x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, shuffle=True, random_state=42, stratify=Y)



num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
])

cat_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

print(f"num_cols: {num_cols}")
print(f"cat_cols: {cat_cols}")
print(f"X shape: {X.shape}")

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(
        C=1,
        penalty='l1',
        solver='liblinear',
        max_iter=500,
        class_weight='balanced'
    ))
])

rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=600,
        max_depth=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ))
])

svm_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LinearSVC(
        C=1,
        class_weight='balanced',
        max_iter=500,
        random_state=42
    ))
])

models = {
    'LogisticRegression': lr_pipeline,
    'RandomForest': rf_pipeline,
    'LinearSVC': svm_pipeline,
}

ACCURACY = []
PERCISION = []
RECALL = []
F1 = []
ROC_AUC = []

for name, pipe in models.items():
    pipe.fit(x_train, y_train)
    y_pred = pipe.predict(x_test)
    
    print(f"\n=== {name} ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):4f}")
    print(f"Precision: {precision_score(y_test, y_pred):4f}")
    print(f"Recall: {recall_score(y_test, y_pred):4f}")
    print(f"F1: {f1_score(y_test, y_pred):4f}")
    scores = (
        pipe.predict_proba(x_test)[:, 1]
        if hasattr(pipe, 'predict_proba')
        else pipe.decision_function(x_test)
    )
    print(f"ROC-AUC {roc_auc_score(y_test, scores):4f}")
    print(f"confusion matrix: \n{confusion_matrix(y_test, y_pred)}")
    ACCURACY.append(accuracy_score(y_test, y_pred))
    PERCISION.append(precision_score(y_test, y_pred))
    RECALL.append(recall_score(y_test, y_pred))
    F1.append(f1_score(y_test, y_pred))
    ROC_AUC.append(roc_auc_score(y_test, scores))


print(f"Accuracy: {ACCURACY}")
print(f"Percision: {PERCISION}")
print(f"RECALL: {RECALL}")
print(f"F1: {F1}")
print(f"Roc_Auc: {ROC_AUC}")



