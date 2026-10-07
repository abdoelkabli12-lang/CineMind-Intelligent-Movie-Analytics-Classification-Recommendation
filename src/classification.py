import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV

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

from sklearn.feature_extraction.text import TfidfVectorizer

from feature_eng import load_data, load_features
from collections import defaultdict
import joblib
from sklearn.base import clone
from pathlib import Path

data = load_data()
df = load_features(data)

X = df.drop(columns=['vote_count', 'high_engagement'])
Y = df['high_engagement']
X['genres'] = X['genres'].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))
X['keywords'] = X['keywords'].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))   
X["overview"] = X["overview"].fillna("")


num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include='object').columns.tolist()

num_cols = [c for c in num_cols if c != 'movie_id']
cat_cols = [c for c in cat_cols if c not in ('title')]


x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, shuffle=True, random_state=42, stratify=Y)



num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
])

cat_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols),
    ("overview_text", TfidfVectorizer(stop_words="english", max_features=3000, ngram_range=(1, 2)), "overview"),
])

lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(
        C=1,
        penalty='l2',
        solver='liblinear',
        max_iter=500,
        class_weight='balanced'
    ))
])


# GRIDSEARCHCV FOR LOGISTIC REGRESSION MODEL!!!!!!!!!

logistic_param_grid = {
    "classifier__C": [0.1, 1, 10],
    "classifier__solver": ["liblinear"],
    "classifier__class_weight": [None, "balanced"],
    "preprocessor__overview_text__max_features": [2000, 3000],
    "preprocessor__overview_text__ngram_range": [(1, 1), (1, 2)],
}

gs_lr = GridSearchCV(
    estimator=lr_pipeline,
    param_grid=logistic_param_grid,
    scoring='roc_auc',
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=-1,
    refit=True,
    verbose=1,
)

gs_lr.fit(x_train, y_train)

print("Best parameters:", gs_lr.best_params_)
print("Best cross-validation ROC-AUC:", gs_lr.best_score_)

best_lr_pipeline = gs_lr.best_estimator_


y_pred = best_lr_pipeline.predict(x_test)
y_scores = best_lr_pipeline.predict_proba(x_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_scores))

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


models_cv = models
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc',
}
full_models = defaultdict(list)
strKFold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, pipe in models_cv.items():
    cv_scores = cross_validate(pipe, x_train, y_train, cv=strKFold, scoring=scoring)

    print(f"=== {name} ===")
    full_models['Model'].append(name)
    for metric in scoring:
        mean_score = cv_scores[f'test_{metric}'].mean()
        full_models[f'Mean CV {metric.replace("_", " ").title()}'].append(mean_score)
        print(f"Mean CV {metric.replace('_', ' ').title()}: {mean_score:.4f}")

comparison = pd.DataFrame(full_models)
comparison = comparison.set_index('Model')

print('\n')

print('=== Model Comparison — 5-Fold Cross-Validation ===')
print(comparison.to_string())

best_model_name = comparison['Mean CV Roc Auc'].idxmax()
best_precision = comparison.loc[best_model_name, 'Mean CV Precision']
best_accuracy = comparison.loc[best_model_name, 'Mean CV Accuracy']
best_f1 = comparison.loc[best_model_name, 'Mean CV F1']

print(f'\n=== Best Model: {best_model_name} ===')
print(f'Mean CV Precision: {best_precision:.4f}')
print(f'Mean CV Accuracy: {best_accuracy:.4f}')
print(f'Mean CV F1:   {best_f1:.4f}')

best_model = clone(models[best_model_name])
best_model.fit(X, Y)

model_bundle = {
    "model": best_model,
    "model_name": best_model_name,
    "numeric_features": num_cols,
    "categorical_features": cat_cols,
    "target": "high_engagement",
}

model_path = Path(__file__).resolve().parents[1] / 'models' / 'airflow' / 'best_model.pkl'
model_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model_bundle, model_path)
