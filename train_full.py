# train_full.py
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from xgboost import XGBClassifier

# =============================
# 1. Charger les données
# =============================
df = pd.read_csv("Financial_inclusion_dataset.csv")

print("Aperçu des données :")
print(df.head())
print("\nColonnes :", df.columns)

# =============================
# 2. Préparation des données
# =============================
target_col = "bank_account"
if target_col not in df.columns:
    raise ValueError(f"La colonne cible '{target_col}' n'existe pas dans le dataset !")

X = df.drop(columns=[target_col])
y = df[target_col].map({"Yes": 1, "No": 0})  # encode la cible

# Séparer num et cat
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

print("\nColonnes catégorielles :", categorical_cols)
print("Colonnes numériques :", numeric_cols)

# =============================
# 3. Pipeline de prétraitement
# =============================
categorical_transformer = OneHotEncoder(handle_unknown="ignore")
numeric_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

# =============================
# 4. Modèle XGBoost
# =============================
xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline(steps=[("preprocessor", preprocessor),
                           ("classifier", xgb)])

# =============================
# 5. Split
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =============================
# 6. Optimisation Hyperparamètres
# =============================
param_dist = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [3, 5, 7, 9],
    "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "classifier__subsample": [0.6, 0.8, 1.0],
    "classifier__colsample_bytree": [0.6, 0.8, 1.0]
}

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=20,
    scoring="roc_auc",
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

print("\n Optimisation des hyperparamètres...")
search.fit(X_train, y_train)

print("\n Meilleurs paramètres trouvés :", search.best_params_)

# =============================
# 7. Évaluation finale
# =============================
best_model = search.best_estimator_

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print("\nAccuracy :", accuracy_score(y_test, y_pred))
print("ROC AUC :", roc_auc_score(y_test, y_prob))
print("\nClassification report :\n", classification_report(y_test, y_pred))

# =============================
# 8. Sauvegarde du modèle
# =============================
joblib.dump(best_model, "optimized_model_xgb.joblib")

# Sauvegarder les infos sur les colonnes
feature_metadata = {
    "categorical_cols": categorical_cols,
    "numeric_cols": numeric_cols,
    "target_col": target_col,
    "best_params": search.best_params_
}
with open("feature_metadata.json", "w") as f:
    json.dump(feature_metadata, f)

print("\n Modèle XGBoost optimisé entraîné et sauvegardé : optimized_model_xgb.joblib")
