import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

# 1. Charger les données
df = pd.read_csv("Financial_inclusion_dataset.csv")

# 2. Définir X et y
X = df.drop("bank_account", axis=1)
y = df["bank_account"].map({"Yes": 1, "No": 0})  # convertir en 0/1

# 3. Identifier colonnes
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

# 4. Préprocesseur
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("numerical", StandardScaler(), numeric_cols),
    ]
)

# 5. Pipeline avec SMOTE + XGBoost
pipeline = ImbPipeline(steps=[
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("classifier", XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        use_label_encoder=False,
        random_state=42
    ))
])

# 6. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 7. Grille d’hyperparamètres
param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [3, 5, 7],
    "classifier__learning_rate": [0.01, 0.1, 0.2],
    "classifier__subsample": [0.8, 1.0],
    "classifier__colsample_bytree": [0.8, 1.0]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    scoring="roc_auc",
    cv=3,
    verbose=2,
    n_jobs=-1
)

# 8. Entraînement
grid.fit(X_train, y_train)

# 9. Évaluation
y_pred = grid.predict(X_test)
y_proba = grid.predict_proba(X_test)[:, 1]

print(" Best parameters:", grid.best_params_)
print("Accuracy :", accuracy_score(y_test, y_pred))
print("ROC AUC :", roc_auc_score(y_test, y_proba))
print("\nClassification report :\n", classification_report(y_test, y_pred))

# 10. Sauvegarde modèle
joblib.dump(grid.best_estimator_, "optimized_model_xgb.joblib")
print(" Optimized XGBoost model saved as optimized_model_xgb.joblib")
