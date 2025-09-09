# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
import json

# =============================
# Charger modèle + metadata
# =============================
model = joblib.load("model_pipeline.joblib")

with open("feature_metadata.json", "r") as f:
    metadata = json.load(f)

categorical_cols = metadata["categorical_cols"]
numeric_cols = metadata["numeric_cols"]

st.title("💳 Prédiction d'inclusion financière en Afrique")
st.write("Remplissez les informations ci-dessous pour prédire si une personne a un compte bancaire.")

# =============================
# Formulaire utilisateur
# =============================
user_input = {}

for col in numeric_cols:
    user_input[col] = st.number_input(f"{col}", value=0.0)

for col in categorical_cols:
    user_input[col] = st.text_input(f"{col}", value="")

# Transformer en DataFrame
input_df = pd.DataFrame([user_input])

# =============================
# Prédiction
# =============================
if st.button("Prédire"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f" Cette personne est probablement **bancarisée** (probabilité {prob:.2f})")
    else:
        st.error(f" Cette personne est probablement **non bancarisée** (probabilité {prob:.2f})")
