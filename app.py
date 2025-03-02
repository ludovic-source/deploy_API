import streamlit as st
import mlflow.pyfunc
import numpy as np

# Charger le modèle MLflow (local ou depuis un serveur)
MODEL_URI = "MLFlow_model_for_API"
model = mlflow.pyfunc.load_model(MODEL_URI)

# Interface Streamlit
st.title("Déploiement MLflow avec Render 🚀")

# Entrée utilisateur
input_data = st.text_input("Entrez les caractéristiques (ex: 5.1,3.5,1.4,0.2)")

if st.button("Prédire"):
    try:
        features = np.array([list(map(float, input_data.split(",")))])
        prediction = model.predict(features)
        st.success(f"Prédiction : {prediction[0]}")
    except:
        st.error("Erreur de format ! Assurez-vous d'entrer des nombres valides.")