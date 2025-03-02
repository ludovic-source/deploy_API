import streamlit as st
import mlflow.pyfunc
import numpy as np
import pandas as pd

# Charger les données clients issus du préprocessing
df = pd.read_csv('C:/Users/admin/Documents/Projets/Projet_7/data_projet/cleaned/data.csv')

# Charger le modèle MLflow (local ou depuis un serveur)
MODEL_URI = "MLFlow_model_for_API"
model = mlflow.pyfunc.load_model(MODEL_URI)

# Interface Streamlit
st.title("Prédiction pour les clients 🚀")

# Afficher une liste des 100 premiers clients
client_options = X['client_id'].head(100).tolist()

# Sélectionner un client parmi la liste
selected_client = st.selectbox("Choisissez un client", client_options)

# Récupérer les caractéristiques du client sélectionné
client_data = X[X['client_id'] == selected_client].drop(columns='client_id')

# Afficher les données du client
st.write("Données du client sélectionné :")
st.write(client_data)

if st.button("Faire la prédiction"):
    # Faire la prédiction avec les données du client
    features = client_data.values.reshape(1, -1)  # Reshaper pour correspondre aux attentes du modèle
    prediction = model.predict(features)

    # Afficher la prédiction
    st.success(f"Prédiction pour {selected_client} : {prediction[0]}")