import streamlit as st
import mlflow.pyfunc
import numpy as np
import pandas as pd

# Charger les données clients issus du préprocessing
df = pd.read_csv('data/data_100.csv')

# Séparer la cible
X = df.drop(columns=["TARGET"])  # Features

# Charger le modèle MLflow (local ou depuis un serveur)
MODEL_URI = "MLFlow_model_for_API"
model = mlflow.pyfunc.load_model(MODEL_URI)

# Interface Streamlit
st.title("Prédiction pour les clients 🚀")

# Afficher une liste des 100 premiers clients
client_options = [f"Client {i}" for i in range(1, len(X) + 1)]

# Sélectionner un client parmi la liste
selected_client_index = st.selectbox("Choisissez un client", client_options)

# Récupérer les caractéristiques du client sélectionné
# Si vous avez une liste d'options sans ID spécifique, vous pouvez utiliser l'index de la liste
client_index = int(selected_client_index.split()[-1]) - 1  # Récupérer l'index du client choisi
client_data = X.iloc[client_index]

# Afficher les données du client
st.write("Données du client sélectionné :")
st.write(client_data)

if st.button("Faire la prédiction"):
    # Faire la prédiction avec les données du client
    features = client_data.values.reshape(1, -1)  # Reshaper pour correspondre aux attentes du modèle
    prediction = model.predict(features)

    # Afficher la prédiction
    st.success(f"Prédiction pour {selected_client_index} : {prediction[0]}")