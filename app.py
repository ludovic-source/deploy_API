import streamlit as st
import mlflow.pyfunc
import numpy as np
import pandas as pd

# Charger les données clients issus du préprocessing
df = pd.read_csv('data/data_100.csv')

# Séparer la cible
X = df.drop(columns=["TARGET"])  # Features

# Charger le modèle MLflow (local ou depuis un serveur)
MODEL_URI = "mlruns/0/96b5c0e6d7204d7b8f070ad0723bb774/artifacts/mlflow_model_for_API_scoring"
model = mlflow.lightgbm.load_model(MODEL_URI)

# Interface Streamlit
st.title("Prédiction pour les clients 🚀")

# Afficher une liste des 100 premiers clients
client_options = [f"Client {i}" for i in range(1, len(X) + 1)]

# Sélectionner un client parmi la liste
selected_client_index = st.selectbox("Choisissez un client", client_options)

# Récupérer les caractéristiques du client sélectionné
client_index = int(selected_client_index.split()[-1]) - 1  # Récupérer l'index du client choisi
client_data = X.iloc[client_index]

# Afficher les données du client
st.write("Données du client sélectionné :")
st.write(client_data)

# Définir le seuil optimal - calculé lors de l'entraînement du modèle optimisé
SEUIL_OPTIMAL = 0.19

if st.button("Faire la prédiction"):
    # Reshaper pour correspondre aux attentes du modèle
    features = client_data.values.reshape(1, -1)  # Reshaper les données pour le modèle

    # Faire la prédiction avec `predict_proba`
    try:
        # Utiliser predict_proba pour obtenir les probabilités
        proba = model.predict_proba(features)
        
        # Récupérer la probabilité pour la classe positive (classe 1, ajuster si nécessaire)
        positive_class_prob = proba[0][1]  # Probabilité de la classe positive
        
        # Afficher la probabilité
        st.write(f"Probabilité pour la classe positive (classe 1) : {positive_class_prob}")

        # Utiliser un seuil pour déterminer la prédiction
        prediction = 1 if positive_class_prob >= SEUIL_OPTIMAL else 0
        
        # Afficher la prédiction et le message l'accord ou non
        if prediction == 1:
            message = "Crédit refusé"
        else:
            message = "Crédit accordé"
            
        st.success(f"Prédiction pour {selected_client_index} : {message} (Seuil optimal: {SEUIL_OPTIMAL})")
        
    except AttributeError:
        st.error("Le modèle ne supporte pas la méthode `predict_proba`. Assurez-vous qu'il s'agit d'un modèle de classification.")