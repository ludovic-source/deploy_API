import streamlit as st
import pandas as pd
import requests

# Charger les données clients issus du préprocessing
df = pd.read_csv('data/data_200.csv')

# Séparer la cible
X = df.drop(columns=["TARGET"])  # Features

# URL de l'API hébergée sur Render
API_URL = "https://api-flask-0j4d.onrender.com/predict"  # Remplacez par l'URL réelle de votre API

# Interface Streamlit
st.title("Prédiction pour les clients 🚀")

# Afficher une liste des 100 premiers clients
client_options = [f"Client {i}" for i in range(1, len(X) + 1)]

# Sélectionner un client parmi la liste
selected_client_index = st.selectbox("Choisissez un client", client_options)

# Récupérer les caractéristiques du client sélectionné
client_index = int(selected_client_index.split()[-1]) - 1  # Récupérer l'index du client choisi
client_data = X.iloc[client_index].to_dict()

# Afficher les données du client
st.write("Données du client sélectionné :")
st.write(client_data)

# Définir le seuil optimal - calculé lors de l'entraînement du modèle optimisé
SEUIL_OPTIMAL = 0.19

if st.button("Faire la prédiction"):
    try:
        # Envoyer les données du client à l'API
        response = requests.post(API_URL, json={"features": client_data})
        
        if response.status_code == 200:
            result = response.json()
            positive_class_prob = result.get("probability", 0)  # Probabilité retournée par l'API
            
            # Afficher la probabilité
            st.write(f"Probabilité pour la classe positive (classe 1) : {round(positive_class_prob, 2)}")
            
            # Déterminer la prédiction avec le seuil optimal
            prediction = 1 if positive_class_prob >= SEUIL_OPTIMAL else 0
            
            # Afficher la prédiction et le message l'accord ou non
            message = "1 - Crédit refusé" if prediction == 1 else "0 - Crédit accordé"
            st.success(f"Prédiction pour {selected_client_index} : {message} (Seuil optimal: {SEUIL_OPTIMAL})")
        else:
            st.error("Erreur lors de la communication avec l'API. Vérifiez l'URL et les données envoyées.")
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de requête : {e}")
