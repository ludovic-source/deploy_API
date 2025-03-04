import streamlit as st
import mlflow.pyfunc
import numpy as np
import pandas as pd

# Charger les données clients issus du préprocessing
df = pd.read_csv('data/data_100.csv')

# Séparer la cible
X = df.drop(columns=["TARGET"])  # Features

# Charger le modèle MLflow
MODEL_URI = "mlflow_model_for_API"
model = mlflow.pyfunc.load_model(MODEL_URI)

# Définir un seuil optimal (exemple : 0.5, à ajuster selon ton besoin)
SEUIL_OPTIMAL = 0.45  # À ajuster selon ton analyse ROC

# Récupérer la liste des colonnes attendues par le modèle
expected_columns = model.metadata.get_input_schema().input_names()

# Interface Streamlit
st.title("Prédiction pour les clients 🚀")

# Afficher une liste des 100 premiers clients
client_options = [f"Client {i}" for i in range(1, len(X) + 1)]

# Sélectionner un client parmi la liste
selected_client_index = st.selectbox("Choisissez un client", client_options)

# Récupérer l'index du client choisi
client_index = int(selected_client_index.split()[-1]) - 1
client_data = X.iloc[client_index]

# Vérifier que les colonnes correspondent
client_data = client_data.reindex(expected_columns, fill_value=0)  # Compléter colonnes manquantes

# Transformer en DataFrame avec colonnes
client_data_df = pd.DataFrame([client_data], columns=expected_columns)

# Afficher les données du client
st.write("Données du client sélectionné :")
st.write(client_data_df)

if st.button("Faire la prédiction"):
    # Obtenir la probabilité avec le modèle
    proba = model.predict(client_data_df)  # Assurez-vous que le modèle renvoie bien des probabilités
    probability = proba[0][1]  # Probabilité de la classe positive (ajuster selon format du modèle)

    # Déterminer la prédiction en fonction du seuil
    prediction = 1 if probability >= SEUIL_OPTIMAL else 0

    # Afficher la probabilité et la décision finale
    st.write(f"**Seuil optimal utilisé :** {SEUIL_OPTIMAL}")
    st.write(f"**Probabilité prédite :** {probability:.4f}")
    st.success(f"**Prédiction pour {selected_client_index} : {'Risque élevé' if prediction == 1 else 'Risque faible'}**")