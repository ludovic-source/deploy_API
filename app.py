import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import mlflow.pyfunc
import shap
import numpy as np
import matplotlib.pyplot as plt

# Charger les données clients
@st.cache_data
def load_data():
    return pd.read_csv('data/data_200.csv')

df = load_data()
X = df.drop(columns=["TARGET"])  # Features
y = df["TARGET"]  # Cible (TARGET)

# URL de l'API
API_URL = "https://api-flask-0j4d.onrender.com/predict"
API_FEATURE_IMPORTANCE_URL = "https://api-flask-0j4d.onrender.com/feature_importance"

# Charger le modèle
MODEL_URI = "mlruns/0/6210849d7ad04b08bf569f1b084101e1/artifacts/mlflow_model_for_API_scoring"
# Charger le pipeline complet
pipeline = mlflow.sklearn.load_model(MODEL_URI)

# Extraire le modèle final (LightGBM) depuis le pipeline
model = pipeline.named_steps['classifier'] 

# Interface Streamlit
st.title("Prédiction de Score de Crédit 🚀")
st.sidebar.header("Sélection du Client")

# Liste des clients
client_options = [f"Client {i}" for i in range(1, len(X) + 1)]
selected_client_index = st.sidebar.selectbox("Choisissez un client", client_options, help="Utilisez la flèche bas/haut pour naviguer")
client_index = int(selected_client_index.split()[-1]) - 1  
client_data = X.iloc[client_index]

st.subheader("Données du client sélectionné")
st.write(client_data)

# Faire la prédiction
SEUIL_OPTIMAL = 0.30

try:
    response = requests.post(API_URL, json={"features": client_data.to_dict()})
    if response.status_code == 200:
        result = response.json()
        score = result.get("probability", 0)
        prediction = "Crédit refusé" if score >= SEUIL_OPTIMAL else "Crédit accordé"
            
        # Jauge de score
        gauge_color = "#D32F2F" if score >= SEUIL_OPTIMAL else "#388E3C"  # Amélioration du contraste
        fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=score,
                title={"text": "Score de Crédit"},
                delta={"reference": SEUIL_OPTIMAL},
                gauge={
                    "axis": {"range": [0, 1]},
                    "bar": {"color": gauge_color},
                    "steps": [
                        {"range": [0, score], "color": "#4CAF50" if score < SEUIL_OPTIMAL else "#FF5252"},
                        {"range": [score, 1], "color": "#FFC107"}
                    ],
                    "threshold": {"line": {"color": "black", "width": 4}, "thickness": 1, "value": SEUIL_OPTIMAL}
                }
        ))
        st.plotly_chart(fig_gauge)
            
        if score >= SEUIL_OPTIMAL:
            st.error(f"Prédiction: {prediction} (Score: {round(score, 2)})", icon="🚨")
        else:
            st.success(f"Prédiction: {prediction} (Score: {round(score, 2)})", icon="✅")
    else:
        st.error("Erreur API.")
except requests.exceptions.RequestException as e:
        st.error(f"Erreur de requête : {e}")

# Comparaison des features du client
st.sidebar.subheader("Analyse des caractéristiques")
selected_feature = st.sidebar.selectbox("Sélectionnez une feature", X.columns, help="Choisissez une feature à analyser")
fig_dist = px.histogram(X, x=selected_feature, marginal="box", title=f"Distribution de {selected_feature}")
fig_dist.add_vline(x=client_data[selected_feature], line_dash="dash", line_color="black")
st.plotly_chart(fig_dist)

# Analyse bi-variée
st.sidebar.subheader("Analyse Bi-variée")
x_feature = st.sidebar.selectbox("Feature en X", X.columns, index=0)
y_feature = st.sidebar.selectbox("Feature en Y", X.columns, index=1)

fig_scatter = px.scatter(X, x=x_feature, y=y_feature, title=f"Relation entre {x_feature} et {y_feature}")

# Ajouter un carré noir pour représenter le client sélectionné
fig_scatter.add_trace(go.Scatter(
    x=[client_data[x_feature]],
    y=[client_data[y_feature]],
    mode='markers',
    marker=dict(color='black', size=12, symbol="square")  # Carré noir
))

st.plotly_chart(fig_scatter)

# Feature importance globale et locale

# Importance des features via SHAP en local
explainer = shap.Explainer(model)
shap_values = explainer(X)

# Importance globale des features
st.subheader("Importance Globale des Features (SHAP)")
feature_importance = np.abs(shap_values.values).mean(axis=0)
sorted_indices = np.argsort(feature_importance)[::-1]
top_features = X.columns[sorted_indices][:10]
top_importance = feature_importance[sorted_indices][:10]

fig_shap_global = px.bar(
    x=top_importance[::-1],  # Inversion des valeurs
    y=top_features[::-1],  # Inversion des labels
    orientation="h", 
    labels={"x": "Importance SHAP", "y": "Feature"},
    title="Top 10 des Features les Plus Importantes"
)
st.plotly_chart(fig_shap_global)

# Importance locale pour le client sélectionné
st.subheader("Importance Locale des Features (Client Sélectionné)")

# Générer un graphique de force pour le client sélectionné
st.write("Les valeurs SHAP positives poussent la prédiction vers le refus de crédit, tandis que les valeurs négatives favorisent l'acceptation.")

# Transformer client_data en DataFrame (1, n_features)
client_data_df = client_data.to_frame().T  # .T transpose la Series pour créer un DataFrame de forme (1, n_features)

# Convertir les colonnes qui sont sous forme de chaîne (par exemple, `object`) en valeurs numériques
client_data_df = client_data_df.apply(pd.to_numeric, errors='coerce')  # 'coerce' remplacera les erreurs par NaN

# Obtenir les valeurs SHAP pour le client sélectionné
shap_local_values = explainer.shap_values(client_data_df)

# Ajouter le graphique SHAP de l'importance locale
fig = plt.figure(figsize=(8, 6))
shap.bar_plot(shap_local_values[0], feature_names=X.columns, max_display=20)
st.pyplot(fig)