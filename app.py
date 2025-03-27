import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import mlflow.pyfunc
from lime.lime_tabular import LimeTabularExplainer

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
model = mlflow.pyfunc.load_model(MODEL_URI)

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
SEUIL_OPTIMAL = 0.40
if st.button("Faire la prédiction", help="Cliquez pour obtenir la prédiction de crédit"):
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
                value=score * 100,
                title={"text": "Score de Crédit"},
                delta={"reference": SEUIL_OPTIMAL * 100},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": gauge_color},
                    "steps": [
                        {"range": [0, score * 100], "color": "#4CAF50" if score < SEUIL_OPTIMAL else "#FF5252"},
                        {"range": [score * 100, 100], "color": "#FFC107"}
                    ],
                    "threshold": {"line": {"color": "black", "width": 4}, "thickness": 1, "value": SEUIL_OPTIMAL * 100}
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

# Initialiser LIME Explainer
explainer = LimeTabularExplainer(
    X.values,               # Données d'entraînement sans la cible
    feature_names=X.columns.tolist(),
    class_names=["Accordé", "Refusé"],  # Nom des classes (si classification)
    mode="classification"
)

# Expliquer la prédiction du client sélectionné
exp = explainer.explain_instance(client_data.values, model.predict_proba)

# Afficher l'explication avec Streamlit
st.subheader("Importance Locale avec LIME")
fig_lime = exp.as_pyplot_figure()
st.pyplot(fig_lime)