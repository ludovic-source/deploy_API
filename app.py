import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go

# Charger les données clients
@st.cache_data
def load_data():
    return pd.read_csv('data/data_200.csv')

df = load_data()
X = df.drop(columns=["TARGET"])  # Features

# URL de l'API
API_URL = "https://api-flask-0j4d.onrender.com/predict"  # Remplacez par votre URL
API_FEATURE_IMPORTANCE_URL = "https://api-flask-0j4d.onrender.com/feature_importance"

# Interface Streamlit
st.title("Prédiction de Score de Crédit 🚀")
st.sidebar.header("Sélection du Client")

# Liste des clients
client_options = [f"Client {i}" for i in range(1, len(X) + 1)]
selected_client_index = st.sidebar.selectbox("Choisissez un client", client_options)
client_index = int(selected_client_index.split()[-1]) - 1  
client_data = X.iloc[client_index]

st.subheader("Données du client sélectionné")
st.write(client_data)

# Faire la prédiction
SEUIL_OPTIMAL = 0.40
if st.button("Faire la prédiction"):
    try:
        response = requests.post(API_URL, json={"features": client_data.to_dict()})
        if response.status_code == 200:
            result = response.json()
            score = result.get("probability", 0)
            prediction = "Crédit refusé" if score >= SEUIL_OPTIMAL else "Crédit accordé"
            
            # Jauge de score avec seuil
            gauge_color = "#FF0000" if score >= SEUIL_OPTIMAL else "#008000"  # Rouge si refusé, vert si accepté
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=score * 100,
                title={"text": "Score de Crédit"},
                delta={"reference": SEUIL_OPTIMAL * 100},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": gauge_color},
                    "steps": [
                        {"range": [0, score * 100], "color": "#008000" if score < SEUIL_OPTIMAL else "#FF0000"},  # Vert ou rouge jusqu'au score
                        {"range": [score * 100, 100], "color": "#FFD700"}  # puis jaune
                        #{"range": [SEUIL_OPTIMAL * 100, 100], "color": "#FFD700" if score < SEUIL_OPTIMAL else "#ff9999"}  # Jaune ou rouge selon dépassement du seuil
                    ],
                    "threshold": {"line": {"color": "black", "width": 5}, "thickness": 1, "value": SEUIL_OPTIMAL * 100}
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

# Comparaison des features du client avec la distribution générale
st.sidebar.subheader("Analyse des caractéristiques")
selected_feature = st.sidebar.selectbox("Sélectionnez une feature", X.columns)
fig_dist = px.histogram(X, x=selected_feature, marginal="box", title=f"Distribution de {selected_feature}")
fig_dist.add_vline(x=client_data[selected_feature], line_dash="dash", line_color="red")
st.plotly_chart(fig_dist)

# Analyse bi-variée
st.sidebar.subheader("Analyse Bi-variée")
x_feature = st.sidebar.selectbox("Feature en X", X.columns, index=0)
y_feature = st.sidebar.selectbox("Feature en Y", X.columns, index=1)
fig_scatter = px.scatter(X, x=x_feature, y=y_feature, title=f"Relation entre {x_feature} et {y_feature}")
fig_scatter.add_trace(go.Scatter(x=[client_data[x_feature]], y=[client_data[y_feature]], mode='markers', marker=dict(color='red', size=10)))
st.plotly_chart(fig_scatter)

# Importance des features
if st.sidebar.button("Afficher les contributions des features"):
    response = requests.post(API_FEATURE_IMPORTANCE_URL, json={"features": client_data.to_dict()})
    if response.status_code == 200:
        feature_importance = pd.DataFrame(response.json()).sort_values("importance", ascending=False)
        fig_importance = px.bar(feature_importance, x="importance", y="feature", orientation='h', title="Importance des features")
        st.plotly_chart(fig_importance)
    else:
        st.error("Erreur API Feature Importance.")
