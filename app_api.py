from flask import Flask, request, jsonify
import mlflow.pyfunc
import pandas as pd
import os

app = Flask(__name__)

# Charger le modèle MLflow (assurez-vous que le chemin est correct)
MODEL_URI = "mlruns/0/bbeb96096dd54e7ea0c743e89343ea24/artifacts/mlflow_model_for_API_scoring"
model = mlflow.lightgbm.load_model(MODEL_URI)

@app.route('/')
def home():
    return "API de prédiction est en ligne !"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer les données envoyées en JSON
        data = request.get_json()
        if "features" not in data:
            return jsonify({"error": "Données incorrectes. Clé 'features' manquante."}), 400
        
        # Convertir les données en DataFrame pandas
        features = pd.DataFrame([data["features"]])
        
        # Faire la prédiction
        proba = model.predict_proba(features)[0][1]  # Probabilité classe 1
        
        return jsonify({"probability": proba})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Pour Render
    app.run(host='0.0.0.0', port=port)
