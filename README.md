# API de Scoring de Crédit

## Introduction

Une entreprise souhaite mettre en œuvre un outil de "scoring crédit" pour calculer la probabilité qu’un client rembourse son crédit, puis classifier la demande en crédit accordé ou refusé. L'objectif est de développer un algorithme de classification en s’appuyant sur des sources de données variées (données comportementales, données provenant d'autres institutions financières, etc.).

## Mission

- Construire un modèle de scoring qui prédit automatiquement la probabilité de faillite d'un client.
- Analyser les features qui contribuent le plus au modèle, tant au niveau global (feature importance globale) que local (feature importance locale).
- Mettre en production le modèle via une API et réaliser une interface de test.
- Mettre en œuvre une approche MLOps complète pour le suivi des expérimentations et l’analyse en production du data drift.
- Utiliser MLFlow pour le suivi des expérimentations, le stockage centralisé des modèles et le déploiement via un "model registry".
- Assurer une gestion du code avec Git et une intégration continue via Github Actions.
- Automatiser les tests avec Pytest (ou Unittest) et les exécuter lors du processus de build via Github Actions.

## Fonctionnalités de l'API

- **Prédiction de scoring** : Retourne la probabilité de défaut de paiement pour un client donné.
- **Classification** : Détermine si un crédit est accordé ou refusé.

## Sources dans github

- Repository : https://github.com/ludovic-source/deploy_API
- Le dossier /data contient les données clients utilisées pour l'API
- Le dossier /tests contient les tests unitaires
- Le dossier /mlruns/0/.../artifacts contient le pipeline du modèle
- Le fichier requirements.txt contient les dépendances nécessaires au bon fonctionnement de l'API (utilisé par Render)

## 🚀 Déploiement sur Render
### 1️⃣ Prérequis
- Python 3.8+
- `pip install flask mlflow pandas lightgbm gunicorn`

### 2️⃣ Installation locale
```bash
# Cloner le projet
git clone https://github.com/ludovic-source/api_flask
cd votre-repo

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'API
python app.py
```
L'API tournera sur `http://127.0.0.1:5000/`

### 3️⃣ Déploiement sur Render
1. Poussez votre code sur GitHub
2. Allez sur [Render](https://render.com/)
3. Créez un **nouveau service web**
4. Liez votre repo GitHub
5. Dans "Build Command", ajoutez :
   ```bash
   pip install -r requirements.txt
   ```
6. Dans "Start Command", ajoutez :
   ```bash
   gunicorn app:app --bind 0.0.0.0:$PORT
   ```
7. Déployez 🚀

## 📡 Utilisation de l'API
### 1️⃣ Tester l'API en local
#### Vérifier que l'API fonctionne :
```bash
curl http://127.0.0.1:5000/
```
#### Faire une prédiction :
```bash
curl -X POST "http://127.0.0.1:5000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": {"feature1": 0.5, "feature2": 1.2, "feature3": -0.7}}'
```

### 2️⃣ API en production
Une fois déployée sur Render, utilisez l'URL fournie :
```bash
curl -X POST "https://api-flask-0j4d.onrender.com/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": {"feature1": 0.5, "feature2": 1.2, "feature3": -0.7}}'
```

## 🛠️ Technologies utilisées
- Flask
- MLflow
- LightGBM
- Render (déploiement)


