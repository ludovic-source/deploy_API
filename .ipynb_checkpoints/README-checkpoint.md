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

## Endpoints de l'API

### 1. Prédiction du scoring

#### Requête : choix du client dans la liste déroulante de 200 clients

Envoi de le requète POST avec les données du client pour la prédiction

#### Réponse : prédiction

- probabilité d'appartenir à la classe 1 (prêt refusé)
- seuil optimal
- décision : prêt accordé (0) ou refusé (1)

## Déploiement et CI/CD

- **GitHub Actions** est utilisé pour l'exécution automatique des tests unitaires lors du build à chaque "push" sur la branche Master
- **Tests unitaires** avec Unittest exécutés automatiquement lors des builds.
- **MLFlow** utilisé pour le suivi des expérimentations et le déploiement du modèle.
- **Streamlit** utilisé pour l'interface
- **Render** utilisé pour le déploiement sur le Cloud à chaque push sur la branche master

## Installation et Utilisation

### Installation

- Ouvrir le dashboard Render en cliquant sur le lien : https://dashboard.render.com/web/srv-cv3alm23esus73deojqg
- Choisir le type de déploiement souhaité ("dernier commit", ...)
- Ouvrir l'onglet "settings" du dashboard Render pour modifier les paramètres, notamment la commande de run de l'application, ou le lien et la branche du repository Github où se trouve les sources

### Lancer l'API

- Lancer l'API en cliquant sur ce lien : https://deploy-api-scoring.onrender.com
- Si l'API n'est pas disponible, aller sur le dashboard et choisir "restart service' en haut à droite pour relancer le service.

### Exécuter les tests

- Automatique lors des pushs sur la branche Master (workflow créé dans gitHub
- Configuration du workflow dans le fichier uni_tests.yml
- Commande pour lancer les tests en local : python -m unittest discover tests/
- Les tests se trouvent dans le dossier /tests