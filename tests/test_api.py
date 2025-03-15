import unittest
import json
import pandas as pd
from app_api import app

class APITestCase(unittest.TestCase):
    
    def setUp(self):
        """Initialiser le client de test et charger les données"""
        self.client = app.test_client()
        self.client.testing = True

        # Charger les données factices
        self.df = pd.read_csv('data/data_100.csv')
        self.X = self.df.drop(columns=["TARGET"])  # Features
        self.X = self.X.apply(pd.to_numeric, errors='coerce')  # Convertir toutes les colonnes en numériques

    def test_home(self):
        """Tester la route d'accueil"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn("API de prédiction", response.data.decode("utf-8"))

    def test_predict_valid_input(self):
        """Tester la prédiction avec des données valides"""
        sample_data = {"features": self.X.iloc[0].to_dict()}  # ✅ Correction ici
        
        response = self.client.post('/predict', json=sample_data)
        self.assertEqual(response.status_code, 200)

        response_data = json.loads(response.data)
        self.assertIn("probability", response_data)
        self.assertIsInstance(response_data["probability"], float)

    def test_predict_invalid_input(self):
        """Tester la prédiction avec des données manquantes"""
        sample_data = {"features": {key: None for key in range(394)}}  # 394 features avec des valeurs nulles
        response = self.client.post('/predict', json=sample_data)
        self.assertEqual(response.status_code, 500)

    def test_predict_wrong_method(self):
        """Tester l'utilisation d'une mauvaise méthode HTTP"""
        response = self.client.get('/predict')  # La route /predict attend un POST
        self.assertEqual(response.status_code, 405)  # 405 = Méthode non autorisée

if __name__ == '__main__':
    unittest.main()
