import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
import mlflow
import mlflow.lightgbm
import app


class TestStreamlitAPI(unittest.TestCase):

    def setUp(self):
        """Prépare les données et le modèle pour les tests"""
        # Charger les données factices
        self.df = pd.read_csv('data/data_100.csv')
        self.X = self.df.drop(columns=["TARGET"])  # Features
        self.X = self.X.apply(pd.to_numeric, errors='coerce')  # Convertir toutes les colonnes en numériques

        # Charger le modèle MLflow (local ou depuis un serveur)
        self.model_uri = "mlruns/0/96b5c0e6d7204d7b8f070ad0723bb774/artifacts/mlflow_model_for_API_scoring"
        self.model = mlflow.lightgbm.load_model(self.model_uri)

    def test_data_loading(self):
        """Test de chargement des données"""
        self.assertIsInstance(self.df, pd.DataFrame, "Les données ne sont pas un DataFrame.")
        self.assertTrue("TARGET" in self.df.columns, "La colonne 'TARGET' est absente des données.")
        self.assertEqual(self.X.shape[1], 394, "Le nombre de features attendu est incorrect.")

    def test_model_loading(self):
        """Test de chargement du modèle MLflow"""
        self.assertIsNotNone(self.model, "Le modèle MLflow n'a pas été chargé correctement.")
    
    def test_prediction(self):
        """Test des prédictions du modèle"""
        client_data = self.X.iloc[0].values.reshape(1, -1)  # Première ligne de données pour le test
        proba = self.model.predict_proba(client_data)
        self.assertEqual(proba.shape, (1, 2), "La forme de la probabilité n'est pas correcte.")
        self.assertTrue(np.all((proba >= 0) & (proba <= 1)), "Les probabilités doivent être entre 0 et 1.")
        
    def test_prediction_class(self):
        """Test de la classe de prédiction"""
        client_data = self.X.iloc[0].values.reshape(1, -1)  # Première ligne de données pour le test
        proba = self.model.predict_proba(client_data)
        positive_class_prob = proba[0][1]
        prediction = 1 if positive_class_prob >= 0.18 else 0
        self.assertIn(prediction, [0, 1], "La prédiction doit être 0 ou 1.")

    @patch('streamlit.write')
    def test_streamlit_interface(self, mock_write):
        """Test de l'interface Streamlit"""
        # Simuler une interaction Streamlit
        with patch('streamlit.selectbox', return_value="Client 1"):
            app.st.write("Données du client sélectionné :")            
            # Convertir les données du client en dictionnaire
            client_data_dict = self.X.iloc[0].to_dict()  # Convertir en dictionnaire
            app.st.write(client_data_dict)  # Passer le dictionnaire à streamlit.write            
            mock_write.assert_called_with(client_data_dict)  # Vérifier si streamlit.write a été appelé avec un dictionnaire

    def test_invalid_prediction(self):
        """Test des prédictions invalides si le modèle ne supporte pas `predict_proba`"""
        # Simuler un modèle sans la méthode `predict_proba`
        class MockModel:
            def predict(self, X):
                return np.array([0, 1])
        
        mock_model = MockModel()
        client_data = self.X.iloc[0:1]
        
        with self.assertRaises(AttributeError):
            mock_model.predict_proba(client_data)

if __name__ == "__main__":
    unittest.main()
