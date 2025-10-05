import pandas as pd
import numpy as np
import logging
import os
import joblib
import json

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from src.domain.services.exoplanet_pipeline import ExoplanetPreprocessor as DataPreprocessor

# Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# Ubicaciones para guardar los artefactos
MODELS_DIR = './models'
MODEL_PATH = os.path.join(MODELS_DIR, 'ensemble_v3_final.pkl')
METRICS_PATH = os.path.join(MODELS_DIR, 'latest_metrics.json')
IMPORTANCE_PATH = os.path.join(MODELS_DIR, 'feature_importance.json')
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, 'feature_names.json')


class TrainModelUseCase:
    """
    Caso de Uso: Entrenamiento de un Ensemble Híbrido con Validación Temporal.
    Guarda el modelo, métricas, importancia de features y nombres de features.
    """
    def __init__(self):
        self.model = None
        self.metrics = {}
        self.feature_names = []
        os.makedirs(MODELS_DIR, exist_ok=True)

    def train_and_evaluate(self, X: np.ndarray, y: np.ndarray, temporal_splits, feature_names: list):
        logger.info("--- Iniciando Entrenamiento de ENSEMBLE HÍBRIDO ---")
        self.feature_names = feature_names

        train_index, test_index = temporal_splits[-1]
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        logger.info(f"Datos divididos con VALIDACIÓN TEMPORAL (Split 5/5): Train={len(X_train)}, Test={len(X_test)}")
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        lr = LogisticRegression(random_state=42, class_weight='balanced')

        ensemble = VotingClassifier(estimators=[('rf', rf), ('lr', lr)], voting='soft')
        
        logger.info("Entrenando ENSEMBLE (RF + LR)...")
        ensemble.fit(X_train, y_train)
        self.model = ensemble
        
        y_pred = ensemble.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        self.metrics = {
            "model_name": "Ensemble_v3_Final",
            "accuracy": round(accuracy, 4),
            "f1_score": round(f1, 4),
            "train_size": len(X_train),
            "test_size": len(X_test),
        }
        
        logger.info(f"ENTRENAMIENTO FINALIZADO. Métricas ENSEMBLE: Accuracy={self.metrics['accuracy']:.4f}, F1-Score={self.metrics['f1_score']:.4f}")
        
        self._save_artifacts()
        
        return self.metrics

    def _save_artifacts(self):
        """Guarda el modelo DENTRO de un diccionario, junto con otros metadatos."""
        if self.model is not None:
            # --- INICIO DE LA CORRECCIÓN CLAVE ---
            # Ahora guardamos un diccionario, no solo el modelo.
            model_data_to_save = {
                'model': self.model,
                'feature_names': self.feature_names
            }
            joblib.dump(model_data_to_save, MODEL_PATH)
            # --- FIN DE LA CORRECCIÓN CLAVE ---
            
            logger.info(f"💾 Modelo y metadatos guardados exitosamente en: {MODEL_PATH}")

            with open(METRICS_PATH, 'w') as f:
                json.dump(self.metrics, f, indent=4)
            logger.info(f"📄 Métricas guardadas en: {METRICS_PATH}")

            rf_model = self.model.named_estimators_['rf']
            importances = rf_model.feature_importances_
            importance_dict = dict(zip(self.feature_names, importances.tolist()))
            with open(IMPORTANCE_PATH, 'w') as f:
                json.dump(importance_dict, f, indent=4)
            logger.info(f"📊 Importancia de características guardada en: {IMPORTANCE_PATH}")
            
            # Guardamos los nombres de features en su propio JSON para el frontend
            with open(FEATURE_NAMES_PATH, 'w') as f:
                json.dump({"feature_names": self.feature_names}, f, indent=4)
            logger.info(f"📝 Nombres de características guardados en: {FEATURE_NAMES_PATH}")
            
        else:
            logger.warning("No hay modelo entrenado para guardar.")


# --- Ejecución del Caso de Uso ---
if __name__ == "__main__":
    logger.info("Iniciando caso de uso de entrenamiento desde __main__...")
    
    preprocessor = DataPreprocessor(data_path='./data/kepler_koi.csv')
    X_final_scaled, y_balanced, temporal_splits = preprocessor.fit_transform_complete()
    
    feature_names_from_pipeline = preprocessor.feature_names
    
    trainer = TrainModelUseCase()
    final_metrics = trainer.train_and_evaluate(X_final_scaled, y_balanced, temporal_splits, feature_names_from_pipeline)

    print("\n--- Resultado del Caso de Uso de Entrenamiento Híbrido Final ---")
    print(pd.Series(final_metrics))