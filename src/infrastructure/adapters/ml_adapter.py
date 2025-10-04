import joblib
import json
import os
from typing import List, Dict, Any, Optional
from src.infrastructure.monitoring.logger import logger 
from src.domain.repositories.ml_repository import MLRepository 


class RandomForestAdapter(MLRepository):

    """
    ADAPTADOR: Implementación del Repositorio ML (PORT) para el Ensemble Híbrido.
    Carga el modelo final y lo usa para predicción.
    """

    MODEL_FILE_PATH = './models/ensemble_v3_final.pkl'
    EXPECTED_FEATURES_COUNT = 32
    METRICS_FILE_PATH = './models/latest_metrics.json'
    IMPORTANCE_FILE_PATH = './models/feature_importance.json'

    def __init__(self):
        self.model = self.load_model()
        self.feature_names = self._load_feature_names()
        
    def load_model(self):
        """Implementa la carga del modelo binario (.pkl) y extrae el objeto model."""

        logger.info(f"Iniciando carga del modelo desde: {self.MODEL_FILE_PATH}")
        if not os.path.exists(self.MODEL_FILE_PATH):
            logger.error(f"Archivo no encontrado: {self.MODEL_FILE_PATH}")
            raise FileNotFoundError(...)
        try:

            # joblib.load devuelve el diccionario guardado en _save_model.
            # Debemos extraer el objeto 'model' de ese diccionario.

            model_data = joblib.load(self.MODEL_FILE_PATH)
            
            # Extraer el objeto VotingClassifier
            model = model_data.get('model') 
            if model is None:
                raise RuntimeError("El archivo .pkl no contiene el objeto 'model' esperado.")
                
            logger.info(" Modelo Ensemble V3 cargado exitosamente.")
            return model
        except Exception as e:
            logger.critical(f"Error CRÍTICO al deserializar el modelo: {e}")
            raise RuntimeError(f"Error al cargar el modelo: {e}")

    def _load_json_data(self, path: str) -> Optional[Dict[str, Any]]:

        """Función auxiliar para cargar datos JSON (métricas/importancia)."""
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error al cargar archivo JSON {path}: {e}")
                return None
        return None

    def _load_feature_names(self) -> List[str]:
        """Carga los nombres de las características desde el modelo guardado."""

        return [f"feature_{i}" for i in range(self.EXPECTED_FEATURES_COUNT)]

    #   Métodos del PORT (MLRepository) implementados
    def predict(self, features: List[float]) -> Dict[str, Any]:

        """Realiza la predicción y retorna un resultado estandarizado."""

        if len(features) != self.EXPECTED_FEATURES_COUNT:
            logger.warning(f"Feature Mismatch: Esperado={self.EXPECTED_FEATURES_COUNT}, Recibido={len(features)}")
            raise ValueError(f"Se esperaban {self.EXPECTED_FEATURES_COUNT} features, pero se recibieron {len(features)}. Ajuste la entrada o el pipeline.")
            
        prediction = self.model.predict([features])[0]
        confidence = self.model.predict_proba([features])[0][1] 
        
        logger.info(f"Predicción generada: Clase={int(prediction)}, Confianza={confidence:.4f}")
        
        return {
            "prediction": int(prediction),
            "confidence": float(confidence),
            "model_name": "Ensemble_v3_Final",
        }

    def get_feature_names(self) -> List[str]:

        """Implementación para devolver los nombres de features."""
        return self.feature_names
        
    def get_feature_importance(self) -> Optional[Dict[str, float]]:

        """Devuelve la importancia de características para interpretabilidad."""
        return self._load_json_data(self.IMPORTANCE_FILE_PATH)

    def get_metrics(self) -> Optional[Dict[str, float]]:

        """Devuelve las métricas de evaluación finales."""
        return self._load_json_data(self.METRICS_FILE_PATH)