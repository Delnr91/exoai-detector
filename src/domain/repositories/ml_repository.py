from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class MLRepository(ABC):
    """
    PORT (Interfaz): Contrato para todos los adaptadores de modelos ML.
    Permite a la Capa de Aplicación solicitar una predicción sin saber 
    si se usa Scikit-learn, TensorFlow o un Ensemble.
    """
    @abstractmethod
    def load_model(self) -> None:
        """Carga el modelo persistido en memoria."""
        pass

    @abstractmethod
    def predict(self, features: List[float]) -> Dict[str, Any]:
        """Realiza una predicción en base a las features de entrada."""
        pass

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Obtiene la lista de características esperadas (57 en v3)."""
        pass

    @abstractmethod
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Obtiene la importancia de características para la interpretabilidad."""
        pass