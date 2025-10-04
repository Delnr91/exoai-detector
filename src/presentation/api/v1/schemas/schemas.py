# src/presentation/api/v1/schemas/schemas.py

from pydantic import BaseModel
from typing import List, Optional

# --- Modelos de Predicción ---

class PredictRequest(BaseModel):
    """
    Schema para la entrada de datos a la API de predicción.
    Debe coincidir con las 6 features esperadas por el modelo.
    """
    features: List[float] # koi_period, koi_impact, koi_duration, koi_depth, koi_prad, koi_model_snr

class PredictResponse(BaseModel):
    """
    Schema para la salida de la API de predicción.
    """
    prediction: int       # 0 o 1
    confidence: float     # Probabilidad de ser exoplaneta
    model_name: str       # 'RandomForest_v1'
    is_potentially_habitable: bool  # Nuevo campo para indicar habitabilidad


# --- Modelos de Métricas y Entrenamiento ---

class MetricsResponse(BaseModel):
    """
    Schema para reportar las métricas de rendimiento del modelo.
    """
    accuracy: float
    f1_score: float
    train_size: int
    test_size: int
    
    # Podemos agregar más métricas si son pre-calculadas
    precision: Optional[float] = None
    recall: Optional[float] = None


class FeatureImportanceResponse(BaseModel):
    """
    Schema para reportar la importancia de características.
    """
    feature: str
    importance: float
    
    class Config:
        # Permite retornar una lista de objetos FeatureImportanceResponse
        from_attributes = True