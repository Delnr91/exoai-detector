from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# --- Modelos de Predicción ---

class PredictRequest(BaseModel):
    """
    Schema para la entrada de datos a la API de predicción.
    Acepta un diccionario de features para ser más robusto y explícito.
    """
    features: Dict[str, float] = Field(..., example={
        "koi_period": 85.5, "koi_impact": 0.146, "koi_duration": 4.5, 
        "koi_depth": 874.8, "koi_prad": 2.26, "koi_model_snr": 25.8
    })

class PredictResponse(BaseModel):
    """
    Schema para la salida de la API de predicción, enriquecido para el pitch.
    """
    prediction_label: str = Field(..., example="Exoplaneta Confirmado")
    confidence_score: float = Field(..., example=0.95)
    prediction_value: int = Field(..., example=1)
    model_version: str = Field("Ensemble_v3_Final", example="Ensemble_v3_Final")
    is_potentially_habitable: bool = Field(False, example=False)
    
# --- Modelos de Métricas y Explicabilidad ---

class MetricsResponse(BaseModel):
    """ Schema para reportar las métricas de rendimiento del modelo. """
    accuracy: float
    f1_score: float
    train_size: int
    test_size: int
    
class FeatureImportanceResponse(BaseModel):
    """ Schema para reportar la importancia de características. """
    importance: Dict[str, float]