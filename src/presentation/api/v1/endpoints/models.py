from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging
from src.presentation.api.v1.schemas.schemas import (
    PredictRequest, PredictResponse, MetricsResponse, FeatureImportanceResponse
)
from src.infrastructure.adapters.ml_adapter import RandomForestAdapter
from src.application.services.exoplanet_service import ExoplanetService

# Configuración del logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicialización de componentes
ML_REPOSITORY = RandomForestAdapter()
EXOPLANET_SERVICE = ExoplanetService(ml_repository=ML_REPOSITORY)
logger.info("Servicio ExoplanetService y Modelo ML cargados correctamente.")

router = APIRouter()

@router.post("/predict", response_model=PredictResponse)
async def predict_exoplanet(req: PredictRequest):
    """ Clasifica un candidato a exoplaneta a partir de un diccionario de características. """
    try:
        # Obtener el orden correcto de las features desde el adaptador
        feature_order = ML_REPOSITORY.get_feature_names()
        
        # Validar que todas las features necesarias están en el request
        if not all(feature in req.features for feature in feature_order):
            missing = sorted([f for f in feature_order if f not in req.features])
            raise ValueError(f"Faltan características en la petición: {missing}")

        # Ordenar las features del request según lo esperado por el modelo
        ordered_features = [req.features[name] for name in feature_order]

        result = EXOPLANET_SERVICE.predict(features=ordered_features)

        prediction_label = "Exoplaneta Confirmado" if result['prediction'] == 1 else "Candidato Falso"
        
        # Lógica de dominio simple para habitabilidad
        is_habitable = (result['prediction'] == 1 and 
                        req.features.get('koi_prad', 100) < 2.5 and 
                        req.features.get('koi_steff', 0) > 4000)

        return PredictResponse(
            prediction_label=prediction_label,
            confidence_score=result['confidence'],
            prediction_value=result['prediction'],
            model_version=result['model_name'],
            is_potentially_habitable=is_habitable
        )
    except ValueError as e:
        logger.warning(f"Error de validación de features: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error en la entrada de features: {str(e)}")
    except Exception as e:
        logger.error(f"Error interno en la predicción: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")

@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """ Obtiene las métricas finales del modelo. """
    metrics = ML_REPOSITORY.get_metrics()
    if not metrics:
        raise HTTPException(status_code=404, detail="Métricas no encontradas.")
    return MetricsResponse(**metrics)

@router.get("/feature-importance", response_model=FeatureImportanceResponse)
async def get_feature_importance():
    """ Obtiene la importancia de características (top 10). """
    importance = ML_REPOSITORY.get_feature_importance()
    if not importance:
        return FeatureImportanceResponse(importance={"No disponible": 0.0})
    
    # Ordenar y devolver solo el top 10 para eficiencia
    top_10 = dict(sorted(importance.items(), key=lambda item: item[1], reverse=True)[:10])
    return FeatureImportanceResponse(importance=top_10)