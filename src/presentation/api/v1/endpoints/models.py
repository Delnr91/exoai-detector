from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
import logging 
import json
import os
from src.presentation.api.v1.schemas.schemas import (
    PredictRequest, PredictResponse, MetricsResponse, FeatureImportanceResponse
)
from src.infrastructure.adapters.ml_adapter import RandomForestAdapter
from src.application.services.exoplanet_service import ExoplanetService 

# Configuración básica de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicialización del Adaptador y Servicio de Aplicación
ML_REPOSITORY = None
EXOPLANET_SERVICE = None

try:
    ML_REPOSITORY = RandomForestAdapter()
    # Inyectar el Adaptador (Infraestructura) en el Servicio (Aplicación)
    EXOPLANET_SERVICE = ExoplanetService(ml_repository=ML_REPOSITORY)
    logger.info("Servicio ExoplanetService y Modelo ML cargados correctamente.")
except FileNotFoundError as e:
    logger.error(f"Error al cargar el modelo al inicio: {e}. El endpoint /predict no funcionará.")
except Exception as e:
    logger.critical(f"Error CRÍTICO al inicializar el Adaptador o Servicio: {e}")

router = APIRouter()

# 1. Endpoint de Predicción
@router.post("/predict", response_model=PredictResponse)
async def predict_exoplanet(req: PredictRequest):

    """
    Clasifica el candidato y evalúa su habitabilidad usando el Ensemble Final.
    """
    logger.info(f"Predicción solicitada. Features count: {len(req.features)}")
    
    if EXOPLANET_SERVICE is None:
        raise HTTPException(status_code=503, detail="Servicio no disponible: El modelo ML no está cargado.")

    try:
        features_for_model = req.features 
        
        # NOTA: Debes ajustar estos valores en tu frontend Streamlit para el demo.
        astronomical_params = {
            'kepid': 'Demo-9120',
            'koi_period': features_for_model[0] if len(features_for_model) > 0 else 0.0,
            # Simulamos que el radio y la temperatura de equilibrio están en el input 
            # o se calculan en el pipeline. Usaremos valores fijos para la demo de la Entidad.
            'koi_prad': 1.5, # Simulación: 1.5 Tierras
            'koi_teq': 300.0 # Simulación: 300 Kelvin (Zona habitable)
        }
        
        # Llama al Servicio de Aplicación, el cual usa la Entidad (lógica de negocio) y el Adaptador
        result = EXOPLANET_SERVICE.classify_and_evaluate(features_for_model, astronomical_params)
        
        logger.info(f"Predicción completada. Resultado: {result['prediction']}, Habitabilidad: {result['is_potentially_habitable']}")
        
        return PredictResponse(
            prediction=result['prediction'],
            confidence=result['confidence'],
            model_name=result['model_name'],
            is_potentially_habitable=result['is_potentially_habitable']
        )
    except ValueError as e:
        logger.error(f"Error de validación de features: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error en la entrada de features: {str(e)}")
    except Exception as e:
        logger.error(f"Error interno en la predicción: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error interno en el servidor: {str(e)}")


# 2. Endpoint de Métricas
@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Obtiene las métricas finales del modelo Ensemble."""
    if ML_REPOSITORY is None:
        raise HTTPException(status_code=503, detail="Servicio no disponible.")
    
    metrics = ML_REPOSITORY.get_metrics()
    if not metrics:
        raise HTTPException(status_code=404, detail="Métricas no encontradas. Ejecute el entrenamiento.")
        
    return MetricsResponse(**metrics)


# 3. Endpoint de Importancia de Features
@router.get("/feature-importance", response_model=FeatureImportanceResponse)
async def get_feature_importance():
    """Obtiene la importancia de features para la interpretabilidad científica."""
    if ML_REPOSITORY is None:
        raise HTTPException(status_code=503, detail="Servicio no disponible.")
        
    importance = ML_REPOSITORY.get_feature_importance()
    if not importance:
        
        importance = {f"feature_{i}": 1.0 / ML_REPOSITORY.EXPECTED_FEATURES_COUNT for i in range(ML_REPOSITORY.EXPECTED_FEATURES_COUNT)}
        
    return FeatureImportanceResponse(importance=importance)