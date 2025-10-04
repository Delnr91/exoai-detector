import logging
from typing import List
from src.domain.entities.exoplanet import Exoplanet # Usamos la entidad de dominio
from src.domain.repositories.ml_repository import MLRepository # Usamos el PORT

logging.basicConfig(level=logging.INFO)

class ExoplanetService:

    """
    Servicio de Aplicación: Traduce la predicción técnica a una entidad 
    de dominio rica en lógica de negocio (DDD).
    """
    def __init__(self, ml_repository: MLRepository):
        """Inyección del Adaptador de Modelo ML (el Port)."""
        self.ml_repository = ml_repository
        
    def classify_and_evaluate(self, features: List[float], astronomical_params: dict) -> dict:

        """
        Clasifica el candidato y evalúa su habitabilidad.
        """
        # 1. Obtener predicción del Adaptador (Infraestructura)
        prediction_result = self.ml_repository.predict(features)
        
        # 2. Crea la Entidad de Dominio (Exoplanet)
        exoplanet_entity = Exoplanet(
            kepid=astronomical_params.get('kepid', 'N/A'),
            period_days=astronomical_params.get('koi_period', 0.0),
            radius_earth=astronomical_params.get('koi_prad', 0.0),
            equilibrium_temp=astronomical_params.get('koi_teq', 0.0),
            confidence_score=prediction_result['confidence']
        )
        
        # 3. Regla de Negocio del Dominio
        is_habitable = exoplanet_entity.is_potentially_habitable()
        
        logging.info(f"Evaluación de Habitabilidad para {exoplanet_entity.kepid}: {is_habitable}")
        
        # 4. Retorno del resultado enriquecido
        return {
            "prediction": prediction_result['prediction'],
            "confidence": prediction_result['confidence'],
            "model_name": prediction_result['model_name'],
            "is_potentially_habitable": is_habitable
        }