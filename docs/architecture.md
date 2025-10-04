# Arquitectura Final del Detector de Exoplanetas 

Este proyecto implementa la Arquitectura Limpia (Clean Architecture) combinada con Domain-Driven Design (DDD) y el patrón Hexagonal (Ports & Adapters) para garantizar mantenibilidad, testabilidad y escalabilidad, cumpliendo con los estándares de proyectos científicos NASA.

## Capas Clave

1.  **DOMAIN LAYER (Núcleo Científico):**
    * **Entidades:** `exoplanet.py` (Lógica de habitabilidad).
    * **Servicios:** `exoplanet_pipeline.py` (Orquesta el Feature Engineering de 32+ variables).
    * **Interfaces:** `ml_repository.py` (Define el contrato para cualquier modelo ML).

2.  **APPLICATION LAYER (Casos de Uso):**
    * **Servicios:** `exoplanet_service.py` (Orquesta la predicción y el chequeo de habitabilidad).
    * **Usecases:** `train_model_use_case.py` (Ejecuta el entrenamiento avanzado del Ensemble).

3.  **INFRASTRUCTURE LAYER (Adaptadores):**
    * **Adaptador ML:** `ml_adapter.py` (Implementa el contrato `MLRepository` y carga el modelo **Ensemble V3 Final (91.20% Acc)**).
    * **Config/Monitoring:** (Listas para integraciones MLOps).

4.  **PRESENTATION LAYER (Interfaces):**
    * **API:** FastAPI (`main.py`) sirve endpoints REST para predicción y métricas.
    * **Web:** Streamlit (`app.py`) Dashboard de Explicabilidad (XAI).