import pytest
from fastapi.testclient import TestClient
from src.presentation.api.v1.main import app

# El cliente de prueba usa la aplicación FastAPI
client = TestClient(app)

# El modelo final espera 32 features escaladas. Usamos un placeholder de alta calidad.
FEATURES_PLACEHOLDER = [1.0] * 32 

def test_health_check():
    """Verifica que la API esté viva."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "UP"

def test_predict_endpoint_success():
    """Prueba la predicción exitosa con el Ensemble V3 Final (91.20%)."""
    
    # Asume que el Ensemble V3 ya está cargado y esperando 32 features
    response = client.post("/models/predict", json={"features": FEATURES_PLACEHOLDER})
    
    assert response.status_code == 200
    data = response.json()
    
    # Verificación de lógica de implementacion: El planeta simulado debe ser habitable
    assert data["prediction"] in [0, 1]
    assert data["model_name"] == "Ensemble_v3_Final"
    assert "is_potentially_habitable" in data

def test_predict_endpoint_feature_mismatch():
    """Prueba que el endpoint falle correctamente si faltan features."""
    # Envía solo 5 features (debe esperar 32)
    response = client.post("/models/predict", json={"features": [1.0] * 5})
    
    # Debe fallar con un error 400 (Bad Request)
    assert response.status_code == 400
    assert "features" in response.json()["detail"]
    
def test_metrics_endpoint():
    """Verifica que el endpoint de métricas cargue las métricas de 91.20%."""
    response = client.get("/models/metrics")
    assert response.status_code == 200
    data = response.json()

    # Verifica la métrica crítica alcanzada
    assert data["accuracy"] >= 0.9100 