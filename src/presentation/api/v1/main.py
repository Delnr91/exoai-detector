from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.presentation.api.v1.endpoints import models

# Configuración básica de la aplicación FastAPI
app = FastAPI(
    title="Exoplanet Detection ML API (NASA Hackathon)",
    description="API REST para detección de exoplanetas usando Random Forest y Clean Architecture.",
    version="1.0.0"
)

# Configurar CORS para conectar Streamlit/React con FastAPI
origins = [
    "http://localhost:8501",  # Puerto por defecto de Streamlit
    "http://localhost:3000",  # Si usas React
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir router de modelos (endpoints como /models/predict)
app.include_router(models.router, prefix="/models", tags=["Models"])

@app.get("/health", tags=["Health"])
async def health_check():

    """Endpoint simple para verificar que la API está operativa."""
    return {"status": "UP", "model_loaded": True}

@app.on_event("startup")
async def startup_event():
    
    # Aquí podríamos cargar el modelo si no lo hiciéramos en el adaptador
    print("🚀 API de Exoplanetas arrancada. Lista para predicciones.")