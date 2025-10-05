import requests
import json
import os

API_URL = "http://localhost:8000/models/predict"
FEATURE_NAMES_PATH = './models/feature_names.json'

def get_all_feature_names():
    try:
        with open(FEATURE_NAMES_PATH) as f:
            return json.load(f).get("feature_names", [])
    except FileNotFoundError:
        return []

def predict_exoplanet(user_inputs: dict):
    all_feature_names = get_all_feature_names()
    if not all_feature_names:
        return {"error": "Archivo 'feature_names.json' no encontrado. Por favor, entrena el modelo."}
    if user_inputs.get('koi_prad', 0) > 1000:
        return {"error": "Datos inválidos: El radio planetario es demasiado grande..."}
    features_payload = {name: 0.0 for name in all_feature_names}
    features_payload.update(user_inputs)
    try:
        response = requests.post(API_URL, json={"features": features_payload})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Error de conexión con la API: {e}."}