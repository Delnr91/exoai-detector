import streamlit as st
import requests
import json
import os

# URL de la API
API_URL = "http://localhost:8000/models"
FEATURE_NAMES_PATH = './models/feature_names.json'

def load_feature_names():
    """Carga la lista completa de nombres de características desde el archivo JSON."""
    try:
        with open(FEATURE_NAMES_PATH, 'r') as f:
            data = json.load(f)
            return data.get("feature_names", [])
    except FileNotFoundError:
        return []

def render():
    st.header("🪐 Predicción en Vivo para Candidatos a Exoplanetas")
    st.markdown("Introduce los parámetros de un candidato para obtener una clasificación instantánea de nuestro modelo de IA.")

    # Cargar todos los nombres de las características
    ALL_FEATURE_NAMES = load_feature_names()
    
    if not ALL_FEATURE_NAMES:
        st.error("No se encontró el archivo 'feature_names.json'. Por favor, ejecuta el script de entrenamiento primero.")
        return

    # Usaremos las 6 características más comunes e intuitivas para la entrada del usuario
    INPUT_FEATURES = {
        'koi_period': {'label': 'Período Orbital (días)', 'value': 80.0, 'help': 'Tiempo que tarda el planeta en orbitar su estrella.'},
        'koi_duration': {'label': 'Duración del Tránsito (horas)', 'value': 3.0, 'help': 'Tiempo que el planeta tarda en cruzar por delante de su estrella.'},
        'koi_depth': {'label': 'Profundidad del Tránsito (ppm)', 'value': 900.0, 'help': 'Disminución del brillo de la estrella durante el tránsito.'},
        'koi_impact': {'label': 'Parámetro de Impacto', 'value': 0.5, 'help': 'Qué tan centrado es el tránsito (0=ecuador, 1=borde).'},
        'koi_prad': {'label': 'Radio Planetario (Radios Terrestres)', 'value': 2.5, 'help': 'Tamaño del planeta en comparación con la Tierra.'},
        'koi_model_snr': {'label': 'Relación Señal-Ruido (SNR)', 'value': 25.0, 'help': 'Qué tan clara es la señal del tránsito respecto al ruido.'},
    }

    # Crear el formulario
    with st.form("prediction_form"):
        st.subheader("Parámetros del Candidato")
        
        user_inputs = {}
        cols = st.columns(2)
        
        for i, (name, config) in enumerate(INPUT_FEATURES.items()):
            user_inputs[name] = cols[i % 2].number_input(
                label=config['label'],
                value=config['value'],
                help=config['help'],
                step=0.1,
                format="%.2f"
            )

        submitted = st.form_submit_button("Clasificar Candidato", type="primary")

    if submitted:
        # Construir el diccionario completo de 32 características
        features_payload = {name: 0.0 for name in ALL_FEATURE_NAMES} # Empezar con ceros
        features_payload.update(user_inputs) # Sobrescribir con los valores del usuario

        try:
            with st.spinner('Nuestro telescopio de IA está analizando el candidato...'):
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"features": features_payload}
                )
                response.raise_for_status()

            result = response.json()
            
            prediction_label = result.get("prediction_label", "Error")
            confidence = result.get("confidence_score", 0)

            if prediction_label == "Exoplaneta Confirmado":
                st.success(f"**Resultado: {prediction_label}**")
                st.metric("Confianza del Modelo", f"{confidence:.2%}")
                st.balloons()
            else:
                st.warning(f"**Resultado: {prediction_label}**")
                st.metric("Confianza del Modelo", f"{confidence:.2%}")

            st.caption(f"Análisis realizado con el modelo: {result.get('model_version')}")

        except requests.exceptions.RequestException as e:
            st.error(f"Error al conectar con la API: {e}")