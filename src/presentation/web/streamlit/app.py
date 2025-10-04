import streamlit as st

import score
import metrics_explain

# Configuración inicial de la página
st.set_page_config(
    page_title="🔭 Exoplanet AI Detector (NASA 2025)", 
    layout="wide"
)

st.title("🌌 Detección de Exoplanetas con IA (NASA 2025)")

# Navegación lateral
st.sidebar.title("Navegación del Dashboard")
page = st.sidebar.radio("Ir a:", ["Predicción en Vivo", "Métricas & Interpretación"])

# Enrutamiento (Esta parte ya estaba correcta)
if page == "Predicción en Vivo":
    score.render()
elif page == "Métricas & Interpretación":
    metrics_explain.render()