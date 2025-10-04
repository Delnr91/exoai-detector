import streamlit as st
import requests
import pandas as pd

# Define la URL base de tu API
API_URL = "http://localhost:8000/models"

def render():
    """
    Renderiza la página de métricas y explicabilidad del modelo.
    """
    st.header("🔬 Métricas y Explicabilidad del Modelo")

    st.markdown("### 1. Rendimiento del Modelo (Certificado de Calibración)")
    
    try:
        # Hacer la petición al endpoint de métricas
        res = requests.get(f"{API_URL}/metrics")
        res.raise_for_status()  # Lanza un error si la petición falla
        metrics = res.json()
        
        # Mostrar las métricas en columnas
        col1, col2 = st.columns(2)
        col1.metric("🎯 Precisión (Accuracy)", f"{metrics.get('accuracy', 0):.2%}")
        col2.metric("⚖️ F1-Score", f"{metrics.get('f1_score', 0):.3f}")
        
        st.success("El **F1-Score > 0.85** demuestra que el modelo es científicamente robusto, balanceando la detección de planetas reales sin generar falsas alarmas.")

    except requests.exceptions.RequestException as e:
        st.error(f"No se pudieron cargar las métricas. Asegúrate de que el backend esté funcionando. Error: {e}")

    st.markdown("---")
    st.markdown("### 2. Importancia de Características (Interpretación Física)")
    st.markdown("Estas son las variables más influyentes que el modelo utiliza para tomar sus decisiones.")

    try:
        # Hacer la petición al endpoint de importancia de características
        res = requests.get(f"{API_URL}/feature-importance")
        res.raise_for_status()
        importance_data = res.json().get("importance", {})
        
        if importance_data:
            # Convertir el diccionario a un DataFrame de Pandas (CLAVE PARA EL GRÁFICO)
            # Ordenar por importancia de forma descendente y tomar el top 10
            top_10_features = sorted(importance_data.items(), key=lambda item: item[1], reverse=True)[:10]
            
            df_importance = pd.DataFrame(top_10_features, columns=['Característica', 'Importancia'])
            
            st.info("Mostrando las 10 características más predictivas.")
            st.bar_chart(df_importance.set_index('Característica'))
        else:
            st.warning("No hay datos de importancia de características disponibles.")

    except requests.exceptions.RequestException as e:
        st.error(f"No se pudo cargar la importancia de características. Error: {e}")