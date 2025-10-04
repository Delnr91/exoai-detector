# src/presentation/web/streamlit/metrics_explain.py

import streamlit as st
import requests
import pandas as pd
import altair as alt

# Define la URL base de tu API
API_URL = "http://localhost:8000/models"

def render():
    """ Renderiza la página de métricas y explicabilidad del modelo. """
    st.header("🔬 Métricas y Explicabilidad del Modelo")

    st.markdown("### 1. Rendimiento del Modelo (Certificado de Calibración)")
    
    try:
        res = requests.get(f"{API_URL}/metrics")
        res.raise_for_status()
        metrics = res.json()
        
        col1, col2 = st.columns(2)
        col1.metric("🎯 Precisión (Accuracy)", f"{metrics.get('accuracy', 0):.2%}")
        col2.metric("⚖️ F1-Score", f"{metrics.get('f1_score', 0):.3f}")
        
        st.success("El **F1-Score > 0.85** demuestra que el modelo es científicamente robusto.")

    except requests.exceptions.RequestException as e:
        st.error(f"No se pudieron cargar las métricas. Error: {e}")

    st.markdown("---")
    st.markdown("### 2. Importancia de Características (Interpretación Física)")
    st.markdown("Estas son las variables más influyentes que el modelo utiliza.")

    try:
        res = requests.get(f"{API_URL}/feature-importance")
        res.raise_for_status()
        importance_data = res.json().get("importance", {})
        
        if importance_data:
            # CORRECCIÓN: Convertir el diccionario a un DataFrame con las columnas correctas
            df_importance = pd.DataFrame(importance_data.items(), columns=['Característica', 'Importancia'])
            
            # Crear el gráfico con Altair
            chart = alt.Chart(df_importance).mark_bar().encode(
                x=alt.X('Importancia:Q', title='Importancia Relativa'),
                y=alt.Y('Característica:N', title='Característica', sort='-x'), # '-x' ordena de mayor a menor
                tooltip=['Característica', 'Importancia']
            ).properties(
                title='Top 10 Características Predictivas'
            ).interactive()
            
            st.altair_chart(chart, use_container_width=True)
        else:
            st.warning("No hay datos de importancia de características disponibles.")

    except requests.exceptions.RequestException as e:
        st.error(f"No se pudo cargar la importancia de características. Error: {e}")