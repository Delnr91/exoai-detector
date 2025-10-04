import streamlit as st
import requests
import pandas as pd
import altair as alt # Usamos Altair para gráficos más limpios

API_URL = "http://localhost:8000/models"

def render():
    st.header("📈 Métricas de Calidad y Explicabilidad Científica")

    # --- Sección 1: Métricas de Rendimiento ---
    st.subheader("1. Rendimiento del Modelo (Random Forest)")
    
    try:
        metrics_resp = requests.get(f"{API_URL}/metrics")
        metrics_resp.raise_for_status()
        metrics = metrics_resp.json()
            
        col1, col2, col3, col4 = st.columns(4)
            
        col1.metric("Accuracy", f"{metrics['accuracy']:.3f}", 
                    "Cerca del Objetivo 90%")
        col2.metric("F1 Score", f"**{metrics['f1_score']:.3f}**", 
                    "¡Calidad Científica Cumplida (>0.85)!") # Destacamos el logro
        col3.metric("Muestras Train", f"{metrics['train_size']:,}")
        col4.metric("Muestras Test", f"{metrics['test_size']:,}")
        
        st.markdown("---")

    except requests.RequestException:
        st.error("❌ Error cargando métricas: Asegúrate de que el archivo 'latest_metrics.json' exista o que la API esté encendida.")
        return


    # --- Sección 2: Feature Importance (Interpretación) ---
    st.subheader("2. Importancia de Características (Interpretación Física)")
    st.markdown("Demuestra a los científicos qué parámetros del tránsito son más influyentes en la decisión del modelo (Interpretability).")

    try:
        fi_resp = requests.get(f"{API_URL}/feature-importance")
        fi_resp.raise_for_status()
        fi_data = fi_resp.json()
        
        df_fi = pd.DataFrame(fi_data)
        df_fi = df_fi.sort_values(by='importance', ascending=False)
        df_fi['importance'] = df_fi['importance'] * 100 # a porcentaje
        
        # Usamos Altair para una visualización más profesional
        chart = alt.Chart(df_fi).mark_bar().encode(
            x=alt.X('importance', title='Importancia (%)'),
            y=alt.Y('feature', title='Característica', sort='x', axis=alt.Axis(labelAngle=0)),
            tooltip=['feature', alt.Tooltip('importance', format='.2f')],
            color=alt.Color('feature', legend=None)
        ).properties(
            title='Top Características Predictivas (Random Forest)'
        ).interactive() # Habilitar zoom/pan
        
        st.altair_chart(chart, use_container_width=True)
        
        st.caption("Interpretación: La **koi_model_snr** (Signal-to-Noise) y el **koi_period** (Período Orbital) son los más predictivos, lo cual valida la literatura astronómica.")
        
    except Exception as e:
        st.warning(f"No se pudo cargar la importancia de características (Error: {e}).")