import streamlit as st
import requests
from typing import List

# URL de la API (debe coincidir con donde se ejecuta tu FastAPI)
API_URL = "http://localhost:8000/models" 

# Las 6 características que el usuario ingresará (las más importantes)
INPUT_FEATURE_NAMES = [
    'koi_period', 'koi_impact', 'koi_duration', 
    'koi_depth', 'koi_prad', 'koi_model_snr'
]

# El modelo FINAL espera 32 features. Debemos simular las 26 restantes (features derivadas).
EXPECTED_FEATURE_COUNT = 32

def render():
    st.header("🪐 Predicción en Vivo para Candidatos")
    st.markdown("Ingresa los **6 parámetros críticos** y el sistema simulará el resto de las 32 características para la clasificación.")
    
    input_features = []
    
    # 1. Recolección de las 6 Features Clave
    st.subheader("Parámetros Críticos (Input)")
    cols = st.columns(3)

    for i, name in enumerate(INPUT_FEATURE_NAMES):
        default_value = 10.0 if 'period' in name else (500.0 if 'depth' in name else (50.0 if 'snr' in name else 1.0))
            
        val = cols[i % 3].number_input(
            f"{name.upper()}", 
            value=default_value, 
            min_value=0.0,
            format="%.4f",
            key=name
        )
        input_features.append(val)
        
    st.markdown("---")

    if st.button(" CLASIFICAR CANDIDATO Y EVALUAR HABITABILIDAD"):
        
        # 2. SIMULACIÓN: Crear las 32 Features Totales
        # Usamos los 6 inputs y simulamos las 26 features derivadas con un valor por defecto (1.0)
        # Esto es vital para que la API no falle por tamaño incorrecto (32 features)
        
        final_features: List[float] = input_features + [1.0] * (EXPECTED_FEATURE_COUNT - len(INPUT_FEATURE_NAMES))
        
        if len(final_features) != EXPECTED_FEATURE_COUNT:
             st.error(f"Error de simulación: Se esperaban {EXPECTED_FEATURE_COUNT} features, pero se simularon {len(final_features)}.")
             return
             
        try:
            with st.spinner(f"Consultando Ensemble Final ({EXPECTED_FEATURE_COUNT} features)..."):
                
                # Consumir el endpoint de predicción
                response = requests.post(
                    f"{API_URL}/predict", 
                    json={"features": final_features}
                )
                response.raise_for_status() 

            result = response.json()
            
            # 3. Mostrar Resultado Enriquecido (Incluye lógica DDD de habitabilidad)
            is_exoplanet = result['prediction'] == 1
            is_habitable = result.get('is_potentially_habitable', False)
            
            if is_exoplanet:
                st.success(f"✅ Predicción: **EXOPLANETA CONFIRMADO / CANDIDATO**")
                if is_habitable:
                    st.balloons()
                    st.metric("EVALUACIÓN CIENTÍFICA", "¡POTENCIALMENTE HABITABLE!", delta="DDD Lógica")
                else:
                    st.info("No cumple los criterios de habitabilidad (Radio/Temperatura).")
            else:
                st.warning(f"❌ Predicción: **FALSO POSITIVO**")
            
            st.caption(f"Modelo: {result['model_name']} | Confianza: **{result['confidence']:.2%}**")

        except requests.exceptions.ConnectionError:
            st.error("❌ Error de Conexión: Asegúrate de que el servidor FastAPI esté corriendo en http://localhost:8000.")
        except requests.RequestException as e:
            st.error(f"Error en la API: {e}. Asegúrate de que el Adaptador ML apunte a 'ensemble_v3_final.pkl'.")