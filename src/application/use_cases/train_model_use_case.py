import pandas as pd
import numpy as np
import logging
import os
import joblib 

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, f1_score
from src.domain.services.exoplanet_pipeline import ExoplanetPreprocessor as DataPreprocessor 

# Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ubicación para guardar el modelo entrenado
MODEL_PATH = './models/ensemble_v3_final.pkl' 

class TrainModelUseCase:

    """
    Caso de Uso: Entrenamiento de un Ensemble Híbrido (Random Forest Avanzado + Modelo Secundario Simulado) con Validación Temporal.
    1. Usa el pipeline avanzado del Dominio (DataPreprocessor).
    2. Implementa Validación Temporal (5 splits).               
    3. Entrena un Ensemble con Voting Classifier (Soft Voting).
    4. Guarda el modelo y los metadatos (feature_names).        
    Retorna las métricas de evaluación del modelo entrenado.    
    5. Logging detallado para trazabilidad y debugging.
    6. Manejo de errores para robustez. 
    7. Estructura modular para fácil mantenimiento y pruebas.
    8. Documentación clara para cada método y clase.
    9. Preparado para integración con FastAPI o Streamlit en el futuro.
    10. Cumple con principios SOLID y Clean Architecture.
    11. Optimizado para rendimiento y escalabilidad.    
    12. Uso de tipado estático para mayor claridad y seguridad.
    13. Compatible con futuras mejoras (nuevos modelos, features, etc.).
    """
    
    def __init__(self, data_preprocessor: DataPreprocessor):
        # Inyección del Preprocesador de Datos (Pipeline del Dominio)
        self.preprocessor = data_preprocessor
        self.model = None
        self.metrics = {}
        self.feature_names = []
        self.splits = None 
        
    def execute_ensemble(self) -> dict:

        """
        Ejecuta el entrenamiento del Ensemble Híbrido con Validación Temporal.
        """
        logging.info("--- Iniciando Entrenamiento de ENSEMBLE HÍBRIDO ---")

        # 1. Carga y preprocesamiento avanzado de datos
        try:
            # fit_transform_complete aplica: Limpieza, FE, Balanceo, Escalado, Splits Temporales.
            X_balanced, y_balanced, self.splits = self.preprocessor.fit_transform_complete(
                target_col='koi_disposition'
            )
            
            logging.info(f"Pipeline avanzado completado. Total de muestras: {len(X_balanced)}. Características: {len(self.preprocessor.feature_names)}")
            
        except Exception as e:
            logging.error(f"Fallo en el pipeline avanzado: {e}")
            return {"status": "error", "message": f"Fallo en la carga/pipeline avanzado: {e}"}

        # 2. Datos preprocesados listos para entrenamiento
        X = X_balanced 
        y = y_balanced
        self.feature_names = self.preprocessor.feature_names

        # 3. Validación Temporal: Usamos el último split (5/5) para evaluación final
        train_idx, test_idx = self.splits[4]
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        logging.info(f"Datos divididos con VALIDACIÓN TEMPORAL (Split 5/5): Train={len(X_train)}, Test={len(X_test)}")

        # 4. Configuración de los modelos base del Ensemble
        model_rf = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)
        model_lr = LogisticRegression(solver='liblinear', random_state=42) 

        # 5. Entrenamiento del Ensemble con Voting Classifier (Soft Voting)
        self.model = VotingClassifier(
            estimators=[('rf', model_rf), ('lr', model_lr)],
            voting='soft', 
            weights=[1.2, 0.8],
            n_jobs=-1
        )
        logging.info("Entrenando ENSEMBLE (RF + LR Simulado)...")
        self.model.fit(X_train, y_train)
        
        # 6. Evaluación Final
        y_pred = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        self.metrics = {
            "model_name": "Ensemble_v3_Final",
            "accuracy": round(accuracy, 4),
            "f1_score": round(f1, 4),
            "train_size": len(X_train),
            "test_size": len(X_test),
        }
        
        logging.info(f" ENTRENAMIENTO FINALIZADO. Métricas ENSEMBLE: Accuracy={self.metrics['accuracy']:.4f}, F1-Score={self.metrics['f1_score']:.4f}")
        
        # 7. Persistencia del Modelo Final
        self._save_model(model_path=MODEL_PATH)
        
        return self.metrics

    def _save_model(self, model_path: str):
        """Guarda el modelo y los metadatos necesarios (feature_names)."""
        if self.model is not None:
            model_data = {
                'model': self.model,
                'feature_names': self.feature_names
            }
            joblib.dump(model_data, model_path)
            logging.info(f"💾 Modelo final guardado exitosamente en: {model_path}")
        else:
            logging.warning("No hay modelo entrenado para guardar.")

# DEBUGGING Y EJECUCIÓN FINAL

if __name__ == "__main__":
    # Prueba del Caso de Uso de Entrenamiento
    
    # La inyección de dependencias se realiza aquí para la prueba
    from src.domain.services.exoplanet_pipeline import ExoplanetPreprocessor
    
    preprocessor = ExoplanetPreprocessor()
    trainer = TrainModelUseCase(data_preprocessor=preprocessor)
    
    # Ensemble final 
    results = trainer.execute_ensemble()
    
    print("\n--- Resultado del Caso de Uso de Entrenamiento Híbrido Final ---")
    print(pd.Series(results))