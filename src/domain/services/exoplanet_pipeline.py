import pandas as pd
import numpy as np
import logging
import os
from src.domain.services.pipeline_modules.data_cleaner import DataCleaner
from src.domain.services.pipeline_modules.feature_creator import FeatureCreator
from src.domain.services.pipeline_modules.data_finalizer import DataFinalizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ExoplanetPreprocessor:

    """
    Clase que encapsula el pipeline avanzado modular para la preparación de datos de exoplanetas.
    El pipeline consta de los siguientes pasos:
    1. Carga y Selección Inicial (DataCleaner)
    2. Limpieza y Filtros Científicos (DataCleaner)
    3. Manejo de Valores Faltantes (DataCleaner)
    4. Feature Engineering Astronómico (FeatureCreator) 
    5. Feature Engineering Estadístico (FeatureCreator)
    6. Balanceo de Clases (DataFinalizer)  
    """
    
    def __init__(self, data_path: str = './data/kepler_koi.csv'):
        self.data_path = data_path
        self.cleaner = DataCleaner()
        self.creator = FeatureCreator()
        self.finalizer = DataFinalizer()
        self.feature_names = []

    def fit_transform_complete(self, target_col='koi_disposition', n_splits=5) -> tuple:

        """
        Ejecuta el pipeline completo de preprocesamiento de datos.
        """

        logging.info("= INICIO DEL PIPELINE DE PREPROCESAMIENTO DE EXOPLANETAS =")
        logging.info("=" * 50)

        # 1. Carga y Selección Inicial (DataCleaner)
        df = self.cleaner.load_and_select(self.data_path)

        # 2. Limpieza y Filtros Científicos (DataCleaner)
        df_processed = self.cleaner.apply_scientific_filters(df)
        
        # 3. Manejo de Valores Faltantes (DataCleaner)
        df_processed = self.cleaner.handle_missing_values(df_processed)

        # 4. Feature Engineering Astronómico (FeatureCreator)
        df_processed = self.creator.create_astronomical_features(df_processed)
        
        # 5. Feature Engineering Estadístico (FeatureCreator)
        df_processed = self.creator.create_statistical_features(df_processed)
        
        # Separar features y target
        X_raw = df_processed.drop(columns=['label', 'target_class', 'kepid'], errors='ignore')
        y_raw = df_processed['target_class']
        
        # 6. Balanceo de Clases (DataFinalizer)
        # CAMBIO CLAVE 1: Aseguramos que el output del target sea un DataFrame.
        X_balanced, y_balanced = self.finalizer.balance_classes(X_raw, y_raw.rename('target'))
        
        # Guardar nombres de features ANTES del escalado
        self.feature_names = list(X_balanced.columns)

        # 7. Escalado Robusto (DataFinalizer)
        X_final_scaled = self.finalizer.scale_features(X_balanced)

        # 8. Creación de Splits Temporales (DataFinalizer)
        temporal_splits = self.finalizer.create_temporal_splits(X_final_scaled, y_balanced.values, n_splits=n_splits)
        
        logging.info("=" * 50)
        logging.info(f"PIPELINE FINALIZADO. Features totales: {len(self.feature_names)}")
        
        return X_final_scaled, y_balanced.values, temporal_splits