import pandas as pd
import logging
from astropy.table import Table

class DataCleaner:

    """
    Responsabilidad: Carga robusta, limpieza astronómica y manejo seguro de NaNs.
    Basado en estándares y recomendaciones de la NASA para datos de exoplanetas.
    1. Carga robusta con Astropy.
    """
    
    # Filtros de calidad basados en estándares NASA
    ASTRO_FILTERS = {
        'koi_period': (0.5, 1000, 'Período orbital válido'),
        'koi_duration': (0.5, 48, 'Duración tránsito observable'), 
        'koi_depth': (10, 100000, 'Profundidad detectable'),
        # 'koi_kepmag': (6, 20, 'Magnitud Kepler válida'), # Omitida por no estar en todos los datasets
        'koi_model_snr': (7.1, 1000, 'SNR umbral NASA')
    }
    
    CRITICAL_FEATURES = [
        'kepid', 'koi_disposition', 'koi_period', 'koi_duration', 
        'koi_depth', 'koi_model_snr', 'koi_impact', 'koi_prad',
        'koi_steff', 'koi_slogg', 'koi_srad', 'koi_smass'
    ]
    
    def load_and_select(self, data_path: str) -> pd.DataFrame:
        """Carga robusta con Astropy y selección de features críticas."""
        try:
            logging.info(f"Intentando cargar datos de {data_path} con Astropy...")
            astropy_table = Table.read(data_path, format='ascii')
            df = astropy_table.to_pandas()
            df.columns = df.columns.str.strip().str.lower()
        except Exception as e:
            logging.error(f"Error fatal al cargar con Astropy: {e}")
            raise 

        # Seleccionar solo features críticas que existan
        available_features = [col for col in self.CRITICAL_FEATURES if col in df.columns]
        df = df[available_features].copy()
        
        # Crear 'label' (0: FP, 1: Planet/Candidate)
        if 'koi_disposition' in df.columns:

            # 1. Asegura que la columna 'koi_disposition' esté en minúsculas para un mapeo seguro
            df['label'] = df['koi_disposition'].str.lower()
            
            # 2. Lógica de mapeo CRÍTICA: Confirmados y Candidatos son 1 (Exoplaneta)
            df['target_class'] = df['label'].apply(
                lambda x: 1 
                if x in ['confirmed', 'candidate'] 
                else 0
            )
            df = df.drop(columns=['koi_disposition'], errors='ignore')
            
        # VALIDACIÓN DEL MUESTREO (IMPORTANTE)
        logging.info(f"Distribución del Target: {df['target_class'].value_counts()}")
            
        logging.info(f"Datos cargados. Filas iniciales: {len(df)}. Features: {list(df.columns)}")
        return df

    def apply_scientific_filters(self, df: pd.DataFrame) -> pd.DataFrame:

        """
        Aplica filtros de calidad (SNR > 7.1, Periodo.) y elimina filas 
        con NaN en el target crítico al inicio.
        """
        logging.info("Aplicando limpieza y filtros astronómicos")
        initial_count = len(df)
        df_clean = df.copy()

        # 1. Eliminar filas donde no hay target (CRÍTICO)
        if 'label' in df_clean.columns:
            df_clean = df_clean.dropna(subset=['label'])
            logging.info(f"   Etiqueta objetivo (label) eliminadas: {initial_count - len(df_clean)}")
        
        # Recalcular después de eliminar NaNs de label
        initial_count = len(df_clean) 

        # 2. Aplicar filtros astronómicos
        for col, (min_val, max_val, desc) in self.ASTRO_FILTERS.items():
            if col in df_clean.columns:
                before = len(df_clean)
                df_clean = df_clean[
                    (df_clean[col] >= min_val) & 
                    (df_clean[col] <= max_val) &
                    df_clean[col].notna() # Elimina NaNs en las columnas que usamos para filtrar
                ]
                removed = before - len(df_clean)
                logging.info(f"   {desc}: -{removed:,} filas")

        reduction = (1 - len(df_clean)/initial_count) * 100
        logging.info(f" Reducción total: {reduction:.1f}% ({len(df_clean):,} filas restantes)")
        return df_clean

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:

        """
        Manejo seguro de NaNs: Imputa con la mediana de forma segura, 
        sin eliminar filas ya limpiadas.
        """
        logging.info(" Manejo seguro de valores faltantes (Imputación)")
        df_filled = df.copy()

        # Usamos la mediana general como imputación más robusta para variables numéricas
        for col in df_filled.columns:
            if df_filled[col].isnull().sum() > 0 and df_filled[col].dtype in ['float64', 'int64']:
                median_val = df_filled[col].median()
                df_filled[col] = df_filled[col].fillna(median_val)

        # Confirmamos que no hay NaNs en las features que pasaremos a FE
        nan_count = df_filled.drop(columns=['label', 'target_class'], errors='ignore').isnull().sum().sum()
        logging.info(f"✅ NaNs restantes en features: {nan_count}")
        
        # Último paso de seguridad: eliminar cualquier fila que aún tenga NaN (deberían ser pocas o ninguna)
        df_filled = df_filled.dropna()

        logging.info(f"✅ Limpieza de NaNs finalizada. Filas finales: {len(df_filled)}")
        return df_filled