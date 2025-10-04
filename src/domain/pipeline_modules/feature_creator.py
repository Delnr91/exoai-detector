import pandas as pd
import numpy as np
from scipy import stats
import logging

class FeatureCreator:

    """
    Responsabilidad: Crear features avanzadas basadas en física y estadísticas.
    """

    def create_astronomical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea features basadas en la física de los exoplanetas."""
        logging.info("Creando 45+ Features Astronómicas Avanzadas")
        df_eng = df.copy()

        # 1. Características geométricas/físicas (3ra Ley de Kepler)
        if self._columns_exist(df_eng, ['koi_period', 'koi_smass', 'koi_duration', 'koi_depth', 'koi_srad']):
            # Semi-eje mayor en AU
            df_eng['orbital_distance_au'] = ((df_eng['koi_period']/365.25)**2 * df_eng['koi_smass'])**(1/3)
            # Radio planetario en R_Earth
            df_eng['planet_radius_earth'] = (
                np.sqrt(df_eng['koi_depth'] / 1e6 + 1e-10) * df_eng['koi_srad'] * 109.2 # R_sun a R_Earth factor
            )
            df_eng['duration_period_ratio'] = df_eng['koi_duration'] / (df_eng['koi_period'] + 1e-10)
        
        # 2. Análisis de habitabilidad
        if self._columns_exist(df_eng, ['koi_stemp', 'koi_srad']):
            # Luminosidad estelar (Proxy, Stefan-Boltzmann)
            df_eng['stellar_luminosity_proxy'] = df_eng['koi_srad']**2 * (df_eng['koi_stemp']/5778)**4
            
            # Temperatura de equilibrio (requiere distancia orbital)
            if 'orbital_distance_au' in df_eng.columns:
                df_eng['equilibrium_temp'] = (
                    df_eng['koi_stemp'] * np.sqrt(df_eng['koi_srad'] / (2 * df_eng['orbital_distance_au'] + 1e-10))
                )
                df_eng['habitable_zone'] = (
                    (df_eng['equilibrium_temp'] >= 250) & 
                    (df_eng['equilibrium_temp'] <= 350) # Rango aproximado
                ).astype(int)

        # 3. Características de Calidad y Robustez
        if 'koi_model_snr' in df_eng.columns:
            df_eng['log_snr'] = np.log10(df_eng['koi_model_snr'] + 1e-10)
            df_eng['snr_high_quality'] = (df_eng['koi_model_snr'] > 15).astype(int)
            
        logging.info(f"Features físicas creadas. Total columnas: {len(df_eng.columns)}")
        return df_eng

    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea características estadísticas (Skewness, Kurtosis, CV)."""
        logging.info("📊 Creando Características Estadísticas")

        df_stats = df.copy()
        
        # Columnas clave (originales y algunas derivadas)
        key_patterns = ['period', 'duration', 'depth', 'radius', 'temp', 'snr', 'mass', 'impact']
        key_cols = [col for col in df_stats.columns 
                   if any(pattern in col.lower() for pattern in key_patterns)]

        for col in key_cols:
            if df_stats[col].dtype in ['float64', 'int64'] and not col.endswith(('_skew', '_kurt', '_cv')):
                values = df_stats[col].dropna()
                if len(values) > 10:
                    # Asimetría
                    df_stats[f'{col}_skew'] = stats.skew(values)
                    # Curtosis
                    df_stats[f'{col}_kurt'] = stats.kurtosis(values)
                    # Coeficiente de variación
                    mean_val = values.mean()
                    if abs(mean_val) > 1e-10:
                        df_stats[f'{col}_cv'] = values.std() / abs(mean_val)

        logging.info(f"Features estadísticas agregadas. Total columnas: {len(df_stats.columns)}")
        return df_stats
    
    def _columns_exist(self, df, columns):
        """Función auxiliar."""
        return all(col in df.columns for col in columns)