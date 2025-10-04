import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import RobustScaler
from sklearn.utils import resample
from sklearn.model_selection import TimeSeriesSplit

class DataFinalizer:

    """
    Responsabilidad: Balanceo estratégico de clases, escalado de features 
    y creación de splits temporales.
    """

    # Usamos RobustScaler, mejor para outliers
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = RobustScaler() 

    def balance_classes(self, X: pd.DataFrame, y: pd.Series) -> tuple:

        """Oversample simple de la minoritaria hasta igualar la mayoritaria."""

        logging.info("Balanceo simple (Oversample) para igualar clases")

        df_combined = pd.concat([X, y], axis=1) 
        
        class_counts = df_combined['target'].value_counts()
        minority_label = class_counts.idxmin()
        majority_count = class_counts.max()
        
        target_size = majority_count
        
        balanced_dfs = []
        for class_label in df_combined['target'].unique():
            class_df = df_combined[df_combined['target'] == class_label]
            
            is_minority = class_label == minority_label

            resampled_df = resample(
                class_df, 
                n_samples=target_size if is_minority else len(class_df), 
                replace=is_minority, 
                random_state=self.random_state
            )
            balanced_dfs.append(resampled_df)

        df_balanced = pd.concat(balanced_dfs, ignore_index=True).sample(
            frac=1, random_state=self.random_state
        )

        X_balanced = df_balanced.drop(columns=['target'])
        y_balanced = df_balanced['target']

        final_dist = y_balanced.value_counts().sort_index()
        logging.info(f"Distribución final: {dict(final_dist)}")
        
        # Validación CRÍTICA: Asegurar que ambas clases existan
        if len(final_dist) < 2:
            raise ValueError("Error de balanceo: Solo se encontró una clase en el set final.")

        return X_balanced, y_balanced

    def scale_features(self, X: pd.DataFrame) -> np.ndarray:
        """Escalado robusto con RobustScaler."""
        logging.info(f"📏 Escalado - método: RobustScaler (Mediana + IQR)")
        
        X_scaled = self.scaler.fit_transform(X)
        logging.info(" Escalado completado")
        return X_scaled

    def create_temporal_splits(self, X: np.ndarray, y: np.ndarray, n_splits=5) -> list:
        """Crea splits temporales (TimeSeriesSplit) para validación robusta."""
        logging.info(f"📅 VALIDACIÓN TEMPORAL: {n_splits} splits")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        splits = list(tscv.split(X))

        for i, (train_idx, test_idx) in enumerate(splits):
            train_ratio = len(train_idx) / len(X) * 100
            logging.info(f"   Split {i+1}: Train={len(train_idx):,} ({train_ratio:.1f}%)")
        
        return splits