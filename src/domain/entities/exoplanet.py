from dataclasses import dataclass
from typing import Optional

@dataclass
class LightCurve:
    """Entidad o Value Object: Representa los datos de brillo de la estrella."""
    time: list
    flux: list
    error: list
    
    def normalize(self):
        """Lógica científica de preprocesamiento (ej. detrending o scaling)."""
        # Normalización simple: (flux - mean) / error
        mean_flux = sum(self.flux) / len(self.flux)
        self.flux = [(f - mean_flux) / self.error[i] for i, f in enumerate(self.flux)]
           
@dataclass
class Exoplanet:
    """
    Entidad: Representa un exoplaneta candidato con sus atributos clave.
    """
    kepid: str
    period_days: float
    radius_earth: float
    equilibrium_temp: float
    confidence_score: float
    
    def is_potentially_habitable(self) -> bool:

        """
        Determina si el exoplaneta es potencialmente habitable basado en criterios científicos.     
        Criterios (ejemplos):
        - Temperatura de equilibrio entre 250K y 350K   
        - Tamaño entre 0.5 y 2.0 veces el radio terrestre
        - Alta confianza en la detección (confidence_score > 0.90)
        Retorna True si cumple todos los criterios, False en caso contrario.
        """
        is_in_zone = 250 <= self.equilibrium_temp <= 350
        is_earth_size = 0.5 <= self.radius_earth <= 2.0
        is_high_confidence = self.confidence_score >= 0.90 
        
        return is_in_zone and is_earth_size and is_high_confidence
    