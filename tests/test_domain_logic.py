import pytest
from src.domain.entities.exoplanet import Exoplanet
from src.domain.exceptions.exceptions import ExoplanetDomainError

def test_exoplanet_is_potentially_habitable_success():
    """Prueba que un planeta en zona habitable (DDD) se clasifique como tal."""

    # Parámetros: Radio 1.5 R_Earth (OK), Temp 290K (OK), Confianza 91.20% (OK)
    exo = Exoplanet(
        kepid="TEST-01",
        period_days=100.0,
        radius_earth=1.5,
        equilibrium_temp=290.0,
        confidence_score=0.9120
    )
    assert exo.is_potentially_habitable() is True

def test_exoplanet_is_not_habitable_low_confidence():
    """Prueba que falle por baja confianza del modelo."""
    
    # Baja Confianza (0.70)
    exo = Exoplanet(
        kepid="TEST-02",
        period_days=100.0,
        radius_earth=1.5,
        equilibrium_temp=290.0,
        confidence_score=0.70
    )
    assert exo.is_potentially_habitable() is False