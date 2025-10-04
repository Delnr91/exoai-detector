class ExoplanetDomainError(Exception):

    """Excepción base para errores de negocio astronómico."""
    pass
    
class FeatureImputationError(ExoplanetDomainError):

    """Lanzada si la imputación de características falla."""
    pass
    
class InsufficientDataError(ExoplanetDomainError):

    """Lanzada si no hay suficientes datos para procesar."""
    pass