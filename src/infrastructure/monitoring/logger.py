import logging
import sys

def setup_logger(name: str = "exoplanet-ml"):
    
    """Configura un logger global para MLOps."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Console handler para stdout
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    )
    ch.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(ch)

    return logger

logger = setup_logger()