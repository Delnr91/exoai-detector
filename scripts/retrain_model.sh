# scripts/retrain_model.sh (Ejemplo de script de producción)

#!/bin/bash
# Script de CI/CD: Automatiza el reentrenamiento del modelo final

echo "--- Iniciando reentrenamiento del Ensemble V3 Final ---"
# Activa el entorno virtual
source venv/bin/activate
# Ejecuta el caso de uso (el entry point del entrenamiento)
python -m src.application.use_cases.train_model_use_case
echo "--- Reentrenamiento Finalizado. Modelo v3 actualizado en /models ---"