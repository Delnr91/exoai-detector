# 🌌 ExoAI  - NASA Space Apps Challenge 2025

**ExoAI ** es un sistema de Inteligencia Artificial de alto rendimiento diseñado para detectar exoplanetas a partir de los datos de las misiones de la NASA, logrando una precisión superior al 91%. El proyecto cuenta con una interfaz web interactiva construida con Dash que permite el análisis en tiempo real de candidatos a exoplanetas.

---

## ✨ MVP (Producto Mínimo Viable) Explicado

El MVP de este proyecto consiste en un sistema completamente funcional que:
1.  **Entrena un modelo** de Machine Learning con datos de la misión Kepler, alcanzando una precisión y F1-Score superiores al 90%.
2.  **Expone este modelo** a través de una API REST robusta y escalable construida con FastAPI.
3.  **Proporciona una interfaz de usuario** interactiva que permite a los usuarios analizar datos de misiones completas (cargando archivos CSV) y recibir un resumen estadístico de los hallazgos de la IA.
4.  **Demuestra inteligencia** al validar los datos de entrada y rechazar aquellos que no son físicamente plausibles para un exoplaneta.

---

## 🛠️ Stack de Tecnologías

| Categoría         | Tecnología                                     | Propósito                                      |
| ----------------- | ---------------------------------------------- | ---------------------------------------------- |
| **Backend** | FastAPI, Uvicorn                               | Creación de una API REST asíncrona y de alto rendimiento. |
| **Frontend** | Dash                                           | Construcción de un dashboard científico e interactivo. |
| **Machine Learning**| Scikit-learn, Pandas, NumPy                    | Entrenamiento del modelo (Random Forest) y manipulación de datos. |
| **Despliegue** | Docker                                         | Contenerización para un despliegue fácil y reproducible.  |

---

## 🏗️ Arquitectura y Patrones de Diseño

Este proyecto no es solo un script, es un sistema de software robusto construido con principios de grado industrial.

* **Arquitectura Limpia + Hexagonal:** El núcleo del proyecto (la "ciencia" y la lógica de negocio) está completamente aislado de los detalles externos como la API o la base de datos. Esto significa que podemos cambiar la interfaz de usuario de Dash a otra tecnología, o cambiar cómo guardamos los modelos, sin tener que reescribir el cerebro del sistema. Es un diseño adaptable y preparado para el futuro.

* **Domain-Driven Design (DDD):** El código "habla el lenguaje de los científicos". Usamos términos como `ExoplanetCandidate`, `TransitDepth` y `OrbitalPeriod` en lugar de variables genéricas. Esto hace que el código sea más fácil de entender, mantener y validar por expertos en astronomía.

---

```
## 📂 Estructura de Archivos

La estructura del proyecto separa claramente el frontend del backend, facilitando el desarrollo y el mantenimiento.

exoai-detector/
├── 📂 dashboard/             # <-- FRONTEND (Tu App de Dash, autocontenida)
│   ├── 📂 assets/            # CSS, Imágenes (logos, equipo), Favicon
│   │   ├── 📂 img/
│   │   └── 📄 custom.css
│   └── 📄 index.py           # El archivo principal y único que contiene toda la lógica de Dash
│
├── 📂 src/                  # <-- BACKEND (Toda la lógica de negocio y la API)
│   ├── 📄 __init__.py
│   ├── 📂 application/        # Casos de uso (ej. entrenar modelo)
│   │   ├── 📄 __init__.py
│   │   ├── 📂 services/
│   │   └── 📂 use_cases/
│   ├── 📂 domain/             # Entidades y lógica científica pura
│   │   ├── 📄 __init__.py
│   │   ├── 📂 entities/
│   │   ├── 📂 pipeline_modules/
│   │   ├── 📂 repositories/
│   │   └── 📂 services/
│   ├── 📂 infrastructure/     # Conexión con herramientas externas (modelo, logs)
│   │   ├── 📄 __init__.py
│   │   ├── 📂 adapters/
│   │   └── 📂 monitoring/
│   └── 📂 presentation/       # La capa que se expone al mundo exterior
│       ├── 📄 __init__.py
│       └── 📂 api/
│           └── 📂 v1/
│               ├── 📂 endpoints/
│               ├── 📂 schemas/
│               └── 📄 main.py  # Punto de entrada de la API
│
├── 📂 data/                 # Datasets de la NASA (kepler_koi.csv, etc.) y casos de prueba
├── 📂 models/               # Modelos entrenados (.pkl) y artefactos (metrics.json, etc.)
├── 📂 tests/                # Pruebas unitarias y de integración
├── 📄 requirements.txt       # Dependencias del proyecto
└── 📄 README.md              # Documentación principal del proyecto

```
---

## 🛰️ Datos y Modelo de Machine Learning

* **Misiones Utilizadas:** El modelo principal fue entrenado utilizando el catálogo de **Kepler Objects of Interest (KOI)** de la NASA, que es una de las fuentes de datos más ricas y validadas para el método de tránsito. El sistema está preparado para analizar también datos de otras misiones como **TESS (TOI)**.

* **Modelo de ML:** Se utiliza un **Ensemble de Clasificadores** con **Random Forest** como el componente principal. Este modelo fue elegido por su alto rendimiento, su capacidad para manejar datos complejos y, crucialmente, por su **interpretabilidad**, permitiéndonos entender qué características son más importantes para tomar una decisión.

---

## ✨ Características Destacadas

* **Análisis Rápido de Misiones:** La función principal de la demo permite cargar un archivo `.csv` con cientos o miles de candidatos de una misión. La aplicación procesa el archivo en bloque, llama a la API para cada candidato y devuelve un resumen visual y estadístico de los resultados, demostrando la potencia y escalabilidad del sistema.

* **Validación de Dominio (Lógica "Anti-Camiones"):** Para asegurar la integridad de las predicciones, el backend incluye una capa de validación. Si un usuario introduce datos que no son físicamente plausibles para un exoplaneta (ej. un "planeta" con un radio 2000 veces el de la Tierra), la API lo rechaza con un error claro. Esto previene el uso indebido y garantiza que solo se procesen datos astronómicos válidos.

---

## 🚀 Cómo Probar el Proyecto Localmente

Sigue estos pasos para ejecutar el proyecto en tu máquina.

**Pre-requisitos:**
* Python 3.9+
* Git

**Instalación y Ejecución:**

1.  **Clonar el Repositorio:**
    ```bash
    git clone [https://github.com/Delnr91/exoai-detector](https://github.com/Delnr91/exoai-detector)
    cd exoai-detector
    ```

2.  **Crear y Activar Entorno Virtual:**
    ```bash
    # En Windows
    python -m venv venv
    venv\Scripts\activate

    # En macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Instalar Dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Entrenar el Modelo (solo la primera vez):**
    Este comando ejecuta el pipeline de entrenamiento y genera los archivos necesarios en la carpeta `models/`.
    ```bash
    python -m src.application.use_cases.train_model_use_case
    ```

5.  **Iniciar el Backend (API):**
    Abre una terminal y ejecuta:
    ```bash
    uvicorn src.presentation.api.v1.main:app --reload --port 8000
    ```
    Deja esta terminal corriendo.

6.  **Iniciar el Frontend (Dashboard):**
    Abre una **segunda terminal** y ejecuta:
    ```bash
    python dashboard/index.py
    ```

7.  **Acceder a la Aplicación:**
    Abre tu navegador web y ve a la siguiente dirección:
    **<http://localhost:8050>**

¡Ahora puedes interactuar con el ExoAI Detector!
