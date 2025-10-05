# web/dashboard/index.py
import dash
from dash import dcc, html, Input, Output, State
import json
import os
import pandas as pd
import requests
import base64
import io

# ========== 1. DEFINICIÓN DE LA APP Y CONFIGURACIÓN ==========
app = dash.Dash(__name__, external_stylesheets=['https://cdn.jsdelivr.net/npm/bootswatch@4.5.2/dist/cyborg/bootstrap.min.css', 'https://use.fontawesome.com/releases/v5.8.1/css/all.css', '/assets/custom.css'], suppress_callback_exceptions=True)
server = app.server

# ========== 2. LÓGICA DE API CLIENT Y CARGA DE DATOS ==========
API_URL = "http://localhost:8000/models/predict"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FEATURE_NAMES_PATH = os.path.join(PROJECT_ROOT, 'models', 'feature_names.json')
METRICS_PATH = os.path.join(PROJECT_ROOT, 'models', 'latest_metrics.json')

def load_metrics():
    """Carga las métricas del modelo desde el archivo JSON."""
    try:
        with open(METRICS_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return {"accuracy": 0, "f1_score": 0}

def get_all_feature_names():
    try:
        with open(FEATURE_NAMES_PATH) as f: return json.load(f).get("feature_names", [])
    except FileNotFoundError: return []

def predict_exoplanet(user_inputs: dict):
    all_feature_names = get_all_feature_names()
    if not all_feature_names: return {"error": "Archivo 'feature_names.json' no encontrado."}
    if user_inputs.get('koi_prad', 0) > 1000: return {"error": "Datos inválidos: Radio planetario irreal."}
    features_payload = {name: 0.0 for name in all_feature_names}
    features_payload.update(user_inputs)
    try:
        response = requests.post(API_URL, json={"features": features_payload})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e: return {"error": f"Error de conexión con la API: {e}."}

# ========== 3. COMPONENTES DE LAYOUT ==========
def create_sidebar():
    return html.Div(id="sidebar", children=[
        html.Div(className="sidebar-header", children=[
            html.Img(src="/assets/img/LOGO_EXOAI.jpg", className="sidebar-logo"),
            html.H2("EXOAI", className="sidebar-title")
        ]),
        html.Ul(className="nav flex-column", children=[
            html.Li(className="nav-item", children=dcc.Link("Inicio", href="/", className="nav-link")),
            html.Li(className="nav-item", children=dcc.Link("Demo de Misión", href="/demo", className="nav-link")),
            html.Li(className="nav-item", children=dcc.Link("Nuestro Equipo", href="/equipo", className="nav-link")),
        ])
    ])

def create_demo_content():
    metrics = load_metrics()
    return html.Div([
        html.H1("Demo de Análisis de Misión", className="text-white mb-4"),
        
        # --- SECCIÓN DE MÉTRICAS REINTEGRADA ---
        html.H4("Rendimiento del Modelo Actual", className="text-white-50"),
        html.Div(className="row mb-4", children=[
            html.Div(className="col-md-6 mb-3", children=html.Div(className="metric-card", children=[
                html.Div("Accuracy", className="metric-title"),
                html.Div(f"{metrics.get('accuracy', 0):.2%}", className="metric-value")
            ])),
            html.Div(className="col-md-6 mb-3", children=html.Div(className="metric-card", children=[
                html.Div("F1-Score", className="metric-title"),
                html.Div(f"{metrics.get('f1_score', 0):.3f}", className="metric-value")
            ]))
        ]),
        html.Hr(),
        # --- FIN DE LA SECCIÓN DE MÉTRICAS ---

        html.P("Carga un archivo CSV con datos de una misión (Kepler, TESS, etc.) para que nuestra IA lo analice en bloque.", className="text-white-50"),
        html.Div(className="row justify-content-center mt-4", children=[
            html.Div(className="col-lg-8", children=html.Div(className="card bg-dark text-white p-4", children=[
                html.H4("Cargar Archivo de Misión (CSV)"),
                dcc.Upload(id='upload-data', children=html.Div(['Arrastra o selecciona un archivo CSV para análisis']), className="drag-area", multiple=False),
                dcc.Loading(id="loading-spinner", type="circle", children=html.Div(id='file-upload-output', className="mt-3"))
            ]))
        ])
    ])

def create_team_content():
    def member_card(image_path, name):
        return html.Div(className="col-lg-2 col-md-4 col-sm-6 mb-4", children=html.Div(className="card text-center bg-dark text-white h-100", children=[
            html.Img(src=image_path, className="card-img-top rounded-circle mt-3 mx-auto", style={'width': '120px', 'height': '120px', 'object-fit': 'cover', 'border': '2px solid #00e_bf_ff'}),
            html.Div(className="card-body", children=[html.H5(name, className="card-title"), html.P("NASA Space Apps 2025", className="card-text text-muted")])
        ]))
    return html.Div([
        html.H1("Nuestro Equipo 🚀", className="text-white mb-4 text-center"),
        html.Div(className="row justify-content-center", children=[
            member_card("/assets/img/caleb.jpg", "Caleb"), member_card("/assets/img/isabeau.jpg", "Isabeau"),
            member_card("/assets/img/nehemias.jpg", "Nehemias"), member_card("/assets/img/samuel.jpg", "Samuel"),
            member_card("/assets/img/daniel.jpg", "Daniel"),
        ])
    ])

def create_home_content():
    return html.Div([
        html.H1("Bienvenido a ExoAI Detector", className="text-white"),
        html.P("La plataforma de IA para la caza de exoplanetas.", className="lead text-white-50 mb-5"),
        html.Div(className="vision-card", children=[
            html.H2("Nuestra Visión", className="vision-title"),
            html.P("Democratizar el descubrimiento de exoplanetas, transformando los vastos archivos de datos de la NASA en conocimiento accesible. Usamos la inteligencia artificial para acelerar la búsqueda de nuevos mundos, permitiendo a científicos y entusiastas por igual participar en la próxima era de la exploración espacial.", className="lead mt-3")
        ])
    ])

# ========== 4. DEFINICIÓN DEL LAYOUT PRINCIPAL ==========
app.layout = html.Div(id="app-container", children=[
    dcc.Location(id='url', refresh=False),
    create_sidebar(),
    html.Div(id="page-content")
])

# ========== 5. CALLBACKS ==========
@app.callback(Output('page-content', 'children'), Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/demo': return create_demo_content()
    elif pathname == '/equipo': return create_team_content()
    else: return create_home_content()

@app.callback(Output('file-upload-output', 'children'), Input('upload-data', 'contents'), State('upload-data', 'filename'), prevent_initial_call=True)
def handle_file_upload(contents, filename):
    if contents is None: return html.Div()
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            required_cols = ['koi_period', 'koi_duration', 'koi_depth', 'koi_impact', 'koi_prad', 'koi_model_snr']
            if not all(col in df.columns for col in required_cols):
                return html.Div(f"Error: El archivo CSV debe contener las columnas: {', '.join(required_cols)}", className="alert alert-danger")

            results = [predict_exoplanet(row[required_cols].to_dict()) for _, row in df.iterrows()]
            
            positives = sum(1 for r in results if r.get("prediction_label") == "Exoplaneta Confirmado")
            false_positives = len(results) - positives
            
            return html.Div(className="result-summary-container", children=[
                html.Div(className="result-summary-header", children=[
                    html.H4(f"Resultados del Análisis de '{filename}'"),
                ]),
                html.Div(className="row", children=[
                    html.Div(className="col-md-4", children=html.Div(className="result-stat-card", children=[
                        html.I(className="fas fa-rocket icon total"),
                        html.Div(len(results), className="value"),
                        html.Div("Candidatos Analizados", className="label")
                    ])),
                    html.Div(className="col-md-4", children=html.Div(className="result-stat-card", children=[
                        html.I(className="fas fa-check-circle icon confirmed"),
                        html.Div(positives, className="value"),
                        html.Div("Exoplanetas Encontrados", className="label")
                    ])),
                    html.Div(className="col-md-4", children=html.Div(className="result-stat-card", children=[
                        html.I(className="fas fa-times-circle icon false-positive"),
                        html.Div(false_positives, className="value"),
                        html.Div("Falsos Positivos", className="label")
                    ]))
                ])
            ])
        else:
            return html.Div("Error: Por favor, carga un archivo CSV.", className="alert alert-danger")
    except Exception as e:
        return html.Div(f"Hubo un error al procesar el archivo: {e}", className="alert alert-danger")

# ========== 6. EJECUCIÓN DEL SERVIDOR ==========
if __name__ == '__main__':
    app.run(debug=True, port=8050)