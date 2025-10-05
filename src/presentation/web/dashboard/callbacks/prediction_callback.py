from dash import Input, Output, State, html, dcc
import json
from services import api_client # Importación simple

def register_callbacks(app):
    INPUT_FEATURES = {
        'koi_period': 'Período Orbital (días)', 'koi_duration': 'Duración del Tránsito (horas)',
        'koi_depth': 'Profundidad del Tránsito (ppm)', 'koi_impact': 'Parámetro de Impacto',
        'koi_prad': 'Radio Planetario (Radios Terrestres)', 'koi_model_snr': 'Relación Señal-Ruido (SNR)'
    }

    @app.callback(Output('prediction-inputs-container', 'children'), Input('golden-case-dropdown', 'value'))
    def update_form_inputs(selected_case_json):
        # ... (código interno sin cambios)
        case_data = {}
        if selected_case_json != 'manual': case_data = json.loads(selected_case_json)
        form_inputs, row_inputs = [], []
        for i, (key, label) in enumerate(INPUT_FEATURES.items()):
            default_value = case_data.get(key, 10.0)
            input_el = html.Div(className="form-group", children=[
                html.Label([label, " ", html.Abbr("?", title=f"Parámetro: {key}")]),
                dcc.Input(id=f'input-{key}', type='number', value=default_value, className="form-control bg-dark text-white")])
            row_inputs.append(html.Div(input_el, className="col-md-6"))
            if (i + 1) % 2 == 0:
                form_inputs.append(html.Div(row_inputs, className="row"))
                row_inputs = []
        if row_inputs: form_inputs.append(html.Div(row_inputs, className="row"))
        return form_inputs

    @app.callback(Output('prediction-output', 'children'), Input('predict-button', 'n_clicks'), [State(f'input-{key}', 'value') for key in INPUT_FEATURES.keys()], prevent_initial_call=True)
    def handle_prediction(n_clicks, *values):
        # ... (código interno sin cambios)
        user_inputs = dict(zip(INPUT_FEATURES.keys(), values))
        if any(v is None for v in values): return html.Div("Error: Todos los campos deben tener un valor.", className="alert alert-warning")
        result = api_client.predict_exoplanet(user_inputs)
        if "error" in result: return html.Div(f"Error: {result['error']}", className="alert alert-danger")
        label, conf, habit = result.get("prediction_label", "E"), result.get("confidence_score", 0), result.get('is_potentially_habitable', False)
        summary = f"Análisis de candidato con período de **{user_inputs['koi_period']} días** y radio de **{user_inputs['koi_prad']}x la Tierra** ha concluido."
        if "Confirmado" in label:
            alert, header = "alert alert-success", f"✅ Resultado: {label}"
            extra = html.H5("¡Potencialmente Habitable!", className="text-success") if habit else ""
        else:
            alert, header = "alert alert-info", f"❌ Resultado: {label}"
            extra = ""
        return html.Div(className=alert, children=[html.H4(header), html.P(f"Confianza: {conf:.2%}"), html.Hr(), dcc.Markdown(summary), extra])