# dashboard/index.py
from dash import html
from app import app
from components import header, prediction_form, team_section
from callbacks import prediction_callbacks

# 1. Registramos los callbacks
prediction_callbacks.register_callbacks(app)

# 2. Definimos el layout
app.layout = html.Div(className="container-fluid", children=[
    header.layout(),
    html.Div(
        className="container",
        style={'paddingTop': '80px'},
        children=[
            prediction_form.layout(),
            team_section.layout()
        ]
    )
])

# 3. Ejecutamos el servidor
if __name__ == '__main__':
    app.run_server(debug=True, port=8050)