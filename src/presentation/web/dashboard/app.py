import dash
app = dash.Dash(__name__, external_stylesheets=['https://cdn.jsdelivr.net/npm/bootswatch@4.5.2/dist/cyborg/bootstrap.min.css'], suppress_callback_exceptions=True)
server = app.server