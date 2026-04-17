import os

import dash
import dash_bootstrap_components as dbc
from layout import create_layout
from callbacks import register_callbacks

# Initialize app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Diabetes Dashboard"

# Layout
app.layout = create_layout()

# Callbacks
register_callbacks(app)

# Run server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run_server(
        host="0.0.0.0",
        port=port,
        debug=True
    )