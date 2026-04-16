import dash
import dash_bootstrap_components as dbc
from layout import create_layout
from callbacks import register_callbacks

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap",
    ],
    suppress_callback_exceptions=True,
)
app.title = "DiabetesIQ — Risk Dashboard"
app.index_string = app.index_string.replace(
    "<body>",
    "<body style='font-family: Inter, sans-serif;'>"
)

app.layout = create_layout()
register_callbacks(app)

if __name__ == "__main__":
    app.run(debug=True)