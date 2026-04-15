from dash import html, dcc
import dash_bootstrap_components as dbc

def create_layout():
    return dbc.Container([

        # Title
        dbc.Row([
            dbc.Col(html.H1("Diabetes Risk Dashboard", className="text-center mb-4"))
        ]),

        # Filters / Inputs
        dbc.Row([
            dbc.Col([
                html.Label("Select Feature"),
                dcc.Dropdown(
                    id="feature-dropdown",
                    options=[
                        {"label": "Age", "value": "age"},
                        {"label": "BMI", "value": "bmi"},
                        {"label": "Glucose (Fasting)", "value": "glucose_fasting"},
                        {"label": "Glucose (Postprandial)", "value": "glucose_postprandial"}
                    ],
                    value="age"
                )
            ], width=4),
        ], className="mb-4"),

        # Graphs
        dbc.Row([
            dbc.Col(dcc.Graph(id="main-graph"), width=12)
        ]),

        # Model output section (future)
        dbc.Row([
            dbc.Col(html.Div(id="prediction-output"), width=12)
        ])

    ], fluid=True)