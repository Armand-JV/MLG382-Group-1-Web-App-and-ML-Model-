from dash import Input, Output
import plotly.express as px
import pandas as pd

# Load your processed data
df = pd.read_csv("../data/processed/clustered_data.csv")

def register_callbacks(app):

    @app.callback(
        Output("main-graph", "figure"),
        Input("feature-dropdown", "value")
    )
    def update_graph(feature):

        if feature not in df.columns:
            return {}

        fig = px.histogram(
            df,
            x=feature,
            color="cluster"
        )

        return fig