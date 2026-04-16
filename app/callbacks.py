from dash import Input, Output, State, callback_context
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import pickle
import joblib
from dash import html
import dash_bootstrap_components as dbc

# ── Load data & models ────────────────────────────────────────────────────────
df = pd.read_csv("../data/processed/clustered_data.csv")
cluster_summary = pd.read_csv("../data/processed/cluster_summary.csv")

try:
    feature_names = pd.read_csv("../data/processed/feature_names.csv")["feature"].tolist()
except Exception:
    feature_names = []

try:
    with open("../artifacts/random_forest_best_model.pkl", "rb") as f:
        rf_model = pickle.load(f)
except Exception:
    rf_model = None

try:
    with open("../artifacts/xgboost_best_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)
except Exception:
    xgb_model = None

try:
    preprocessor = joblib.load("../artifacts/preprocessor.pkl")
except Exception:
    preprocessor = None

try:
    label_encoder = pickle.load(open("../artifacts/label_encoder.pkl", "rb"))
except Exception:
    label_encoder = None

# Colour palette per cluster
CLUSTER_COLORS = {0: "#3b82f6", 1: "#f59e0b", 2: "#ef4444"}
CLUSTER_LABELS = {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk"}

PLOT_TEMPLATE = "plotly_white"


def _filter_df(cluster_val):
    if cluster_val == "all" or cluster_val is None:
        return df
    return df[df["cluster"] == int(cluster_val)]


def register_callbacks(app):

    # ── KPIs ─────────────────────────────────────────────────────────────────
    @app.callback(
        Output("kpi-total",    "children"),
        Output("kpi-highrisk", "children"),
        Output("kpi-avgbmi",   "children"),
        Output("kpi-avggluc",  "children"),
        Input("cluster-filter", "value"),
    )
    def update_kpis(cluster_val):
        d = _filter_df(cluster_val)
        total     = f"{len(d):,}"
        high_risk = f"{(d['cluster'] == 2).sum():,}" if "cluster" in d.columns else "—"
        avg_bmi   = f"{d['bmi'].mean():.1f}" if "bmi" in d.columns else "—"
        avg_gluc  = f"{d['glucose_fasting'].mean():.0f} mg/dL" if "glucose_fasting" in d.columns else "—"
        return total, high_risk, avg_bmi, avg_gluc

    # ── Main feature chart ────────────────────────────────────────────────────
    @app.callback(
        Output("main-graph", "figure"),
        Input("feature-dropdown", "value"),
        Input("chart-type",       "value"),
        Input("cluster-filter",   "value"),
    )
    def update_main_graph(feature, chart_type, cluster_val):
        d = _filter_df(cluster_val)
        if feature not in d.columns:
            return {}

        color_map = {str(k): v for k, v in CLUSTER_COLORS.items()}
        d = d.copy()
        d["cluster_label"] = d["cluster"].map(CLUSTER_LABELS)

        kwargs = dict(
            data_frame=d, color="cluster_label",
            color_discrete_map={v: CLUSTER_COLORS[k] for k, v in CLUSTER_LABELS.items()},
            template=PLOT_TEMPLATE,
            labels={"cluster_label": "Cluster"},
        )

        if chart_type == "histogram":
            fig = px.histogram(**kwargs, x=feature, barmode="overlay", opacity=0.75)
        elif chart_type == "box":
            fig = px.box(**kwargs, x="cluster_label", y=feature)
        elif chart_type == "scatter":
            fig = px.scatter(**kwargs, x=feature, y="cluster_label", opacity=0.5)
        elif chart_type == "violin":
            fig = px.violin(**kwargs, x="cluster_label", y=feature, box=True)
        else:
            fig = px.histogram(**kwargs, x=feature)

        fig.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor="white",
        )
        return fig

    # ── Cluster pie ───────────────────────────────────────────────────────────
    @app.callback(
        Output("cluster-pie", "figure"),
        Input("cluster-filter", "value"),
    )
    def update_pie(cluster_val):
        d = _filter_df(cluster_val)
        counts = d["cluster"].value_counts().reset_index()
        counts.columns = ["cluster", "count"]
        counts["label"] = counts["cluster"].map(CLUSTER_LABELS)
        fig = px.pie(
            counts, names="label", values="count",
            color="label",
            color_discrete_map={v: CLUSTER_COLORS[k] for k, v in CLUSTER_LABELS.items()},
            template=PLOT_TEMPLATE, hole=0.45,
        )
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10),
                          legend=dict(orientation="h", yanchor="top", y=-0.1))
        return fig

    # ── Diabetes stage bar ────────────────────────────────────────────────────
    @app.callback(
        Output("stage-bar", "figure"),
        Input("cluster-filter", "value"),
    )
    def update_stage_bar(cluster_val):
        d = _filter_df(cluster_val)
        if "diabetes_stage" not in d.columns or "cluster" not in d.columns:
            return {}
        grp = d.groupby(["cluster", "diabetes_stage"]).size().reset_index(name="count")
        grp["cluster_label"] = grp["cluster"].map(CLUSTER_LABELS)
        fig = px.bar(
            grp, x="cluster_label", y="count", color="diabetes_stage",
            barmode="stack", template=PLOT_TEMPLATE,
            labels={"cluster_label": "Cluster", "count": "Patients", "diabetes_stage": "Stage"},
        )
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        return fig

    # ── Scatter explorer ──────────────────────────────────────────────────────
    @app.callback(
        Output("scatter-plot", "figure"),
        Input("scatter-x",      "value"),
        Input("scatter-y",      "value"),
        Input("cluster-filter", "value"),
    )
    def update_scatter(x_feat, y_feat, cluster_val):
        d = _filter_df(cluster_val).copy()
        if x_feat not in d.columns or y_feat not in d.columns:
            return {}
        d["cluster_label"] = d["cluster"].map(CLUSTER_LABELS)
        fig = px.scatter(
            d, x=x_feat, y=y_feat, color="cluster_label",
            color_discrete_map={v: CLUSTER_COLORS[k] for k, v in CLUSTER_LABELS.items()},
            opacity=0.6, template=PLOT_TEMPLATE,
            labels={"cluster_label": "Cluster"},
            trendline="ols",
        )
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        return fig

    # ── Prediction ────────────────────────────────────────────────────────────
    @app.callback(
        Output("prediction-output", "children"),
        Input("predict-btn", "n_clicks"),
        State("input-age",      "value"),
        State("input-bmi",      "value"),
        State("input-glucose",  "value"),
        State("input-hba1c",    "value"),
        State("input-activity", "value"),
        State("input-stress",   "value"),
        State("model-selector", "value"),
        prevent_initial_call=True,
    )
    def run_prediction(n_clicks, age, bmi, glucose, hba1c, activity, stress, model_choice):
        model = rf_model if model_choice == "rf" else xgb_model
        model_name = "Random Forest" if model_choice == "rf" else "XGBoost"

        if model is None:
            return dbc.Alert("Model not loaded. Check artifacts directory.", color="warning")

        # Build input row matching training feature order
        input_dict = {
            "age": age, "bmi": bmi,
            "glucose_fasting": glucose, "hba1c": hba1c,
            "physical_activity_hours_per_week": activity,
            "stress_level": stress,
        }
        X_input = pd.DataFrame([input_dict])

        # Apply preprocessor if available
        if preprocessor is not None:
            try:
                X_input = preprocessor.transform(X_input)
            except Exception:
                pass

        try:
            pred = model.predict(X_input)[0]
            proba = model.predict_proba(X_input)[0]

            if label_encoder is not None:
                try:
                    pred_label = label_encoder.inverse_transform([pred])[0]
                except Exception:
                    pred_label = str(pred)
            else:
                pred_label = str(pred)

            # Risk colour
            if "high" in str(pred_label).lower() or pred == 2:
                color, icon = "danger", "🔴"
            elif "moderate" in str(pred_label).lower() or pred == 1:
                color, icon = "warning", "🟡"
            else:
                color, icon = "success", "🟢"

            proba_bars = []
            classes = label_encoder.classes_ if label_encoder else [str(i) for i in range(len(proba))]
            for cls, p in zip(classes, proba):
                proba_bars.append(
                    html.Div([
                        html.Div([
                            html.Span(cls, style={"fontSize": "0.8rem", "color": "#475569", "width": "120px", "display": "inline-block"}),
                            dbc.Progress(value=round(p * 100, 1), label=f"{p*100:.1f}%",
                                         color="primary", style={"height": "18px", "flex": 1}),
                        ], style={"display": "flex", "alignItems": "center", "gap": "10px", "marginBottom": "6px"})
                    ])
                )

            return dbc.Alert([
                html.H5(f"{icon} Predicted Stage: {pred_label}", className="alert-heading"),
                html.P(f"Model: {model_name}", className="mb-2 text-muted", style={"fontSize": "0.85rem"}),
                html.Hr(),
                html.P("Class probabilities:", className="mb-2 fw-semibold", style={"fontSize": "0.85rem"}),
                *proba_bars,
            ], color=color, style={"borderRadius": "10px"})

        except Exception as e:
            return dbc.Alert(f"Prediction error: {str(e)}", color="danger")

    # ── Cluster summary table ─────────────────────────────────────────────────
    @app.callback(
        Output("cluster-table", "children"),
        Input("cluster-filter", "value"),
    )
    def update_cluster_table(cluster_val):
        try:
            d = cluster_summary.copy()
        except Exception:
            # Build a live summary if CSV not loaded
            d = _filter_df(cluster_val).copy()
            if "cluster" not in d.columns:
                return html.P("No cluster data available.")
            numeric_cols = d.select_dtypes(include=np.number).columns.tolist()
            d = d.groupby("cluster")[numeric_cols].mean().round(2).reset_index()

        if cluster_val != "all":
            d = d[d["cluster"] == int(cluster_val)] if "cluster" in d.columns else d

        header = [html.Th(c, style={"fontWeight": "600", "fontSize": "0.8rem", "color": "#475569"}) for c in d.columns]
        rows = []
        for _, row in d.iterrows():
            rows.append(html.Tr([html.Td(str(v), style={"fontSize": "0.82rem"}) for v in row]))

        return dbc.Table(
            [html.Thead(html.Tr(header)), html.Tbody(rows)],
            bordered=False, striped=True, hover=True, responsive=True,
            style={"fontSize": "0.85rem"},
        )