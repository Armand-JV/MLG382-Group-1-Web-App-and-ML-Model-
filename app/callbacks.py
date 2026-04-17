# TASK 6.2 & 6.3 – Diabetes Risk Dashboard Callbacks
# Handles: ML predictions, all visualisations, clustering insights
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from dash import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from dash import html
import dash_bootstrap_components as dbc
import joblib
import pickle
import warnings
warnings.filterwarnings("ignore")

# Colour palette (mirrors layout) 
C ={
    "navy":  "#0066FF",
    "teal":  "#1A9E8F",
    "red":   "#E84855",
    "amber": "#F5A623",
    "slate": "#4A5568",
    "light": "#F7FAFC",
    "white": "#FFFFFF",
}

CLUSTER_COLORS = ["#1A9E8F", "#F5A623", "#E84855", "#5A67D8", "#68D391"]
PLOT_LAYOUT = dict(
    plot_bgcolor="rgba(247,250,252,1)",
    paper_bgcolor="rgba(255,255,255,0)",
    font=dict(family="Nunito Sans, sans-serif", color="#4A5568"),
    margin=dict(l=20, r=20, t=40, b=20),
)

# Load data 
try:
    df = pd.read_csv(os.path.join(BASE_DIR, "data/processed/clustered_data.csv"))
except Exception:
    # Fallback: empty dataframe so dashboard still loads
    df = pd.DataFrame(columns=[
        "cluster", "diabetes_stage", "diagnosed_diabetes",
        "diabetes_risk_score", "age", "bmi", "glucose_fasting",
        "glucose_postprandial", "systolic_bp", "diastolic_bp",
        "cholesterol_total", "hba1c", "physical_activity_minutes_per_week",
    ])

# Load models 
try:
    import traceback
    xgb_model     = pickle.load(open(os.path.join(BASE_DIR, "artifacts/xgboost_best_model.pkl"), "rb"))
    preprocessor  = joblib.load(os.path.join(BASE_DIR, "data/processed/preprocessor.joblib"))
    label_encoder = joblib.load(os.path.join(BASE_DIR, "data/processed/label_encoder.joblib"))
    kmeans_model  = joblib.load(os.path.join(BASE_DIR, "artifacts/kmeans_model.joblib"))
except Exception as e:
    print(f"[callbacks] Model load error: {e}")
    xgb_model = preprocessor = label_encoder = kmeans_model = None

FEATURE_NAMES = [
    "Age", "BMI", "Glucose (Fasting)", "Glucose (Postprandial)",
    "Systolic BP", "Diastolic BP", "Cholesterol", "HbA1c", "Physical Activity",
]

# The 9 columns collected from the UI form
INPUT_COLS = [
    "age", "bmi", "glucose_fasting", "glucose_postprandial",
    "systolic_bp", "diastolic_bp", "cholesterol_total",
    "hba1c", "physical_activity_minutes_per_week",
]

# ALL numeric columns the preprocessor was trained on (from feature_names.csv)
ALL_NUMERIC_COLS = [
    "age", "alcohol_consumption_per_week", "physical_activity_minutes_per_week",
    "diet_score", "sleep_hours_per_day", "screen_time_hours_per_day",
    "family_history_diabetes", "hypertension_history", "cardiovascular_history",
    "bmi", "waist_to_hip_ratio", "systolic_bp", "diastolic_bp", "heart_rate",
    "cholesterol_total", "hdl_cholesterol", "ldl_cholesterol", "triglycerides",
    "glucose_fasting", "glucose_postprandial", "insulin_level", "hba1c",
    "diabetes_risk_score",
]

# ALL categorical columns the preprocessor was trained on
ALL_CAT_COLS = [
    "gender", "ethnicity", "education_level", "income_level",
    "employment_status", "smoking_status",
    "age_group", "bmi_category", "bp_category",   # engineered
]

# Sensible population-median defaults for columns not on the input form
COL_DEFAULTS = {
    "alcohol_consumption_per_week": 2.0,
    "diet_score": 5.0,
    "sleep_hours_per_day": 7.0,
    "screen_time_hours_per_day": 4.0,
    "family_history_diabetes": 0,
    "hypertension_history": 0,
    "cardiovascular_history": 0,
    "waist_to_hip_ratio": 0.85,
    "heart_rate": 72.0,
    "hdl_cholesterol": 50.0,
    "ldl_cholesterol": 100.0,
    "triglycerides": 150.0,
    "insulin_level": 10.0,
    "diabetes_risk_score": 0.5,
    # categorical defaults (most common category)
    "gender": "Male",
    "ethnicity": "White",
    "education_level": "Highschool",
    "income_level": "Middle",
    "employment_status": "Employed",
    "smoking_status": "Never",
}


def _apply_feature_engineering(row: dict) -> dict:
    """Mirror data_preprocessing.engineer_features for a single dict row."""
    age = row.get("age", 30)
    if age <= 30:
        row["age_group"] = "Young"
    elif age <= 45:
        row["age_group"] = "Middle-aged"
    elif age <= 60:
        row["age_group"] = "Senior"
    else:
        row["age_group"] = "Elderly"

    bmi = row.get("bmi", 22)
    if bmi < 18.5:
        row["bmi_category"] = "Underweight"
    elif bmi < 25:
        row["bmi_category"] = "Normal"
    elif bmi < 30:
        row["bmi_category"] = "Overweight"
    else:
        row["bmi_category"] = "Obese"

    sbp = row.get("systolic_bp", 120)
    dbp = row.get("diastolic_bp", 80)
    if sbp >= 140 or dbp >= 90:
        row["bp_category"] = "Hypertension"
    elif sbp >= 120 or dbp >= 80:
        row["bp_category"] = "Prehypertension"
    else:
        row["bp_category"] = "Normal"

    return row


def _build_full_input(vals: list) -> pd.DataFrame:
    """
    Build a single-row DataFrame that matches exactly what the preprocessor
    expects: all numeric + categorical columns including engineered features.
    """
    row = dict(zip(INPUT_COLS, vals))

    # Fill every missing column with its default
    for col in ALL_NUMERIC_COLS + ALL_CAT_COLS:
        if col not in row:
            row[col] = COL_DEFAULTS.get(col, 0)

    # Apply the same feature engineering as data_preprocessing.py
    row = _apply_feature_engineering(row)

    # Ensure column order matches: numerics first, then categoricals
    ordered_cols = ALL_NUMERIC_COLS + ALL_CAT_COLS
    return pd.DataFrame([{c: row[c] for c in ordered_cols}])


def register_callbacks(app):

    
    # TAB 1 – OVERVIEW: stat cards
    
    @app.callback(
        [Output("metric-total-patients", "children"),
         Output("metric-diabetes-cases", "children"),
         Output("metric-avg-risk",       "children"),
         Output("metric-clusters",       "children")],
        Input("cluster-filter", "value"),
    )
    def update_metrics(cluster_filter):
        fdf = df if cluster_filter == "all" else df[df["cluster"] == cluster_filter]
        total    = len(fdf)
        cases    = int(fdf["diagnosed_diabetes"].sum()) if "diagnosed_diabetes" in fdf.columns else 0
        avg_risk = round(fdf["diabetes_risk_score"].mean(), 2) if "diabetes_risk_score" in fdf.columns else 0
        n_clust  = fdf["cluster"].nunique() if "cluster" in fdf.columns else 0
        return total, cases, avg_risk, n_clust

    # Cluster dropdown refresh 
    @app.callback(
        Output("cluster-filter", "options"),
        Input("diabetes-stage-filter", "value"),
    )
    def update_cluster_options(stage_filter):
        fdf = df if stage_filter == "all" else df[df["diabetes_stage"] == stage_filter]
        opts = [{"label": "All Clusters", "value": "all"}]
        opts += [{"label": f"Cluster {int(c)}", "value": c}
                 for c in sorted(fdf["cluster"].unique())]
        return opts

    # Feature distribution chart
    @app.callback(
        Output("main-graph", "figure"),
        [Input("feature-dropdown",      "value"),
         Input("chart-type-dropdown",   "value"),
         Input("cluster-filter",        "value"),
         Input("diabetes-stage-filter", "value")],
    )
    def update_main_graph(feature, chart_type, cluster_filter, stage_filter):
        fdf = df.copy()
        if stage_filter != "all":
            fdf = fdf[fdf["diabetes_stage"] == stage_filter]
        if cluster_filter != "all":
            fdf = fdf[fdf["cluster"] == cluster_filter]
        if feature not in fdf.columns:
            return go.Figure()

        label = feature.replace("_", " ").title()
        color_seq = CLUSTER_COLORS

        if chart_type == "histogram":
            fig = px.histogram(fdf, x=feature, color="cluster", nbins=30,
                               title=f"Distribution of {label}",
                               color_discrete_sequence=color_seq, barmode="overlay", opacity=0.7)
        elif chart_type == "box":
            fig = px.box(fdf, x="cluster", y=feature,
                         title=f"{label} by Cluster",
                         color="cluster", color_discrete_sequence=color_seq)
        else:
            fig = px.scatter(fdf, x=feature, y="diabetes_risk_score",
                             color="cluster", title=f"{label} vs Risk Score",
                             color_discrete_sequence=color_seq, opacity=0.6)

        fig.update_layout(**PLOT_LAYOUT, hovermode="x unified")
        return fig

    # Risk score distribution
    @app.callback(
        Output("risk-score-graph", "figure"),
        [Input("cluster-filter", "value"), Input("diabetes-stage-filter", "value")],
    )
    def update_risk_score_graph(cluster_filter, stage_filter):
        fdf = df.copy()
        if stage_filter != "all":
            fdf = fdf[fdf["diabetes_stage"] == stage_filter]
        if cluster_filter != "all":
            fdf = fdf[fdf["cluster"] == cluster_filter]

        fig = px.histogram(fdf, x="diabetes_risk_score", color="diabetes_stage",
                           nbins=25, title="Risk Score Distribution by Stage",
                           color_discrete_map={"No Diabetes": C["teal"], "Type 2": C["red"]},
                           barmode="overlay", opacity=0.75)
        fig.update_layout(**PLOT_LAYOUT)
        return fig

    # Cluster composition donut 
    @app.callback(
        Output("cluster-composition-graph", "figure"),
        Input("diabetes-stage-filter", "value"),
    )
    def update_cluster_composition(stage_filter):
        fdf = df if stage_filter == "all" else df[df["diabetes_stage"] == stage_filter]
        counts = fdf["cluster"].value_counts().sort_index()
        fig = px.pie(values=counts.values,
                     names=[f"Cluster {int(i)}" for i in counts.index],
                     title="Patient Distribution across Clusters",
                     hole=0.45,
                     color_discrete_sequence=CLUSTER_COLORS)
        fig.update_traces(textinfo="label+percent", pull=[0.03] * len(counts))
        fig.update_layout(**PLOT_LAYOUT)
        return fig

    # Diabetes stage bar
    @app.callback(
        Output("diabetes-stage-graph", "figure"),
        Input("cluster-filter", "value"),
    )
    def update_diabetes_stage_graph(cluster_filter):
        fdf = df if cluster_filter == "all" else df[df["cluster"] == cluster_filter]
        counts = fdf["diabetes_stage"].value_counts()
        fig = px.bar(x=counts.index, y=counts.values,
                     title="Diabetes Stage Distribution",
                     labels={"x": "Stage", "y": "Count"},
                     color=counts.index,
                     color_discrete_map={"No Diabetes": C["teal"], "Type 2": C["red"]})
        fig.update_layout(**PLOT_LAYOUT, showlegend=False)
        return fig

    
    # TAB 2 – PREDICTIONS
    
    @app.callback(
        Output("prediction-output", "children"),
        Input("predict-button", "n_clicks"),
        [State("input-age",                "value"),
         State("input-bmi",                "value"),
         State("input-glucose-fasting",    "value"),
         State("input-glucose-postprandial","value"),
         State("input-systolic-bp",        "value"),
         State("input-diastolic-bp",       "value"),
         State("input-cholesterol",        "value"),
         State("input-hba1c",             "value"),
         State("input-activity",          "value")],
        prevent_initial_call=True,
    )
    def predict_risk(n_clicks, age, bmi, glucose_fasting, glucose_postprandial,
                     systolic_bp, diastolic_bp, cholesterol, hba1c, activity):

        vals = [age, bmi, glucose_fasting, glucose_postprandial,
                systolic_bp, diastolic_bp, cholesterol, hba1c, activity]

        if not all(v is not None for v in vals):
            return html.Div([
                html.Div("⚠️", style={"fontSize": "40px", "textAlign": "center"}),
                html.P("Please fill in all fields before predicting.",
                       style={"color": C["amber"], "textAlign": "center",
                              "fontWeight": "600", "marginTop": "10px"}),
            ])

        if xgb_model is None:
            return html.Div([
                html.P("❌ Model not loaded. Please check server logs.",
                       style={"color": C["red"], "textAlign": "center", "fontWeight": "600"}),
            ])

        try:
            input_data  = _build_full_input(vals)
            input_scaled = preprocessor.transform(input_data)
            prediction = xgb_model.predict(input_scaled)[0]
            proba = xgb_model.predict_proba(input_scaled)[0]

            cluster = kmeans_model.predict(input_scaled)[0] if kmeans_model else "N/A"

            is_high = prediction == 1
            risk_pct = proba[1] * 100
            conf_pct = risk_pct if is_high else proba[0] * 100

            bar_color = C["red"] if is_high else C["teal"]
            icon  = "🔴" if is_high else "🟢"
            label = "HIGH RISK" if is_high else "LOW RISK"

            return html.Div([
                # Risk badge
                html.Div([
                    html.Span(icon, style={"fontSize": "48px"}),
                    html.H4(label, style={"color": bar_color, "fontWeight": "800",
                                          "margin": "8px 0 4px",
                                          "fontFamily": "Nunito Sans, sans-serif"}),
                    html.P(f"Diabetes Probability: {risk_pct:.1f}%",
                           style={"color": C["slate"], "fontWeight": "600", "margin": "0"}),
                ], style={"textAlign": "center", "padding": "16px 0 12px"}),

                # Confidence bar
                html.Div([
                    html.Div(style={
                        "height": "10px", "borderRadius": "6px",
                        "background": f"linear-gradient(90deg, {bar_color} {risk_pct:.0f}%, #E2E8F0 {risk_pct:.0f}%)",
                    }),
                    dbc.Row([
                        dbc.Col(html.Small("0%", style={"color": "#A0AEC0", "fontSize": "11px"})),
                        dbc.Col(html.Small("100%", style={"color": "#A0AEC0", "fontSize": "11px",
                                                           "textAlign": "right"})),
                    ]),
                ], style={"margin": "0 16px 12px"}),

                html.Hr(style={"margin": "8px 16px"}),

                # Stats row
                dbc.Row([
                    dbc.Col([
                        html.Small("Confidence", style={"color": C["slate"], "fontSize": "11px",
                                                         "fontWeight": "700", "textTransform": "uppercase"}),
                        html.Div(f"{conf_pct:.1f}%", style={"color": C["navy"], "fontWeight": "800",
                                                              "fontSize": "20px"}),
                    ], style={"textAlign": "center"}),
                    dbc.Col([
                        html.Small("Risk Score", style={"color": C["slate"], "fontSize": "11px",
                                                         "fontWeight": "700", "textTransform": "uppercase"}),
                        html.Div(f"{proba[1]:.3f}", style={"color": C["navy"], "fontWeight": "800",
                                                            "fontSize": "20px"}),
                    ], style={"textAlign": "center"}),
                    dbc.Col([
                        html.Small("Cluster", style={"color": C["slate"], "fontSize": "11px",
                                                      "fontWeight": "700", "textTransform": "uppercase"}),
                        html.Div(str(cluster), style={"color": C["navy"], "fontWeight": "800",
                                                       "fontSize": "20px"}),
                    ], style={"textAlign": "center"}),
                ], style={"padding": "8px 16px 16px"}),
            ])

        except Exception as e:
            return html.Div([
                html.P(f"❌ Prediction error: {str(e)}",
                       style={"color": C["red"], "fontWeight": "600", "textAlign": "center"}),
            ])

    # Feature importance char
    @app.callback(
        Output("feature-importance-graph", "figure"),
        Input("predict-button", "n_clicks"),
        prevent_initial_call=True,
    )
    def update_feature_importance(n_clicks):
        if xgb_model is None:
            fig = go.Figure()
            fig.add_annotation(text="Model not loaded", showarrow=False)
            return fig
        try:
            imp = xgb_model.feature_importances_

            # Use real preprocessor feature names if available, else fall back to indices
            if preprocessor is not None:
                try:
                    feat_labels = list(preprocessor.get_feature_names_out())
                except Exception:
                    feat_labels = [f"feature_{i}" for i in range(len(imp))]
            else:
                feat_labels = [f"feature_{i}" for i in range(len(imp))]

            imp_df = (
                pd.DataFrame({"Feature": feat_labels, "Importance": imp})
                .sort_values("Importance")
                .tail(15)   # show top 15 to keep the chart readable
            )
            # Tidy up sklearn's "num__" / "cat__" prefixes for display
            imp_df["Feature"] = imp_df["Feature"].str.replace(r"^(num|cat)__", "", regex=True).str.replace("_", " ").str.title()
            fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                         title="XGBoost Feature Importance",
                         color="Importance",
                         color_continuous_scale=[[0, "#A8D5CF"], [1, C["teal"]]])
            fig.update_coloraxes(showscale=False)
            fig.update_layout(**PLOT_LAYOUT, height=320)
            return fig
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f"Error: {e}", showarrow=False)
            return fig

    # SHAP summary placeholder 
    @app.callback(
        Output("shap-summary", "children"),
        Input("predict-button", "n_clicks"),
        prevent_initial_call=True,
    )
    def update_shap_summary(n_clicks):
        return html.Div([
            html.H6("SHAP Summary", style={"color": C["navy"], "fontWeight": "700"}),
            html.P("SHAP (SHapley Additive Explanations) quantifies each feature's "
                   "contribution to the model's prediction for this patient.",
                   style={"color": C["slate"], "fontSize": "13px"}),
            html.P("Positive SHAP → increases risk probability.",
                   style={"color": C["red"], "fontSize": "13px", "fontWeight": "600"}),
            html.P("Negative SHAP → decreases risk probability.",
                   style={"color": C["teal"], "fontSize": "13px", "fontWeight": "600"}),
        ], style={"padding": "8px"})

    # SHAP force plot
    @app.callback(
        Output("shap-force-plot", "children"),
        Input("predict-button", "n_clicks"),
        [State("input-age",                "value"),
         State("input-bmi",                "value"),
         State("input-glucose-fasting",    "value"),
         State("input-glucose-postprandial","value"),
         State("input-systolic-bp",        "value"),
         State("input-diastolic-bp",       "value"),
         State("input-cholesterol",        "value"),
         State("input-hba1c",             "value"),
         State("input-activity",          "value")],
        prevent_initial_call=True,
    )
    def update_shap_force(n_clicks, age, bmi, glucose_fasting, glucose_postprandial,
                          systolic_bp, diastolic_bp, cholesterol, hba1c, activity):
        vals = [age, bmi, glucose_fasting, glucose_postprandial,
                systolic_bp, diastolic_bp, cholesterol, hba1c, activity]
        if not all(v is not None for v in vals):
            return html.P("Complete all fields to see SHAP analysis.",
                          style={"color": C["slate"], "textAlign": "center", "fontSize": "13px"})
        if xgb_model is None or preprocessor is None:
            return html.P("Model not loaded.", style={"color": C["red"], "textAlign": "center"})

        try:
            import shap
            input_data   = _build_full_input(vals)
            input_scaled = preprocessor.transform(input_data)
            explainer  = shap.TreeExplainer(xgb_model)
            shap_vals  = explainer.shap_values(input_scaled)
            base_val   = explainer.expected_value
            pred_val   = xgb_model.predict_proba(input_scaled)[0][1]

            rows = sorted(zip(FEATURE_NAMES, shap_vals[0]), key=lambda x: abs(x[1]), reverse=True)

            bars = []
            for fname, sv in rows:
                color = C["red"] if sv > 0 else C["teal"]
                pct   = min(abs(sv) * 600, 100)
                bars.append(html.Div([
                    html.Div(fname, style={"fontSize": "12px", "color": C["slate"],
                                           "fontWeight": "600", "marginBottom": "2px"}),
                    html.Div([
                        html.Div(style={"width": f"{pct:.0f}%", "height": "8px",
                                        "background": color, "borderRadius": "4px"}),
                    ], style={"background": "#E2E8F0", "borderRadius": "4px",
                               "marginBottom": "2px"}),
                    html.Div(f"{sv:+.4f}", style={"fontSize": "11px", "color": color,
                                                   "fontWeight": "700"}),
                ], style={"marginBottom": "10px"}))

            return html.Div([
                html.Div([
                    html.Span(f"Base: {base_val:.2f}",
                              style={"fontSize": "12px", "color": C["slate"], "marginRight": "16px"}),
                    html.Span(f"Prediction: {pred_val:.3f}",
                              style={"fontSize": "12px", "color": C["navy"], "fontWeight": "700"}),
                ], style={"marginBottom": "14px"}),
                *bars,
            ], style={"padding": "4px"})

        except Exception as e:
            return html.Div([
                html.P(f"SHAP: {str(e)}", style={"color": C["slate"], "fontSize": "12px"}),
            ])

    
    # TAB 3 – CLUSTERING
    
    @app.callback(
        Output("cluster-scatter-graph", "figure"),
        Input("main-tabs", "value"),
    )
    def update_cluster_scatter(tab):
        if df.empty:
            return go.Figure()
        try:
            x_col = "glucose_fasting" if "glucose_fasting" in df.columns else df.columns[0]
            y_col = "bmi"             if "bmi"             in df.columns else df.columns[1]
            fig = px.scatter(
                df, x=x_col, y=y_col, color=df["cluster"].astype(str),
                title="K-Means Cluster Visualisation (Glucose vs BMI)",
                labels={x_col: "Glucose (Fasting)", y_col: "BMI"},
                color_discrete_sequence=CLUSTER_COLORS,
                opacity=0.65, size_max=6,
            )
            fig.update_traces(marker=dict(size=5))
            fig.update_layout(**PLOT_LAYOUT, legend_title="Cluster")
            return fig
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f"Error: {e}", showarrow=False)
            return fig

    @app.callback(
        Output("cluster-profiles", "children"),
        Input("main-tabs", "value"),
    )
    def update_cluster_profiles(tab):
        if df.empty:
            return html.P("No data available.", style={"color": C["slate"]})

        profiles = []
        cluster_descs = {
            0: ("Low Risk",    C["teal"],  "Younger patients with healthy BMI, normal glucose and high activity."),
            1: ("Medium Risk", C["amber"], "Middle-aged with borderline glucose and moderate BMI; sedentary lifestyle."),
            2: ("High Risk",   C["red"],   "Older patients with elevated glucose, high BMI and HbA1c — most diabetic cases."),
        }

        for c in sorted(df["cluster"].unique()):
            ci = int(c)
            label, color, desc = cluster_descs.get(ci, (f"Cluster {ci}", C["navy"], ""))
            n = len(df[df["cluster"] == c])
            profiles.append(html.Div([
                html.Div([
                    html.Span(f"Cluster {ci}", style={"fontWeight": "800", "color": color,
                                                       "fontSize": "14px", "marginRight": "8px"}),
                    dbc.Badge(label, pill=True,
                              style={"background": color, "color": "#fff", "fontSize": "11px"}),
                ], style={"marginBottom": "4px"}),
                html.P(desc, style={"color": C["slate"], "fontSize": "12px", "marginBottom": "4px"}),
                html.Small(f"n = {n} patients", style={"color": "#A0AEC0"}),
                html.Hr(style={"margin": "10px 0"}),
            ]))

        return html.Div(profiles, style={"padding": "4px"})

    @app.callback(
        Output("cluster-heatmap", "figure"),
        Input("main-tabs", "value"),
    )
    def update_cluster_heatmap(tab):
        if df.empty:
            return go.Figure()
        try:
            feat_cols = [c for c in INPUT_COLS if c in df.columns]
            cluster_means = df.groupby("cluster")[feat_cols].mean()
            # Normalise 0-1 per feature for readability
            normed = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min() + 1e-9)
            nice_names = [c.replace("_", " ").title() for c in feat_cols]

            fig = go.Figure(data=go.Heatmap(
                z=normed.values,
                x=nice_names,
                y=[f"Cluster {int(i)}" for i in cluster_means.index],
                colorscale=[[0, "#EDF2F7"], [0.5, "#A8D5CF"], [1, C["navy"]]],
                text=[[f"{v:.2f}" for v in row] for row in cluster_means.values],
                texttemplate="%{text}",
                hoverongaps=False,
            ))
            fig.update_layout(**PLOT_LAYOUT, title="Mean Feature Values per Cluster (normalised)",
                              height=220, xaxis_tickangle=-30)
            return fig
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f"Error: {e}", showarrow=False)
            return fig