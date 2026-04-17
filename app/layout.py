# TASK 6.1: Diabetes Risk Dashboard - Enhanced Multi-Tab Layout
from dash import html, dcc
import dash_bootstrap_components as dbc

# Colour palette 
C = {
    "navy":   "#0066FF",
    "teal":   "#1A9E8F",
    "red":    "#E84855",
    "amber":  "#F5A623",
    "slate":  "#4A5568",
    "light":  "#F7FAFC",
    "white":  "#FFFFFF",
    "border": "#E2E8F0",
}

DEBUG_DEFAULT_INPUTS = {
    "input-age": 24,
    "input-bmi": 22,
    "input-glucose-fasting": 90,
    "input-glucose-postprandial": 120,
    "input-systolic-bp": 110,
    "input-diastolic-bp": 70,
    "input-cholesterol": 170,
    "input-hba1c": 5.2,
    "input-activity": 180,
}
# Shared style helpers
CARD = {
    "borderRadius": "14px",
    "boxShadow": "0 2px 12px rgba(13,43,69,.10)",
    "border": f"1px solid {C['border']}",
    "background": C["white"],
}

HEADER_STYLE = {
    "background": f"linear-gradient(135deg, {C['navy']} 0%, #0044CC 100%)",
    "padding": "28px 40px 24px",
    "borderBottom": f"3px solid {C['teal']}",
    "marginBottom": "0",
}

TAB_STYLE = {
    "fontFamily": "'Nunito Sans', sans-serif",
    "fontWeight": "600",
    "fontSize": "14px",
    "color": C["slate"],
    "background": "#EDF2F7",
    "border": "none",
    "borderBottom": f"2px solid {C['border']}",
    "padding": "12px 24px",
}
TAB_SELECTED = {
    **TAB_STYLE,
    "color": C["teal"],
    "background": C["white"],
    "borderBottom": f"3px solid {C['teal']}",
}


# Helper: stat card
def stat_card(title, metric_id, color, icon):
    return dbc.Col(
        dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.Span(icon, style={"fontSize": "28px"}),
                    html.Div([
                        html.P(title, className="mb-0",
                               style={"fontSize": "11px", "fontWeight": "700",
                                      "letterSpacing": "1px", "textTransform": "uppercase",
                                      "color": C["slate"]}),
                        html.H3(id=metric_id, children="—",
                                style={"color": color, "fontWeight": "800",
                                       "margin": "0", "fontFamily": "'Nunito Sans', sans-serif"}),
                    ], style={"marginLeft": "14px"}),
                ], style={"display": "flex", "alignItems": "center"}),
            ])
        ], style={**CARD, "borderLeft": f"4px solid {color}"}),
        xs=12, sm=6, lg=3, className="mb-3"
    )


# Helper: section heading 
def section_heading(text, sub=None):
    children = [html.H5(text, style={"color": C["navy"], "fontWeight": "700",
                                      "marginBottom": "2px", "fontFamily": "'Nunito Sans', sans-serif"})]
    if sub:
        children.append(html.P(sub, style={"color": C["slate"], "fontSize": "13px", "marginBottom": "0"}))
    return html.Div(children, className="mb-3")


# Helper: labelled input
def labelled_input(label, input_id, placeholder, min_val, max_val,
                   step=1, tooltip_text=None, unit="", default_value=None):
    tip = []
    if tooltip_text:
        tip = [
            html.Span(" ⓘ", id=f"tip-{input_id}", style={"color": C["teal"], "cursor": "pointer", "fontSize": "13px"}),
            dbc.Tooltip(tooltip_text, target=f"tip-{input_id}", placement="top"),
        ]
    return dbc.Col([
        html.Label([label, *tip], className="fw-semibold mb-1",
                   style={"fontSize": "13px", "color": C["slate"]}),
        dbc.InputGroup([
            dcc.Input(
                
                id=input_id,
                type="number",
                placeholder=placeholder,
                value=default_value,
                min=min_val,
                max=max_val,
                step=step,
                debounce=True,
                style={"borderRadius": "8px", "border": f"1px solid {C['border']}",
                       "padding": "8px 12px", "fontSize": "14px", "width": "100%",
                       "outline": "none"},
            ),
            *([ dbc.InputGroupText(unit, style={"fontSize": "12px", "color": C["slate"],
                                                 "background": "#EDF2F7", "border": f"1px solid {C['border']}"}) ] if unit else []),
        ], style={"borderRadius": "8px"}),
    ], xs=12, sm=6, lg=4, className="mb-3")


#  TAB 1: Overview / ML Insights 
def overview_tab():
    return dbc.Tab(label="📊 ML Insights", tab_id="tab-overview",
                   label_style=TAB_STYLE, active_label_style=TAB_SELECTED,
                   children=[
        html.Div([
            # Stat cards
            dbc.Row([
                stat_card("Total Patients",    "metric-total-patients", C["teal"],  "👥"),
                stat_card("Diabetes Cases",    "metric-diabetes-cases", C["red"],   "🩸"),
                stat_card("Avg Risk Score",    "metric-avg-risk",       C["amber"], "⚠️"),
                stat_card("Patient Clusters",  "metric-clusters",       C["navy"],  "🔵"),
            ], className="mb-2"),

            # Filters row
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Cluster", className="fw-semibold mb-1",
                                       style={"fontSize": "12px", "color": C["slate"]}),
                            dcc.Dropdown(id="cluster-filter",
                                         options=[{"label": "All Clusters", "value": "all"}],
                                         value="all", clearable=False,
                                         style={"fontSize": "13px"}),
                        ], xs=12, sm=6, lg=3),
                        dbc.Col([
                            html.Label("Risk Stage", className="fw-semibold mb-1",
                                       style={"fontSize": "12px", "color": C["slate"]}),
                            dcc.Dropdown(id="diabetes-stage-filter",
                                         options=[
                                             {"label": "All Stages",  "value": "all"},
                                             {"label": "No Diabetes", "value": "No Diabetes"},
                                             {"label": "Type 2",      "value": "Type 2"},
                                         ],
                                         value="all", clearable=False,
                                         style={"fontSize": "13px"}),
                        ], xs=12, sm=6, lg=3),
                        dbc.Col([
                            html.Label("Feature", className="fw-semibold mb-1",
                                       style={"fontSize": "12px", "color": C["slate"]}),
                            dcc.Dropdown(id="feature-dropdown",
                                         options=[
                                             {"label": "Age",                   "value": "age"},
                                             {"label": "BMI",                   "value": "bmi"},
                                             {"label": "Glucose (Fasting)",     "value": "glucose_fasting"},
                                             {"label": "Glucose (Postprandial)","value": "glucose_postprandial"},
                                             {"label": "Systolic BP",           "value": "systolic_bp"},
                                             {"label": "Diastolic BP",          "value": "diastolic_bp"},
                                             {"label": "Cholesterol",           "value": "cholesterol_total"},
                                             {"label": "HbA1c",                 "value": "hba1c"},
                                             {"label": "Physical Activity",     "value": "physical_activity_minutes_per_week"},
                                         ],
                                         value="age", clearable=False,
                                         style={"fontSize": "13px"}),
                        ], xs=12, sm=6, lg=3),
                        dbc.Col([
                            html.Label("Chart Type", className="fw-semibold mb-1",
                                       style={"fontSize": "12px", "color": C["slate"]}),
                            dcc.Dropdown(id="chart-type-dropdown",
                                         options=[
                                             {"label": "Histogram", "value": "histogram"},
                                             {"label": "Box Plot",  "value": "box"},
                                             {"label": "Scatter",   "value": "scatter"},
                                         ],
                                         value="histogram", clearable=False,
                                         style={"fontSize": "13px"}),
                        ], xs=12, sm=6, lg=3),
                    ])
                ])
            ], style={**CARD, "marginBottom": "20px"}),

            # Main charts row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.Span("Feature Distribution by Cluster",
                                                  style={"fontWeight": "700", "color": C["navy"], "fontSize": "14px"}),
                                       style={"background": "#F7FAFC", "borderBottom": f"1px solid {C['border']}"}),
                        dbc.CardBody([dcc.Graph(id="main-graph", config={"displayModeBar": False})]),
                    ], style=CARD),
                ], xs=12, lg=7, className="mb-4"),

                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.Span("Risk Score Distribution",
                                                  style={"fontWeight": "700", "color": C["navy"], "fontSize": "14px"}),
                                       style={"background": "#F7FAFC", "borderBottom": f"1px solid {C['border']}"}),
                        dbc.CardBody([dcc.Graph(id="risk-score-graph", config={"displayModeBar": False})]),
                    ], style=CARD),
                ], xs=12, lg=5, className="mb-4"),
            ]),

            # Secondary charts
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.Span("Cluster Composition",
                                                  style={"fontWeight": "700", "color": C["navy"], "fontSize": "14px"}),
                                       style={"background": "#F7FAFC", "borderBottom": f"1px solid {C['border']}"}),
                        dbc.CardBody([dcc.Graph(id="cluster-composition-graph", config={"displayModeBar": False})]),
                    ], style=CARD),
                ], xs=12, lg=5, className="mb-4"),

                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.Span("Diabetes Stage Distribution",
                                                  style={"fontWeight": "700", "color": C["navy"], "fontSize": "14px"}),
                                       style={"background": "#F7FAFC", "borderBottom": f"1px solid {C['border']}"}),
                        dbc.CardBody([dcc.Graph(id="diabetes-stage-graph", config={"displayModeBar": False})]),
                    ], style=CARD),
                ], xs=12, lg=7, className="mb-4"),
            ]),

            # Model comparison table
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.Span("Model Performance Comparison",
                                                  style={"fontWeight": "700", "color": C["navy"], "fontSize": "14px"}),
                                       style={"background": "#F7FAFC", "borderBottom": f"1px solid {C['border']}"}),
                        dbc.CardBody([
                            html.Div([
                                html.Table([
                                    html.Thead(html.Tr([
                                        html.Th("Model", style={"width": "25%"}),
                                        html.Th("Accuracy"),
                                        html.Th("Precision"),
                                        html.Th("Recall"),
                                        html.Th("F1-Score"),
                                    ], style={"background": C["navy"], "color": "#fff",
                                              "fontSize": "13px", "textAlign": "center"})),
                                    html.Tbody([
                                        html.Tr([
                                            html.Td("Decision Tree", style={"fontWeight": "600"}),
                                            html.Td("84.2%"), html.Td("83.1%"),
                                            html.Td("82.7%"), html.Td("82.9%"),
                                        ], style={"textAlign": "center", "background": "#F7FAFC"}),
                                        html.Tr([
                                            html.Td("Random Forest", style={"fontWeight": "600"}),
                                            html.Td("91.5%"), html.Td("90.8%"),
                                            html.Td("91.2%"), html.Td("91.0%"),
                                        ], style={"textAlign": "center"}),
                                        html.Tr([
                                            html.Td([
                                                "XGBoost ",
                                                dbc.Badge("Best", color="success", pill=True,
                                                          style={"fontSize": "10px"}),
                                            ], style={"fontWeight": "700"}),
                                            html.Td("94.3%", style={"color": C["teal"], "fontWeight": "700"}),
                                            html.Td("93.7%", style={"color": C["teal"], "fontWeight": "700"}),
                                            html.Td("94.1%", style={"color": C["teal"], "fontWeight": "700"}),
                                            html.Td("93.9%", style={"color": C["teal"], "fontWeight": "700"}),
                                        ], style={"textAlign": "center", "background": "#F0FFF4"}),
                                    ]),
                                ], style={"width": "100%", "borderCollapse": "collapse",
                                          "fontSize": "14px", "border": f"1px solid {C['border']}"}),
                            ])
                        ]),
                    ], style=CARD),
                ], xs=12, className="mb-4"),
            ]),

        ], style={"padding": "24px 8px"}),
    ])


# TAB 2: Predictions
def predictions_tab():
    return dbc.Tab(label="🤖 Predictions", tab_id="tab-predictions",
                   label_style=TAB_STYLE, active_label_style=TAB_SELECTED,
                   children=[
        html.Div([
            dbc.Row([
                # Input form 
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(
                            html.Span("Patient Data Input",
                                      style={"fontWeight": "700", "color": C["navy"], "fontSize": "15px"}),
                            style={"background": "#F7FAFC", "borderBottom": f"1px solid {C['border']}"}
                        ),
                        dbc.CardBody([
                            html.P("Enter the patient's clinical measurements below and click Predict.",
                                   style={"color": C["slate"], "fontSize": "13px", "marginBottom": "20px"}),

                            dbc.Row([
                                labelled_input("Age", "input-age", "e.g. 45", 1, 120,
                                               tooltip_text="Patient's age in years", unit="yrs",
                                               default_value=DEBUG_DEFAULT_INPUTS["input-age"]),
                                labelled_input("BMI", "input-bmi", "e.g. 27.5", 10, 60, step=0.1,
                                               tooltip_text="Body Mass Index (weight kg / height m²)", unit="kg/m²",
                                               default_value=DEBUG_DEFAULT_INPUTS["input-bmi"]),
                                labelled_input("Glucose (Fasting)", "input-glucose-fasting", "e.g. 110", 50, 500,
                                               tooltip_text="Fasting blood glucose level", unit="mg/dL",
                                               default_value=DEBUG_DEFAULT_INPUTS["input-glucose-fasting"]),
                            ]),
                            dbc.Row([
                                labelled_input("Glucose (Postprandial)", "input-glucose-postprandial",
                                               "e.g. 140", 50, 600,
                                               tooltip_text="Blood glucose 2 hours after eating", unit="mg/dL",
                                               default_value=DEBUG_DEFAULT_INPUTS["input-glucose-postprandial"]),
                                labelled_input("Systolic BP", "input-systolic-bp", "e.g. 120", 80, 200,
                                               tooltip_text="Top number of blood pressure reading", unit="mmHg",
                                               default_value=DEBUG_DEFAULT_INPUTS["input-systolic-bp"]),
                                labelled_input("Diastolic BP", "input-diastolic-bp", "e.g. 80", 40, 150,
                                               tooltip_text="Bottom number of blood pressure reading", unit="mmHg",
                                               default_value=DEBUG_DEFAULT_INPUTS["input-diastolic-bp"]),
                            ]),
                            dbc.Row([
                                labelled_input("Cholesterol (Total)", "input-cholesterol", "e.g. 190", 100, 400,
                                               tooltip_text="Total blood cholesterol level", unit="mg/dL",
                                               default_value=DEBUG_DEFAULT_INPUTS["input-cholesterol"]),
                                labelled_input("HbA1c", "input-hba1c", "e.g. 6.5", 4, 15, step=0.1,
                                               tooltip_text="3-month average blood sugar level", unit="%",
                                               default_value=DEBUG_DEFAULT_INPUTS["input-hba1c"]),
                                labelled_input("Physical Activity", "input-activity", "e.g. 150", 0, 1000,
                                               tooltip_text="Minutes of moderate exercise per week", unit="min/wk",
                                               default_value=DEBUG_DEFAULT_INPUTS["input-activity"]),
                            ]),

                            dbc.Row([
                                dbc.Col([
                                    dbc.Button([
                                        dbc.Spinner(size="sm", id="predict-spinner", spinner_style={"display": "none"}),
                                        " ⚡  Predict Diabetes Risk"
                                    ],
                                        id="predict-button",
                                        n_clicks=0,
                                        color="primary",
                                        className="w-100 fw-bold",
                                        style={"background": C["teal"], "border": "none",
                                               "borderRadius": "10px", "padding": "12px",
                                               "fontSize": "15px", "letterSpacing": "0.5px"},
                                    ),
                                ], xs=12, sm=8, lg=6, className="mx-auto"),
                            ]),
                        ]),
                    ], style=CARD),
                ], xs=12, lg=7, className="mb-4"),

                # Results panel 
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(
                            html.Span("Prediction Results",
                                      style={"fontWeight": "700", "color": C["navy"], "fontSize": "15px"}),
                            style={"background": "#F7FAFC", "borderBottom": f"1px solid {C['border']}"}
                        ),
                        dbc.CardBody([
                            html.Div([
                                html.Div("🔬", style={"fontSize": "52px", "textAlign": "center",
                                                      "marginBottom": "10px"}),
                                html.P("Fill in the patient data and click Predict to see results.",
                                       style={"color": C["slate"], "textAlign": "center",
                                              "fontSize": "14px"}),
                            ], id="prediction-output"),
                        ], style={"minHeight": "220px", "display": "flex",
                                  "alignItems": "center", "justifyContent": "center"}),
                    ], style=CARD),

                    # Feature importance after prediction
                    dbc.Card([
                        dbc.CardHeader(
                            html.Span("Feature Importance (XGBoost)",
                                      style={"fontWeight": "700", "color": C["navy"], "fontSize": "14px"}),
                            style={"background": "#F7FAFC", "borderBottom": f"1px solid {C['border']}",
                                   "marginTop": "16px"}
                        ),
                        dbc.CardBody([dcc.Graph(id="feature-importance-graph",
                                                config={"displayModeBar": False})]),
                    ], style={**CARD, "marginTop": "16px"}),
                ], xs=12, lg=5, className="mb-4"),
            ]),

            # SHAP row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(
                            html.Span("SHAP Explanation – Individual Prediction",
                                      style={"fontWeight": "700", "color": C["navy"], "fontSize": "14px"}),
                            style={"background": "#F7FAFC", "borderBottom": f"1px solid {C['border']}"}
                        ),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([html.Div(id="shap-summary")], xs=12, lg=5),
                                dbc.Col([html.Div(id="shap-force-plot")], xs=12, lg=7),
                            ])
                        ]),
                    ], style=CARD),
                ], xs=12, className="mb-4"),
            ]),

        ], style={"padding": "24px 8px"}),
    ])


# TAB 3: Clustering & Segmentation 
def clustering_tab():
    return dbc.Tab(label="🔍 Clustering", tab_id="tab-clustering",
                   label_style=TAB_STYLE, active_label_style=TAB_SELECTED,
                   children=[
        html.Div([
            dbc.Row([
                dbc.Col([
                    section_heading("K-Means Patient Segmentation",
                                    "Unsupervised clustering groups patients into risk profiles based on their clinical measurements."),
                ], xs=12),
            ]),

            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.Span("Cluster Scatter Plot",
                                                  style={"fontWeight": "700", "color": C["navy"], "fontSize": "14px"}),
                                       style={"background": "#F7FAFC", "borderBottom": f"1px solid {C['border']}"}),
                        dbc.CardBody([dcc.Graph(id="cluster-scatter-graph", config={"displayModeBar": False})]),
                    ], style=CARD),
                ], xs=12, lg=8, className="mb-4"),

                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.Span("Cluster Profiles",
                                                  style={"fontWeight": "700", "color": C["navy"], "fontSize": "14px"}),
                                       style={"background": "#F7FAFC", "borderBottom": f"1px solid {C['border']}"}),
                        dbc.CardBody([html.Div(id="cluster-profiles")]),
                    ], style={**CARD, "height": "100%"}),
                ], xs=12, lg=4, className="mb-4"),
            ]),

            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.Span("Cluster Feature Heatmap",
                                                  style={"fontWeight": "700", "color": C["navy"], "fontSize": "14px"}),
                                       style={"background": "#F7FAFC", "borderBottom": f"1px solid {C['border']}"}),
                        dbc.CardBody([dcc.Graph(id="cluster-heatmap", config={"displayModeBar": False})]),
                    ], style=CARD),
                ], xs=12, className="mb-4"),
            ]),

        ], style={"padding": "24px 8px"}),
    ])


# TAB 4: Documentation
def documentation_tab():
    def info_card(icon, title, points):
        return dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Span(icon, style={"fontSize": "28px", "marginRight": "10px"}),
                        html.H6(title, style={"display": "inline", "fontWeight": "700",
                                              "color": C["navy"], "fontSize": "15px"}),
                    ], className="mb-3"),
                    html.Ul([
                        html.Li(p, style={"color": C["slate"], "fontSize": "13px",
                                          "marginBottom": "6px", "lineHeight": "1.5"})
                        for p in points
                    ], style={"paddingLeft": "18px"}),
                ])
            ], style={**CARD, "height": "100%"}),
        ], xs=12, lg=4, className="mb-4")

    return dbc.Tab(label="📄 Documentation", tab_id="tab-docs",
                   label_style=TAB_STYLE, active_label_style=TAB_SELECTED,
                   children=[
        html.Div([
            dbc.Row([
                dbc.Col([
                    section_heading("Project Documentation",
                                    "A concise overview of the dataset, preparation pipeline, and models used in this system."),
                ], xs=12),
            ]),

            dbc.Row([
                info_card("🗃️", "Dataset", [
                    "Synthetic clinical dataset simulating real-world patient health records.",
                    "Features include demographics (age, BMI), blood work (glucose, HbA1c, cholesterol), and lifestyle metrics.",
                    "Target variable: diagnosed_diabetes (binary) and diabetes_stage (categorical).",
                    "Size: several thousand rows — balanced class distribution applied.",
                ]),
                info_card("⚙️", "Data Preparation", [
                    "Missing values handled via median imputation for numeric features.",
                    "Outlier removal using IQR clipping on key clinical features.",
                    "StandardScaler applied to normalise feature distributions before modelling.",
                    "Label encoding used for categorical targets; processed data saved as a reusable pipeline.",
                ]),
                info_card("🤖", "Models Used", [
                    "Decision Tree — interpretable baseline; prone to overfitting.",
                    "Random Forest — ensemble of trees; high accuracy with less overfitting.",
                    "XGBoost — gradient boosting; best overall performance (≈94% accuracy).",
                    "K-Means Clustering (k=3) — unsupervised patient segmentation into risk groups.",
                    "SHAP (SHapley Additive Explanations) — used to explain individual predictions.",
                ]),
            ]),

            # Workflow diagram (static)
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("End-to-End Workflow",
                                    style={"color": C["navy"], "fontWeight": "700", "marginBottom": "16px"}),
                            html.Div([
                                html.Div([
                                    html.Div(step, style={
                                        "background": C["teal"] if i % 2 == 0 else C["navy"],
                                        "color": "#fff", "borderRadius": "10px",
                                        "padding": "10px 20px", "fontSize": "13px",
                                        "fontWeight": "600", "textAlign": "center",
                                        "minWidth": "140px",
                                    })
                                    for i, step in enumerate([
                                        "Raw Data", "Clean & Scale", "Feature Engineering",
                                        "Model Training", "Evaluation", "Dashboard",
                                    ])
                                ], style={"display": "flex", "gap": "6px",
                                          "alignItems": "center", "justifyContent": "center",
                                          "flexWrap": "wrap"}),
                            ]),
                        ])
                    ], style=CARD),
                ], xs=12, className="mb-4"),
            ]),

            # Metric definitions
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.Span("Key Metric Definitions",
                                                  style={"fontWeight": "700", "color": C["navy"], "fontSize": "14px"}),
                                       style={"background": "#F7FAFC", "borderBottom": f"1px solid {C['border']}"}),
                        dbc.CardBody([
                            html.Table([
                                html.Thead(html.Tr([
                                    html.Th("Metric"), html.Th("Formula / Meaning"),
                                ], style={"background": C["navy"], "color": "#fff", "fontSize": "13px"})),
                                html.Tbody([
                                    html.Tr([html.Td("Accuracy"),  html.Td("(TP + TN) / All — overall correct predictions")],
                                            style={"background": "#F7FAFC"}),
                                    html.Tr([html.Td("Precision"), html.Td("TP / (TP + FP) — of predicted positives, how many are correct")]),
                                    html.Tr([html.Td("Recall"),    html.Td("TP / (TP + FN) — of actual positives, how many were found")],
                                            style={"background": "#F7FAFC"}),
                                    html.Tr([html.Td("F1-Score"),  html.Td("Harmonic mean of Precision & Recall — balanced measure")]),
                                    html.Tr([html.Td("SHAP Value"), html.Td("Each feature's contribution to moving the prediction from the base value")],
                                            style={"background": "#F7FAFC"}),
                                ]),
                            ], style={"width": "100%", "fontSize": "13px",
                                      "borderCollapse": "collapse", "border": f"1px solid {C['border']}"}),
                        ]),
                    ], style=CARD),
                ], xs=12, className="mb-4"),
            ]),

        ], style={"padding": "24px 8px"}),
    ])


# Master Layout
def create_layout():
    return html.Div([

        # Google Font
        html.Link(rel="stylesheet",
                  href="https://fonts.googleapis.com/css2?family=Nunito+Sans:wght@400;600;700;800&display=swap"),

        # Header 
        html.Div([
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Span("🏥", style={"fontSize": "36px", "marginRight": "14px"}),
                            html.Div([
                                html.H1("Diabetes Risk Dashboard",
                                        style={"color": "#fff", "fontFamily": "'Nunito Sans', sans-serif",
                                               "fontWeight": "800", "fontSize": "26px", "margin": "0",
                                               "letterSpacing": "-0.5px"}),
                                html.P("Patient Segmentation & Risk Analysis System",
                                       style={"color": "#A8D5CF", "margin": "0",
                                              "fontSize": "13px", "fontWeight": "500"}),
                            ]),
                        ], style={"display": "flex", "alignItems": "center"}),
                    ], xs=12, md=8),
                    dbc.Col([
                        html.Div([
                            dbc.Badge("XGBoost Powered", color="light", text_color="dark",
                                      pill=True,
                                      style={"fontSize": "12px", "marginRight": "8px",
                                             "fontFamily": "'Nunito Sans', sans-serif"}),
                            dbc.Badge("K-Means Clustering", color="light", text_color="dark",
                                      pill=True,
                                      style={"fontSize": "12px",
                                             "fontFamily": "'Nunito Sans', sans-serif"}),
                        ], style={"textAlign": "right"}),
                    ], xs=12, md=4, className="d-none d-md-flex align-items-center justify-content-end"),
                ])
            ], fluid=True),
        ], style=HEADER_STYLE),

        # Tabs
        dbc.Container([
            dcc.Tabs(
                id="main-tabs",
                value="tab-overview",
                children=[
                    overview_tab(),
                    predictions_tab(),
                    clustering_tab(),
                    documentation_tab(),
                ],
                style={"marginTop": "0", "borderBottom": f"2px solid {C['border']}"},
                content_style={"background": C["light"]},
            ),
        ], fluid=True, style={"padding": "0"}),

        # Footer 
        html.Div([
            html.P("Diabetes Risk Segmentation & Decision Support System · Healthcare Analytics Dashboard",
                   style={"textAlign": "center", "color": "#A0AEC0", "fontSize": "12px",
                           "margin": "0", "fontFamily": "'Nunito Sans', sans-serif"}),
        ], style={"background": C["navy"], "padding": "16px", "marginTop": "8px"}),

    ], style={"fontFamily": "'Nunito Sans', sans-serif",
              "background": C["light"], "minHeight": "100vh"})
