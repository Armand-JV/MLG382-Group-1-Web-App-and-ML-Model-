from dash import html, dcc
import dash_bootstrap_components as dbc

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "260px",
    "padding": "2rem 1rem",
    "backgroundColor": "#0f172a",
    "color": "white",
    "overflowY": "auto",
    "zIndex": 1000,
}

CONTENT_STYLE = {
    "marginLeft": "270px",
    "padding": "2rem",
    "backgroundColor": "#f8fafc",
    "minHeight": "100vh",
}

CARD_STYLE = {
    "borderRadius": "12px",
    "border": "none",
    "boxShadow": "0 1px 3px rgba(0,0,0,0.08)",
}

NAV_LINK_STYLE = {
    "color": "#94a3b8",
    "padding": "10px 14px",
    "borderRadius": "8px",
    "marginBottom": "4px",
    "cursor": "pointer",
    "display": "block",
    "textDecoration": "none",
    "fontWeight": "500",
    "transition": "all 0.2s",
}


def stat_card(title, value_id, icon, color):
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Div([
                    html.P(title, className="text-muted mb-1",
                           style={"fontSize": "0.78rem", "fontWeight": "600", "letterSpacing": "0.05em", "textTransform": "uppercase"}),
                    html.H3(id=value_id, className="mb-0 fw-bold", style={"color": "#0f172a"}),
                ]),
                html.Div(icon, style={
                    "width": "48px", "height": "48px", "borderRadius": "12px",
                    "backgroundColor": color, "display": "flex",
                    "alignItems": "center", "justifyContent": "center",
                    "fontSize": "22px"
                })
            ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"})
        ])
    ], style=CARD_STYLE)


def create_layout():
    sidebar = html.Div([
        html.Div([
            html.Div("🩺", style={"fontSize": "28px", "marginBottom": "6px"}),
            html.H5("DiabetesIQ", className="fw-bold mb-0", style={"color": "white"}),
            html.P("Decision Support System", style={"color": "#64748b", "fontSize": "0.75rem"}),
        ], className="mb-4 pb-3", style={"borderBottom": "1px solid #1e293b"}),

        html.P("NAVIGATION", style={"color": "#475569", "fontSize": "0.65rem",
                                     "letterSpacing": "0.1em", "fontWeight": "700", "marginBottom": "8px"}),

        html.Div([
            html.A("📊  Overview", id="nav-overview", href="#", style=NAV_LINK_STYLE),
            html.A("🔬  Patient Segmentation", id="nav-cluster", href="#", style=NAV_LINK_STYLE),
            html.A("🤖  Risk Prediction", id="nav-predict", href="#", style=NAV_LINK_STYLE),
            html.A("📈  Model Performance", id="nav-model", href="#", style=NAV_LINK_STYLE),
            html.A("💡  SHAP Explainability", id="nav-shap", href="#", style=NAV_LINK_STYLE),
        ]),

        html.Hr(style={"borderColor": "#1e293b", "margin": "1.5rem 0"}),

        html.P("DATA FILTERS", style={"color": "#475569", "fontSize": "0.65rem",
                                       "letterSpacing": "0.1em", "fontWeight": "700", "marginBottom": "10px"}),

        html.Label("Cluster Filter", style={"color": "#94a3b8", "fontSize": "0.8rem"}),
        dcc.Dropdown(
            id="cluster-filter",
            options=[
                {"label": "All Clusters", "value": "all"},
                {"label": "Cluster 0 — Low Risk", "value": 0},
                {"label": "Cluster 1 — Moderate Risk", "value": 1},
                {"label": "Cluster 2 — High Risk", "value": 2},
            ],
            value="all",
            clearable=False,
            style={"marginBottom": "12px", "fontSize": "0.85rem"},
        ),

        html.Label("Feature (X-Axis)", style={"color": "#94a3b8", "fontSize": "0.8rem"}),
        dcc.Dropdown(
            id="feature-dropdown",
            options=[
                {"label": "Age", "value": "age"},
                {"label": "BMI", "value": "bmi"},
                {"label": "Fasting Glucose", "value": "glucose_fasting"},
                {"label": "Postprandial Glucose", "value": "glucose_postprandial"},
                {"label": "HbA1c", "value": "hba1c"},
                {"label": "Physical Activity", "value": "physical_activity_hours_per_week"},
                {"label": "Sleep Hours", "value": "sleep_hours_per_night"},
                {"label": "Stress Level", "value": "stress_level"},
            ],
            value="bmi",
            clearable=False,
            style={"marginBottom": "12px", "fontSize": "0.85rem"},
        ),

        html.Label("Chart Type", style={"color": "#94a3b8", "fontSize": "0.8rem"}),
        dcc.RadioItems(
            id="chart-type",
            options=[
                {"label": " Histogram", "value": "histogram"},
                {"label": " Box Plot", "value": "box"},
                {"label": " Scatter", "value": "scatter"},
                {"label": " Violin", "value": "violin"},
            ],
            value="histogram",
            labelStyle={"display": "block", "color": "#94a3b8", "fontSize": "0.82rem", "marginBottom": "4px"},
        ),
    ], style=SIDEBAR_STYLE)

    # ── KPI Row ──────────────────────────────────────────────────────────────
    kpi_row = dbc.Row([
        dbc.Col(stat_card("Total Patients",   "kpi-total",    "👥", "#dbeafe"), md=3),
        dbc.Col(stat_card("High Risk",        "kpi-highrisk", "🔴", "#fee2e2"), md=3),
        dbc.Col(stat_card("Avg BMI",          "kpi-avgbmi",   "⚖️", "#fef9c3"), md=3),
        dbc.Col(stat_card("Avg Fasting Glu.", "kpi-avggluc",  "🩸", "#dcfce7"), md=3),
    ], className="g-3 mb-4")

    # ── Main feature chart ───────────────────────────────────────────────────
    main_chart = dbc.Card([
        dbc.CardHeader(
            html.H6("Feature Distribution by Cluster", className="mb-0 fw-semibold"),
            style={"backgroundColor": "white", "border": "none", "paddingTop": "1rem"}
        ),
        dbc.CardBody(dcc.Graph(id="main-graph", config={"displayModeBar": False}))
    ], style=CARD_STYLE)

    # ── Cluster composition ──────────────────────────────────────────────────
    cluster_pie = dbc.Card([
        dbc.CardHeader(html.H6("Cluster Size Distribution", className="mb-0 fw-semibold"),
                       style={"backgroundColor": "white", "border": "none", "paddingTop": "1rem"}),
        dbc.CardBody(dcc.Graph(id="cluster-pie", config={"displayModeBar": False}))
    ], style=CARD_STYLE)

    # ── Diabetes stage bar ───────────────────────────────────────────────────
    stage_bar = dbc.Card([
        dbc.CardHeader(html.H6("Diabetes Stage per Cluster", className="mb-0 fw-semibold"),
                       style={"backgroundColor": "white", "border": "none", "paddingTop": "1rem"}),
        dbc.CardBody(dcc.Graph(id="stage-bar", config={"displayModeBar": False}))
    ], style=CARD_STYLE)

    # ── Scatter matrix ───────────────────────────────────────────────────────
    scatter_matrix = dbc.Card([
        dbc.CardHeader(html.H6("Multi-Feature Scatter Explorer", className="mb-0 fw-semibold"),
                       style={"backgroundColor": "white", "border": "none", "paddingTop": "1rem"}),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("X Axis", style={"fontSize": "0.8rem", "color": "#64748b"}),
                    dcc.Dropdown(id="scatter-x", value="bmi",
                                 options=[
                                     {"label": "BMI", "value": "bmi"},
                                     {"label": "Age", "value": "age"},
                                     {"label": "Fasting Glucose", "value": "glucose_fasting"},
                                     {"label": "HbA1c", "value": "hba1c"},
                                 ], clearable=False),
                ], md=4),
                dbc.Col([
                    html.Label("Y Axis", style={"fontSize": "0.8rem", "color": "#64748b"}),
                    dcc.Dropdown(id="scatter-y", value="hba1c",
                                 options=[
                                     {"label": "HbA1c", "value": "hba1c"},
                                     {"label": "Fasting Glucose", "value": "glucose_fasting"},
                                     {"label": "BMI", "value": "bmi"},
                                     {"label": "Age", "value": "age"},
                                 ], clearable=False),
                ], md=4),
            ], className="mb-3"),
            dcc.Graph(id="scatter-plot", config={"displayModeBar": False})
        ])
    ], style=CARD_STYLE)

    # ── Risk Prediction Form ─────────────────────────────────────────────────
    prediction_panel = dbc.Card([
        dbc.CardHeader(html.H6("🤖 Real-Time Risk Prediction", className="mb-0 fw-semibold"),
                       style={"backgroundColor": "white", "border": "none", "paddingTop": "1rem"}),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Age", style={"fontSize": "0.82rem", "color": "#475569"}),
                    dcc.Slider(id="input-age", min=18, max=90, step=1, value=45,
                               marks={18: "18", 45: "45", 90: "90"},
                               tooltip={"placement": "bottom", "always_visible": True}),
                ], md=6, className="mb-3"),
                dbc.Col([
                    html.Label("BMI", style={"fontSize": "0.82rem", "color": "#475569"}),
                    dcc.Slider(id="input-bmi", min=15, max=55, step=0.5, value=27,
                               marks={15: "15", 35: "35", 55: "55"},
                               tooltip={"placement": "bottom", "always_visible": True}),
                ], md=6, className="mb-3"),
                dbc.Col([
                    html.Label("Fasting Glucose (mg/dL)", style={"fontSize": "0.82rem", "color": "#475569"}),
                    dcc.Slider(id="input-glucose", min=60, max=300, step=1, value=100,
                               marks={60: "60", 180: "180", 300: "300"},
                               tooltip={"placement": "bottom", "always_visible": True}),
                ], md=6, className="mb-3"),
                dbc.Col([
                    html.Label("HbA1c (%)", style={"fontSize": "0.82rem", "color": "#475569"}),
                    dcc.Slider(id="input-hba1c", min=4, max=14, step=0.1, value=5.5,
                               marks={4: "4", 7: "7", 14: "14"},
                               tooltip={"placement": "bottom", "always_visible": True}),
                ], md=6, className="mb-3"),
                dbc.Col([
                    html.Label("Physical Activity (hrs/week)", style={"fontSize": "0.82rem", "color": "#475569"}),
                    dcc.Slider(id="input-activity", min=0, max=20, step=0.5, value=5,
                               marks={0: "0", 10: "10", 20: "20"},
                               tooltip={"placement": "bottom", "always_visible": True}),
                ], md=6, className="mb-3"),
                dbc.Col([
                    html.Label("Stress Level (1–10)", style={"fontSize": "0.82rem", "color": "#475569"}),
                    dcc.Slider(id="input-stress", min=1, max=10, step=1, value=4,
                               marks={1: "1", 5: "5", 10: "10"},
                               tooltip={"placement": "bottom", "always_visible": True}),
                ], md=6, className="mb-3"),
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("Model", style={"fontSize": "0.82rem", "color": "#475569"}),
                    dcc.Dropdown(
                        id="model-selector",
                        options=[
                            {"label": "Random Forest", "value": "rf"},
                            {"label": "XGBoost", "value": "xgb"},
                        ],
                        value="rf",
                        clearable=False,
                    ),
                ], md=4),
                dbc.Col([
                    dbc.Button("Run Prediction", id="predict-btn", color="primary",
                               className="mt-4 w-100", style={"fontWeight": "600"}),
                ], md=4),
            ], className="mb-3"),
            html.Div(id="prediction-output"),
        ])
    ], style=CARD_STYLE)

    # ── Cluster Summary Table ────────────────────────────────────────────────
    cluster_table = dbc.Card([
        dbc.CardHeader(html.H6("📋 Cluster Profile Summary", className="mb-0 fw-semibold"),
                       style={"backgroundColor": "white", "border": "none", "paddingTop": "1rem"}),
        dbc.CardBody(html.Div(id="cluster-table"))
    ], style=CARD_STYLE)

    # ── Page content ─────────────────────────────────────────────────────────
    content = html.Div([
        # Header
        html.Div([
            html.H4("Diabetes Risk Dashboard", className="fw-bold mb-1", style={"color": "#0f172a"}),
            html.P("BC Analytics · CRISP-DM · XGBoost & Random Forest · K-Means Segmentation",
                   style={"color": "#64748b", "fontSize": "0.85rem"}),
        ], className="mb-4"),

        kpi_row,

        dbc.Row([
            dbc.Col(main_chart, md=8),
            dbc.Col(cluster_pie, md=4),
        ], className="g-3 mb-4"),

        dbc.Row([
            dbc.Col(stage_bar, md=6),
            dbc.Col(scatter_matrix, md=6),
        ], className="g-3 mb-4"),

        dbc.Row([
            dbc.Col(prediction_panel, md=12),
        ], className="g-3 mb-4"),

        dbc.Row([
            dbc.Col(cluster_table, md=12),
        ], className="g-3 mb-4"),

    ], style=CONTENT_STYLE)

    return html.Div([sidebar, content])