"""NYISO Analytics Dashboard - With ML Predictions."""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "full_featured_data"
PREDICTIONS_PATH = PROJECT_ROOT / "results" / "predictions.parquet"
METRICS_PATH = PROJECT_ROOT / "results" / "model_metrics.json"

# Load data
print("Loading data...")
if PREDICTIONS_PATH.exists():
    df = pd.read_parquet(PREDICTIONS_PATH)
    print(f"Loaded predictions: {len(df):,} records")
elif DATA_PATH.exists():
    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded featured data: {len(df):,} records")
else:
    print("No data found!")
    df = pd.DataFrame()

# Load metrics
metrics = {}
if METRICS_PATH.exists():
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    print("Loaded model metrics")

# Get zones
ZONES = sorted(df["Name"].unique().tolist()) if "Name" in df.columns and len(df) > 0 else ["N.Y.C."]

# Initialize app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    suppress_callback_exceptions=True,
    title="NYISO Analytics"
)

COLORS = {
    "primary": "#00d4ff",
    "secondary": "#ff6b6b",
    "success": "#4ecdc4",
    "warning": "#ffd93d",
    "purple": "#a855f7",
    "bg": "#0f0f23",
    "card": "#1a1a3e"
}

# Calculate stats
if len(df) > 0:
    avg_price = df["LBMP_avg"].mean()
    max_load = df["Load_MW"].max()
    spike_count = df["is_price_spike"].sum() if "is_price_spike" in df.columns else 0
    total_records = len(df)

    # Model metrics
    price_r2 = metrics.get("price_prediction", {}).get("r2", 0)
    demand_r2 = metrics.get("demand_forecast", {}).get("r2", 0)
    spike_auc = metrics.get("spike_detection", {}).get("auc", 0)
else:
    avg_price = max_load = spike_count = total_records = 0
    price_r2 = demand_r2 = spike_auc = 0

# Layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H2("NYISO Analytics Dashboard", className="text-info mb-1"),
            html.P("Electricity Market Analysis with ML Predictions", className="text-muted")
        ])
    ], className="mb-3"),

    # Filters
    dbc.Row([
        dbc.Col([
            html.Label("Zone", className="text-light"),
            dcc.Dropdown(id="zone-filter", options=[{"label": z, "value": z} for z in ZONES],
                        value=ZONES[0] if ZONES else None, clearable=False, className="mb-2")
        ], md=2),
        dbc.Col([
            html.Label("Month", className="text-light"),
            dcc.Dropdown(id="month-filter", options=[{"label": f"Month {m}", "value": m} for m in range(1, 13)],
                        value=None, placeholder="All", className="mb-2")
        ], md=2),
    ], className="mb-3"),

    # Model Performance Cards
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("Price Prediction", className="text-muted mb-1"),
                html.H3(f"R² = {price_r2:.3f}", style={"color": COLORS["primary"]}),
                html.Small(f"RMSE: {metrics.get('price_prediction', {}).get('rmse', 0):.1f} $/MWh", className="text-muted")
            ])
        ], style={"backgroundColor": COLORS["card"]}), md=2),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("Demand Forecast", className="text-muted mb-1"),
                html.H3(f"R² = {demand_r2:.3f}", style={"color": COLORS["success"]}),
                html.Small(f"RMSE: {metrics.get('demand_forecast', {}).get('rmse', 0):.1f} MW", className="text-muted")
            ])
        ], style={"backgroundColor": COLORS["card"]}), md=2),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("Spike Detection", className="text-muted mb-1"),
                html.H3(f"AUC = {spike_auc:.3f}", style={"color": COLORS["warning"]}),
                html.Small(f"Acc: {metrics.get('spike_detection', {}).get('accuracy', 0)*100:.1f}%", className="text-muted")
            ])
        ], style={"backgroundColor": COLORS["card"]}), md=2),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("Avg Price", className="text-muted mb-1"),
                html.H3(f"${avg_price:.2f}", style={"color": COLORS["secondary"]}),
                html.Small("/MWh", className="text-muted")
            ])
        ], style={"backgroundColor": COLORS["card"]}), md=2),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("Peak Load", className="text-muted mb-1"),
                html.H3(f"{max_load:,.0f}", style={"color": COLORS["purple"]}),
                html.Small("MW", className="text-muted")
            ])
        ], style={"backgroundColor": COLORS["card"]}), md=2),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("Records", className="text-muted mb-1"),
                html.H3(f"{total_records:,}", style={"color": "#888"}),
                html.Small(f"Spikes: {int(spike_count)}", className="text-muted")
            ])
        ], style={"backgroundColor": COLORS["card"]}), md=2),
    ], className="mb-4"),

    # Row 1: Price Prediction
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Price: Actual vs Predicted", className="bg-dark"),
                dbc.CardBody(dcc.Graph(id="price-prediction-chart"))
            ], style={"backgroundColor": COLORS["card"]})
        ], md=8),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Price Feature Importance", className="bg-dark"),
                dbc.CardBody(dcc.Graph(id="price-importance-chart"))
            ], style={"backgroundColor": COLORS["card"]})
        ], md=4),
    ], className="mb-4"),

    # Row 2: Demand Forecast
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Demand: Actual vs Predicted", className="bg-dark"),
                dbc.CardBody(dcc.Graph(id="demand-prediction-chart"))
            ], style={"backgroundColor": COLORS["card"]})
        ], md=8),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Demand Feature Importance", className="bg-dark"),
                dbc.CardBody(dcc.Graph(id="demand-importance-chart"))
            ], style={"backgroundColor": COLORS["card"]})
        ], md=4),
    ], className="mb-4"),

    # Row 3: Analysis Charts
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Price Heatmap (Hour x Day)", className="bg-dark"),
                dbc.CardBody(dcc.Graph(id="price-heatmap"))
            ], style={"backgroundColor": COLORS["card"]})
        ], md=4),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Zone Comparison", className="bg-dark"),
                dbc.CardBody(dcc.Graph(id="zone-chart"))
            ], style={"backgroundColor": COLORS["card"]})
        ], md=4),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Price Distribution", className="bg-dark"),
                dbc.CardBody(dcc.Graph(id="price-histogram"))
            ], style={"backgroundColor": COLORS["card"]})
        ], md=4),
    ], className="mb-4"),

    # Row 4: Time Series & Correlation
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Daily Price & Load Trends", className="bg-dark"),
                dbc.CardBody(dcc.Graph(id="timeseries-chart"))
            ], style={"backgroundColor": COLORS["card"]})
        ], md=8),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Hourly Patterns", className="bg-dark"),
                dbc.CardBody(dcc.Graph(id="hourly-chart"))
            ], style={"backgroundColor": COLORS["card"]})
        ], md=4),
    ]),

], fluid=True, style={"backgroundColor": COLORS["bg"], "minHeight": "100vh", "padding": "20px"})


# Callbacks
@callback(
    Output("price-prediction-chart", "figure"),
    [Input("zone-filter", "value"), Input("month-filter", "value")]
)
def update_price_prediction(zone, month):
    if len(df) == 0 or "predicted_price" not in df.columns:
        return go.Figure()

    filtered = df[df["Name"] == zone] if zone else df
    if month:
        filtered = filtered[filtered["month"] == month]

    sample = filtered.sample(n=min(2000, len(filtered)), random_state=42).sort_values("hour_timestamp")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sample["hour_timestamp"], y=sample["LBMP_avg"],
                             name="Actual", line=dict(color=COLORS["primary"], width=1)))
    fig.add_trace(go.Scatter(x=sample["hour_timestamp"], y=sample["predicted_price"],
                             name="Predicted", line=dict(color=COLORS["secondary"], width=1, dash="dot")))

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.1), margin=dict(l=50, r=20, t=30, b=50),
        yaxis_title="Price ($/MWh)", height=300
    )
    return fig


@callback(
    Output("price-importance-chart", "figure"),
    Input("zone-filter", "value")
)
def update_price_importance(zone):
    importance = metrics.get("price_feature_importance", {})
    if not importance:
        return go.Figure()

    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=False)
    features = [x[0] for x in sorted_imp[-10:]]
    values = [x[1] for x in sorted_imp[-10:]]

    fig = go.Figure(go.Bar(x=values, y=features, orientation="h", marker_color=COLORS["primary"]))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=120, r=20, t=20, b=30), height=300, xaxis_title="Importance"
    )
    return fig


@callback(
    Output("demand-prediction-chart", "figure"),
    [Input("zone-filter", "value"), Input("month-filter", "value")]
)
def update_demand_prediction(zone, month):
    if len(df) == 0 or "predicted_load" not in df.columns:
        return go.Figure()

    filtered = df[df["Name"] == zone] if zone else df
    if month:
        filtered = filtered[filtered["month"] == month]

    sample = filtered.sample(n=min(2000, len(filtered)), random_state=42).sort_values("hour_timestamp")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sample["hour_timestamp"], y=sample["Load_MW"],
                             name="Actual", line=dict(color=COLORS["success"], width=1)))
    fig.add_trace(go.Scatter(x=sample["hour_timestamp"], y=sample["predicted_load"],
                             name="Predicted", line=dict(color=COLORS["warning"], width=1, dash="dot")))

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.1), margin=dict(l=50, r=20, t=30, b=50),
        yaxis_title="Load (MW)", height=300
    )
    return fig


@callback(
    Output("demand-importance-chart", "figure"),
    Input("zone-filter", "value")
)
def update_demand_importance(zone):
    importance = metrics.get("demand_feature_importance", {})
    if not importance:
        return go.Figure()

    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=False)
    features = [x[0] for x in sorted_imp[-10:]]
    values = [x[1] for x in sorted_imp[-10:]]

    fig = go.Figure(go.Bar(x=values, y=features, orientation="h", marker_color=COLORS["success"]))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=120, r=20, t=20, b=30), height=300, xaxis_title="Importance"
    )
    return fig


@callback(
    Output("price-heatmap", "figure"),
    [Input("zone-filter", "value"), Input("month-filter", "value")]
)
def update_heatmap(zone, month):
    if len(df) == 0:
        return go.Figure()

    filtered = df[df["Name"] == zone] if zone else df
    if month:
        filtered = filtered[filtered["month"] == month]

    pivot = filtered.groupby(["day_of_week", "hour"])["LBMP_avg"].mean().unstack(fill_value=0)

    fig = go.Figure(go.Heatmap(z=pivot.values, x=list(range(24)),
                                y=["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],
                                colorscale="Viridis"))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=60, r=20, t=20, b=40), height=280, xaxis_title="Hour"
    )
    return fig


@callback(
    Output("zone-chart", "figure"),
    Input("month-filter", "value")
)
def update_zones(month):
    if len(df) == 0:
        return go.Figure()

    filtered = df if month is None else df[df["month"] == month]
    zone_stats = filtered.groupby("Name").agg({"LBMP_avg": "mean", "Load_MW": "mean"}).reset_index()
    zone_stats = zone_stats.sort_values("LBMP_avg", ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(y=zone_stats["Name"], x=zone_stats["LBMP_avg"],
                         orientation="h", name="Price", marker_color=COLORS["primary"]))

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=70, r=20, t=20, b=40), height=280, xaxis_title="Avg Price ($/MWh)"
    )
    return fig


@callback(
    Output("price-histogram", "figure"),
    [Input("zone-filter", "value"), Input("month-filter", "value")]
)
def update_histogram(zone, month):
    if len(df) == 0:
        return go.Figure()

    filtered = df[df["Name"] == zone] if zone else df
    if month:
        filtered = filtered[filtered["month"] == month]

    fig = go.Figure(go.Histogram(x=filtered["LBMP_avg"], nbinsx=50, marker_color=COLORS["primary"]))

    # Add spike threshold line
    if "price_ma_24h" in filtered.columns:
        threshold = filtered["price_ma_24h"].mean() + 3 * filtered["LBMP_avg"].std()
        fig.add_vline(x=threshold, line_dash="dash", line_color=COLORS["warning"],
                      annotation_text="Spike Threshold")

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=50, r=20, t=20, b=40), height=280, xaxis_title="Price ($/MWh)"
    )
    return fig


@callback(
    Output("timeseries-chart", "figure"),
    [Input("zone-filter", "value"), Input("month-filter", "value")]
)
def update_timeseries(zone, month):
    if len(df) == 0:
        return go.Figure()

    filtered = df[df["Name"] == zone] if zone else df
    if month:
        filtered = filtered[filtered["month"] == month]

    # Aggregate to daily
    filtered = filtered.copy()
    filtered["date"] = pd.to_datetime(filtered["hour_timestamp"]).dt.date
    daily = filtered.groupby("date").agg({"LBMP_avg": "mean", "Load_MW": "mean"}).reset_index()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["LBMP_avg"], name="Price",
                             line=dict(color=COLORS["primary"])), secondary_y=False)
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["Load_MW"], name="Load",
                             line=dict(color=COLORS["success"])), secondary_y=True)

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.1), margin=dict(l=50, r=50, t=30, b=50), height=300
    )
    fig.update_yaxes(title_text="Price ($/MWh)", secondary_y=False)
    fig.update_yaxes(title_text="Load (MW)", secondary_y=True)
    return fig


@callback(
    Output("hourly-chart", "figure"),
    [Input("zone-filter", "value"), Input("month-filter", "value")]
)
def update_hourly(zone, month):
    if len(df) == 0:
        return go.Figure()

    filtered = df[df["Name"] == zone] if zone else df
    if month:
        filtered = filtered[filtered["month"] == month]

    hourly = filtered.groupby("hour").agg({"LBMP_avg": "mean", "Load_MW": "mean"}).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(x=hourly["hour"], y=hourly["LBMP_avg"], name="Price", marker_color=COLORS["primary"]))

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=50, r=20, t=20, b=40), height=300, xaxis_title="Hour", yaxis_title="Avg Price"
    )
    return fig


if __name__ == "__main__":
    print("\n" + "="*50)
    print("Starting NYISO Dashboard")
    print("="*50)
    print(f"Data: {len(df):,} records")
    print(f"URL: http://127.0.0.1:8050")
    print("="*50 + "\n")
    app.run(host="0.0.0.0", port=8050, debug=False)
