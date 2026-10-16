"""
AI-Based Wind Energy Prediction Tool
Streamlit Dashboard — Full Stack
Run with: streamlit run app.py
"""

import streamlit as st
import requests
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import math
import os
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="WindAI — Energy Predictor",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

:root {
    --bg: #0a0e17;
    --surface: #111827;
    --border: #1e2d40;
    --accent: #00d4ff;
    --accent2: #00ff9d;
    --warn: #ffb700;
    --danger: #ff4444;
    --text: #e2e8f0;
    --muted: #64748b;
}

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

.stApp { background-color: var(--bg); }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}

/* Metric cards */
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}
.metric-label {
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    font-family: 'Space Mono', monospace;
    margin-bottom: 8px;
}
.metric-value {
    font-size: 32px;
    font-weight: 800;
    color: var(--accent);
    line-height: 1;
}
.metric-unit {
    font-size: 14px;
    color: var(--muted);
    font-family: 'Space Mono', monospace;
    margin-top: 4px;
}

/* Status badge */
.status-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    font-family: 'Space Mono', monospace;
}
.status-running { background: rgba(0,255,157,0.15); color: #00ff9d; border: 1px solid #00ff9d44; }
.status-rated   { background: rgba(0,212,255,0.15); color: #00d4ff; border: 1px solid #00d4ff44; }
.status-idle    { background: rgba(255,183,0,0.15);  color: #ffb700; border: 1px solid #ffb70044; }
.status-shutdown{ background: rgba(255,68,68,0.15);  color: #ff4444; border: 1px solid #ff444444; }

/* Section headers */
.section-title {
    font-size: 11px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--muted);
    font-family: 'Space Mono', monospace;
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
    margin-bottom: 16px;
}

/* Info row */
.info-row {
    display: flex;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid var(--border);
    font-size: 14px;
}
.info-key { color: var(--muted); font-family: 'Space Mono', monospace; font-size: 12px; }
.info-val { color: var(--text); font-weight: 600; }

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #0a0e17 0%, #0d1f35 50%, #0a1628 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: '🌬️';
    position: absolute;
    right: 40px;
    top: 50%;
    transform: translateY(-50%);
    font-size: 80px;
    opacity: 0.15;
}
.hero-title {
    font-size: 36px;
    font-weight: 800;
    background: linear-gradient(90deg, #00d4ff, #00ff9d);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 6px;
}
.hero-sub {
    color: var(--muted);
    font-size: 14px;
    font-family: 'Space Mono', monospace;
}

/* Streamlit overrides */
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stNumberInput"] label { color: var(--muted) !important; font-size: 13px; }

.stButton > button {
    background: linear-gradient(135deg, #00d4ff22, #00ff9d22) !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #00d4ff44, #00ff9d44) !important;
    transform: translateY(-1px);
}

div[data-testid="metric-container"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 16px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
API_KEY    = "a2a8fa5cfe4e4dd6386275b5845d63b1"
AIR_DENSITY = 1.225

TURBINE_PRESETS = {
    "Vestas V90 — 2 MW (Utility Scale)": {
        "rotor_diameter": 90, "cp": 0.40, "tsr": 7.0,
        "cut_in": 3.0, "rated_speed": 12.0, "cut_out": 25.0, "rated_power_kw": 2000
    },
    "Siemens SWT-3.6 — 3.6 MW (Offshore)": {
        "rotor_diameter": 120, "cp": 0.42, "tsr": 8.0,
        "cut_in": 3.5, "rated_speed": 13.0, "cut_out": 25.0, "rated_power_kw": 3600
    },
    "GE 1.5s — 1.5 MW (Onshore)": {
        "rotor_diameter": 77, "cp": 0.38, "tsr": 6.5,
        "cut_in": 3.5, "rated_speed": 11.5, "cut_out": 20.0, "rated_power_kw": 1500
    },
    "Small Turbine — 10 kW": {
        "rotor_diameter": 7, "cp": 0.35, "tsr": 6.0,
        "cut_in": 2.5, "rated_speed": 10.0, "cut_out": 20.0, "rated_power_kw": 10
    },
    "⚙️ Custom": None
}

FEATURES = [
    "wind_speed_ms", "wind_deg", "temperature_c",
    "humidity_pct", "pressure_hpa", "hour_of_day",
    "air_density", "wind_speed_cubed"
]

# ─────────────────────────────────────────────
#  PHYSICS
# ─────────────────────────────────────────────
def physics_power_kw(wind_speed_ms, turbine, air_density=AIR_DENSITY):
    radius     = turbine["rotor_diameter"] / 2
    swept_area = math.pi * radius ** 2
    if wind_speed_ms < turbine["cut_in"] or wind_speed_ms > turbine["cut_out"]:
        return 0.0
    ws = min(wind_speed_ms, turbine["rated_speed"])
    power_w = 0.5 * air_density * swept_area * (ws ** 3) * turbine["cp"]
    return round(min(power_w / 1000, turbine["rated_power_kw"]), 2)

def turbine_rpm(wind_speed_ms, turbine):
    radius = turbine["rotor_diameter"] / 2
    if wind_speed_ms < turbine["cut_in"]:
        return 0.0
    omega = (turbine["tsr"] * wind_speed_ms) / radius
    return round((omega * 60) / (2 * math.pi), 2)

def turbine_status(wind_speed_ms, turbine):
    ws = wind_speed_ms
    if ws < turbine["cut_in"]:
        return "idle", "IDLE — Wind Too Low"
    elif ws > turbine["cut_out"]:
        return "shutdown", "SHUTDOWN — Wind Too High"
    elif ws >= turbine["rated_speed"]:
        return "rated", "RATED POWER"
    else:
        return "running", "RUNNING NORMALLY"

# ─────────────────────────────────────────────
#  ML MODEL
# ─────────────────────────────────────────────
def generate_training_data(turbine, n=5000):
    np.random.seed(42)
    ws   = np.random.weibull(2.0, n) * 8
    wd   = np.random.uniform(0, 360, n)
    temp = np.random.normal(25, 10, n)
    hum  = np.random.uniform(20, 100, n)
    pres = np.random.normal(1013, 10, n)
    hr   = np.random.randint(0, 24, n)
    rho  = (pres * 100) / (287.05 * (temp + 273.15))
    power = np.array([
        physics_power_kw(w, turbine, r) * np.random.uniform(0.95, 1.05)
        for w, r in zip(ws, rho)
    ])
    return pd.DataFrame({
        "wind_speed_ms": ws, "wind_deg": wd, "temperature_c": temp,
        "humidity_pct": hum, "pressure_hpa": pres, "hour_of_day": hr,
        "air_density": rho, "wind_speed_cubed": ws**3,
        "power_kw": np.clip(power, 0, None)
    })

@st.cache_resource(show_spinner=False)
def get_model(turbine_key, turbine):
    df = generate_training_data(turbine)
    X, y = df[FEATURES], df["power_kw"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_tr_s, y_tr)
    y_pred = model.predict(X_te_s)
    metrics = {
        "mae":  round(mean_absolute_error(y_te, y_pred), 2),
        "rmse": round(math.sqrt(mean_squared_error(y_te, y_pred)), 2),
        "r2":   round(r2_score(y_te, y_pred), 4)
    }
    return model, scaler, metrics

# ─────────────────────────────────────────────
#  WEATHER API
# ─────────────────────────────────────────────
def fetch_weather(city="Delhi", country="IN"):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},{country}&appid={API_KEY}&units=metric"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        d = r.json()
        ws = d["wind"]["speed"]
        temp = d["main"]["temp"]
        pres = d["main"]["pressure"]
        rho  = (pres * 100) / (287.05 * (temp + 273.15))
        return {
            "city"          : d["name"],
            "wind_speed_ms" : ws,
            "wind_speed_kmh": round(ws * 3.6, 2),
            "wind_deg"      : d["wind"].get("deg", 0),
            "temperature_c" : temp,
            "humidity_pct"  : d["main"]["humidity"],
            "pressure_hpa"  : pres,
            "description"   : d["weather"][0]["description"].title(),
            "icon"          : d["weather"][0]["icon"],
            "hour_of_day"   : datetime.utcnow().hour,
            "air_density"   : round(rho, 4),
            "wind_speed_cubed": round(ws**3, 3),
            "timestamp"     : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "error"         : None
        }
    except Exception as e:
        return {"error": str(e)}

# ─────────────────────────────────────────────
#  PREDICTION
# ─────────────────────────────────────────────
def predict(weather, model, scaler, turbine):
    X = scaler.transform([[
        weather["wind_speed_ms"], weather["wind_deg"], weather["temperature_c"],
        weather["humidity_pct"],  weather["pressure_hpa"], weather["hour_of_day"],
        weather["air_density"],   weather["wind_speed_cubed"]
    ]])
    ml_kw  = max(0.0, round(model.predict(X)[0], 2))
    phy_kw = physics_power_kw(weather["wind_speed_ms"], turbine, weather["air_density"])
    rpm    = turbine_rpm(weather["wind_speed_ms"], turbine)
    st, sl = turbine_status(weather["wind_speed_ms"], turbine)
    return {
        "ml_kw"     : ml_kw,
        "phy_kw"    : phy_kw,
        "rpm"       : rpm,
        "status"    : st,
        "status_label": sl,
        "homes"     : round(ml_kw / 1.5),
        "cap_factor": round((ml_kw / turbine["rated_power_kw"]) * 100, 1)
    }

# ─────────────────────────────────────────────
#  POWER CURVE CHART
# ─────────────────────────────────────────────
def power_curve_chart(turbine, current_ws=None):
    ws_range = np.linspace(0, 30, 300)
    power    = [physics_power_kw(w, turbine) for w in ws_range]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ws_range, y=power,
        mode="lines", name="Power Curve",
        line=dict(color="#00d4ff", width=2.5),
        fill="tozeroy", fillcolor="rgba(0,212,255,0.08)"
    ))

    if current_ws is not None:
        current_p = physics_power_kw(current_ws, turbine)
        fig.add_trace(go.Scatter(
            x=[current_ws], y=[current_p],
            mode="markers", name="Current",
            marker=dict(color="#00ff9d", size=12, symbol="circle",
                        line=dict(color="#00ff9d", width=2))
        ))
        fig.add_vline(x=current_ws, line_dash="dot", line_color="rgba(0,255,157,0.27)")

    # Mark key speeds
    for speed, label, color in [
        (turbine["cut_in"],    "Cut-in",   "#ffb700"),
        (turbine["rated_speed"], "Rated",  "#00d4ff"),
        (turbine["cut_out"],   "Cut-out",  "#ff4444"),
    ]:
        fig.add_vline(x=speed, line_dash="dash", line_color=color, opacity=0.4)
        fig.add_annotation(x=speed, y=turbine["rated_power_kw"]*0.95,
                           text=label, font=dict(color=color, size=10),
                           showarrow=False, yshift=5)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Space Mono", color="#64748b", size=11),
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", y=-0.15, font=dict(size=10)),
        xaxis=dict(title="Wind Speed (m/s)", gridcolor="#1e2d40", zeroline=False),
        yaxis=dict(title="Power (kW)", gridcolor="#1e2d40", zeroline=False),
        hovermode="x unified"
    )
    return fig

# ─────────────────────────────────────────────
#  WIND SPEED SIMULATION CHART
# ─────────────────────────────────────────────
def simulation_chart(turbine):
    ws_vals = np.linspace(0, turbine["cut_out"] + 3, 50)
    power   = [physics_power_kw(w, turbine) for w in ws_vals]
    rpm     = [turbine_rpm(w, turbine) for w in ws_vals]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=ws_vals, y=power, name="Power (kW)",
        marker_color=[
            "#1e2d40" if p == 0 else
            f"rgba(0,{int(212 * p / turbine['rated_power_kw'])},255,0.8)"
            for p in power
        ]
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Space Mono", color="#64748b", size=10),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(title="Wind Speed (m/s)", gridcolor="#1e2d40"),
        yaxis=dict(title="Power (kW)", gridcolor="#1e2d40"),
        showlegend=False
    )
    return fig

# ─────────────────────────────────────────────
#  GAUGE CHART
# ─────────────────────────────────────────────
def gauge_chart(value, max_val, label, color="#00d4ff"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": label, "font": {"family": "Space Mono", "size": 12, "color": "#64748b"}},
        number={"font": {"family": "Syne", "size": 28, "color": color}},
        gauge={
            "axis": {"range": [0, max_val], "tickcolor": "#1e2d40",
                     "tickfont": {"size": 9, "color": "#64748b"}},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": "#111827",
            "bordercolor": "#1e2d40",
            "steps": [
                {"range": [0, max_val * 0.33], "color": "#0d1a26"},
                {"range": [max_val * 0.33, max_val * 0.66], "color": "#0d2033"},
                {"range": [max_val * 0.66, max_val], "color": "#0d2640"},
            ],
            "threshold": {
                "line": {"color": "#00ff9d", "width": 2},
                "thickness": 0.8, "value": value
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
        height=200
    )
    return fig

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 16px 0 8px'>
        <div style='font-size:20px; font-weight:800; color:#00d4ff;'>WindAI</div>
        <div style='font-size:10px; letter-spacing:3px; color:#64748b; font-family:Space Mono;'>ENERGY PREDICTOR</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("<div class='section-title'>📍 Location</div>", unsafe_allow_html=True)
    city    = st.text_input("City", value="Delhi")
    country = st.text_input("Country Code", value="IN")

    st.markdown("---")
    st.markdown("<div class='section-title'>🔩 Turbine Setup</div>", unsafe_allow_html=True)

    preset_name = st.selectbox("Select Preset", list(TURBINE_PRESETS.keys()))

    if preset_name == "⚙️ Custom":
        st.markdown("<div style='font-size:12px; color:#64748b; margin-top:8px;'>Custom Specs</div>", unsafe_allow_html=True)
        rotor_d     = st.number_input("Rotor Diameter (m)", 5.0, 200.0, 90.0, step=1.0)
        cp          = st.slider("Power Coefficient (Cp)", 0.25, 0.50, 0.40, 0.01)
        tsr         = st.slider("Tip Speed Ratio (TSR)", 4.0, 10.0, 7.0, 0.1)
        cut_in      = st.number_input("Cut-in Speed (m/s)", 1.0, 6.0, 3.0, 0.5)
        rated_speed = st.number_input("Rated Speed (m/s)", 8.0, 20.0, 12.0, 0.5)
        cut_out     = st.number_input("Cut-out Speed (m/s)", 15.0, 35.0, 25.0, 0.5)
        rated_kw    = st.number_input("Rated Power (kW)", 1.0, 15000.0, 2000.0, 100.0)
        turbine = {
            "rotor_diameter": rotor_d, "cp": cp, "tsr": tsr,
            "cut_in": cut_in, "rated_speed": rated_speed,
            "cut_out": cut_out, "rated_power_kw": rated_kw
        }
        turbine_key = f"custom_{rotor_d}_{cp}_{tsr}"
    else:
        turbine = TURBINE_PRESETS[preset_name]
        turbine_key = preset_name

    st.markdown("---")
    fetch_btn = st.button("⚡  RUN PREDICTION", use_container_width=True)

# ─────────────────────────────────────────────
#  MAIN CONTENT
# ─────────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <div class='hero-title'>Wind Energy Predictor</div>
    <div class='hero-sub'>AI-powered · Real-time weather · ML forecast · Delhi, IN</div>
</div>
""", unsafe_allow_html=True)

# ─── Train model (cached) ───
with st.spinner("⚙️ Loading ML model..."):
    model, scaler, model_metrics = get_model(turbine_key, turbine)

# Model quality strip
c1, c2, c3 = st.columns(3)
c1.metric("Model MAE",  f"{model_metrics['mae']} kW")
c2.metric("Model RMSE", f"{model_metrics['rmse']} kW")
c3.metric("Model R²",   f"{model_metrics['r2']}")

st.markdown("---")

# ─── Run prediction on button click ───
if fetch_btn:
    with st.spinner("🌐 Fetching weather data..."):
        weather = fetch_weather(city, country)

    if weather.get("error"):
        st.error(f"❌ Could not fetch weather: {weather['error']}")
    else:
        pred = predict(weather, model, scaler, turbine)
        st.session_state["weather"] = weather
        st.session_state["pred"]    = pred
        st.session_state["turbine"] = turbine

# ─── Display results if available ───
if "pred" in st.session_state:
    w = st.session_state["weather"]
    p = st.session_state["pred"]
    t = st.session_state["turbine"]

    # Status badge
    badge_class = f"status-{p['status']}"
    st.markdown(f"""
    <div style='margin-bottom:24px; display:flex; align-items:center; gap:16px;'>
        <span class='status-badge {badge_class}'>{p['status_label']}</span>
        <span style='color:#64748b; font-family:Space Mono; font-size:12px;'>
            Last updated: {w['timestamp']}
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Top metric cards ──
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>ML Prediction</div>
            <div class='metric-value'>{p['ml_kw']}</div>
            <div class='metric-unit'>kilowatts</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Wind Speed</div>
            <div class='metric-value'>{w['wind_speed_ms']}</div>
            <div class='metric-unit'>m/s · {w['wind_speed_kmh']} km/hr</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Turbine RPM</div>
            <div class='metric-value'>{p['rpm']}</div>
            <div class='metric-unit'>rotations/min</div>
        </div>""", unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Homes Powered</div>
            <div class='metric-value'>{p['homes']}</div>
            <div class='metric-unit'>avg households</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # ── Gauges + Weather + Turbine info ──
    left, mid, right = st.columns([2, 2, 2])

    with left:
        st.markdown("<div class='section-title'>⚡ Power Gauges</div>", unsafe_allow_html=True)
        st.plotly_chart(gauge_chart(p['ml_kw'],  t['rated_power_kw'], "ML Prediction (kW)", "#00d4ff"), use_container_width=True, config={"displayModeBar": False})
        st.plotly_chart(gauge_chart(p['cap_factor'], 100, "Capacity Factor (%)", "#00ff9d"), use_container_width=True, config={"displayModeBar": False})

    with mid:
        st.markdown("<div class='section-title'>🌡️ Live Weather</div>", unsafe_allow_html=True)
        weather_rows = [
            ("Condition",    w['description']),
            ("Wind Speed",   f"{w['wind_speed_ms']} m/s ({w['wind_speed_kmh']} km/hr)"),
            ("Direction",    f"{w['wind_deg']}°"),
            ("Temperature",  f"{w['temperature_c']}°C"),
            ("Humidity",     f"{w['humidity_pct']}%"),
            ("Pressure",     f"{w['pressure_hpa']} hPa"),
            ("Air Density",  f"{w['air_density']} kg/m³"),
        ]
        for k, v in weather_rows:
            st.markdown(f"""
            <div class='info-row'>
                <span class='info-key'>{k}</span>
                <span class='info-val'>{v}</span>
            </div>""", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='section-title'>🔩 Turbine Specs</div>", unsafe_allow_html=True)
        turbine_rows = [
            ("Preset",       preset_name.split("—")[0].strip()),
            ("Rotor Ø",      f"{t['rotor_diameter']} m"),
            ("Cp",           t['cp']),
            ("TSR",          t['tsr']),
            ("Cut-in",       f"{t['cut_in']} m/s"),
            ("Rated Speed",  f"{t['rated_speed']} m/s"),
            ("Cut-out",      f"{t['cut_out']} m/s"),
            ("Rated Power",  f"{t['rated_power_kw']} kW"),
        ]
        for k, v in turbine_rows:
            st.markdown(f"""
            <div class='info-row'>
                <span class='info-key'>{k}</span>
                <span class='info-val'>{v}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Charts ──
    chart_l, chart_r = st.columns(2)

    with chart_l:
        st.markdown("<div class='section-title'>📈 Power Curve</div>", unsafe_allow_html=True)
        st.plotly_chart(
            power_curve_chart(t, w["wind_speed_ms"]),
            use_container_width=True,
            config={"displayModeBar": False}
        )

    with chart_r:
        st.markdown("<div class='section-title'>📊 Output vs Wind Speed</div>", unsafe_allow_html=True)
        st.plotly_chart(
            simulation_chart(t),
            use_container_width=True,
            config={"displayModeBar": False}
        )

    st.markdown("---")

    # ── ML vs Physics comparison ──
    st.markdown("<div class='section-title'>🤖 ML Prediction vs Physics Estimate</div>", unsafe_allow_html=True)
    comp_col1, comp_col2, comp_col3 = st.columns(3)
    diff = round(p['ml_kw'] - p['phy_kw'], 2)
    comp_col1.metric("ML Model Output",      f"{p['ml_kw']} kW")
    comp_col2.metric("Physics Formula",      f"{p['phy_kw']} kW")
    comp_col3.metric("Difference",           f"{diff} kW", delta=f"{diff} kW")

    # ── Export ──
    st.markdown("---")
    report = {**w, **p, "turbine": t, "model_metrics": model_metrics}
    report.pop("error", None)
    st.download_button(
        label="💾  Download Report (JSON)",
        data=json.dumps(report, indent=2),
        file_name=f"wind_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

else:
    # ── Placeholder charts before first prediction ──
    st.info("👈  Configure your turbine in the sidebar and click **RUN PREDICTION** to begin.")
    st.markdown("<div class='section-title'>📈 Power Curve Preview</div>", unsafe_allow_html=True)
    st.plotly_chart(power_curve_chart(turbine), use_container_width=True, config={"displayModeBar": False})
