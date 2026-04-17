"""
Renewable Energy Forecast — Energy Forecast.py
Solar & Wind Forecast with Open-Meteo + LSTM
Fixed: pandas 3.x groupby, normalize_features, tensor shapes, altair theme,
       seed sequence NaN, default hyperparams, use_container_width deprecation
"""

_SOWI_AI_PAGE = "pages/Sowi_AI_Analyst.py"

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import requests
import altair as alt
from datetime import datetime, timedelta
import math
import time

st.set_page_config(
    page_title="Renewable Energy Forecast",
    page_icon="🌞⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root {
  --bg0:#06060e; --bg1:#0c0c18; --surface:#181828; --surface2:#1f1f32; --surface3:#26263c;
  --bd:rgba(255,255,255,0.06); --bd-hi:rgba(255,255,255,0.11);
  --amber:#f5b432; --amber-dim:rgba(245,180,50,0.10); --amber-glow:rgba(245,180,50,0.18);
  --green:#22c55e;  --green-dim:rgba(34,197,94,0.10);  --green-glow:rgba(34,197,94,0.18);
  --slate:#8899bb; --ok:#3ecf8e; --warn:#f5a623;
  --t1:#eeeef4; --t2:#8888a0; --t3:#44445a;
  --font:'Sora',system-ui,sans-serif; --mono:'JetBrains Mono',monospace;
  --r:12px; --rl:20px; --pill:100px;
}
html,body,[class*="css"]{ font-family:var(--font)!important; }
.stApp { background:var(--bg0)!important; }
[data-testid="stAppViewContainer"] {
  background-color:var(--bg0);
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 400 400' opacity='0.04'%3E%3Cg transform='translate(60,60)'%3E%3Crect x='-4' y='0' width='8' height='50' fill='%238899bb'/%3E%3Cpath d='M0,0 L-6,-30 L0,-40 L6,-30 Z' fill='%238899bb'/%3E%3Cpath d='M0,0 L-30,-6 L-40,0 L-30,6 Z' fill='%238899bb'/%3E%3Cpath d='M0,0 L6,30 L0,40 L-6,30 Z' fill='%238899bb'/%3E%3C/g%3E%3Cg transform='translate(280,100)'%3E%3Crect x='-25' y='-15' width='50' height='30' rx='4' fill='%23f5b432'/%3E%3Cline x1='-25' y1='-5' x2='25' y2='-5' stroke='%23c47a1a' stroke-width='1.5'/%3E%3Cline x1='-25' y1='5' x2='25' y2='5' stroke='%23c47a1a' stroke-width='1.5'/%3E%3Crect x='-4' y='15' width='8' height='20' fill='%238899bb'/%3E%3C/g%3E%3Cg transform='translate(180,280)'%3E%3Crect x='-3' y='0' width='6' height='40' fill='%238899bb'/%3E%3Cpath d='M0,0 L-5,-24 L0,-32 L5,-24 Z' fill='%238899bb'/%3E%3Cpath d='M0,0 L-24,-5 L-32,0 L-24,5 Z' fill='%238899bb'/%3E%3Cpath d='M0,0 L5,24 L0,32 L-5,24 Z' fill='%238899bb'/%3E%3C/g%3E%3Cg transform='translate(80,240)'%3E%3Crect x='-20' y='-12' width='40' height='24' rx='3' fill='%23f5b432'/%3E%3Cline x1='-20' y1='-4' x2='20' y2='-4' stroke='%23c47a1a' stroke-width='1'/%3E%3Cline x1='-20' y1='4' x2='20' y2='4' stroke='%23c47a1a' stroke-width='1'/%3E%3Crect x='-3' y='12' width='6' height='15' fill='%238899bb'/%3E%3C/g%3E%3C/svg%3E");
  background-repeat:repeat; background-size:400px 400px;
}
.block-container { background:rgba(10,10,20,0.93);border-radius:var(--rl);padding:0 2rem 3rem!important;max-width:1120px!important;margin:0 auto!important; }
[data-testid="stSidebar"] { background:var(--bg1)!important;border-right:1px solid var(--bd)!important; }
#MainMenu,footer,header{ visibility:hidden; }
::-webkit-scrollbar{ width:4px; } ::-webkit-scrollbar-thumb{ background:var(--bd-hi);border-radius:4px; }
[data-testid="stSidebar"] *{ font-family:var(--font)!important; }
[data-testid="stSidebarContent"]{ padding-top:1.2rem!important; }
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p{ color:var(--t2)!important;font-size:0.82rem!important; }
[data-testid="stSidebar"] input{ background:var(--surface2)!important;border:1px solid var(--bd-hi)!important;border-radius:var(--r)!important;color:var(--t1)!important; }
[data-testid="stSidebar"] .stButton>button[kind="primary"]{
  background:linear-gradient(135deg,rgba(245,180,50,.2),rgba(245,180,50,.1))!important;
  border:1px solid rgba(245,180,50,.4)!important;color:var(--amber)!important;
  font-weight:600!important;border-radius:var(--r)!important;width:100%!important;transition:all .18s!important;
}
[data-testid="stSidebar"] .stButton>button[kind="primary"]:hover{ box-shadow:0 0 18px var(--amber-glow)!important;transform:translateY(-1px)!important; }
h1,h2,h3,h4{ font-family:var(--font)!important;color:var(--t1)!important; }
[data-testid="metric-container"]{ background:var(--surface)!important;border:1px solid var(--bd)!important;border-radius:var(--r)!important;padding:.9rem 1rem!important; }
[data-testid="metric-container"] label{ font-size:.68rem!important;font-weight:600!important;letter-spacing:.08em!important;text-transform:uppercase!important;color:var(--t3)!important; }
[data-testid="metric-container"] [data-testid="stMetricValue"]{ font-size:1.5rem!important;font-weight:700!important;color:var(--t1)!important; }
.stAlert{ background:var(--surface)!important;border:1px solid var(--bd-hi)!important;border-radius:var(--r)!important;color:var(--t1)!important; }
[data-baseweb="tab-list"]{ background:var(--surface)!important;border-radius:var(--r) var(--r) 0 0!important;border-bottom:1px solid var(--bd)!important;gap:2px!important; }
[data-baseweb="tab"]{ background:transparent!important;color:var(--t2)!important;font-size:.82rem!important;font-weight:500!important; }
[aria-selected="true"][data-baseweb="tab"]{ background:var(--surface2)!important;color:var(--amber)!important;border-bottom:2px solid var(--amber)!important; }
[data-baseweb="tab-panel"]{ background:var(--surface2)!important;border:1px solid var(--bd)!important;border-top:none!important;border-radius:0 0 var(--r) var(--r)!important;padding:1rem!important; }
[data-testid="stExpander"]{ background:var(--surface)!important;border:1px solid var(--bd)!important;border-radius:var(--r)!important; }
[data-testid="stDataFrame"]{ background:var(--surface)!important;border:1px solid var(--bd)!important;border-radius:var(--r)!important; }
[data-testid="stDataFrame"] th{ background:var(--surface2)!important;color:var(--t2)!important;font-size:.75rem!important;font-weight:600!important; }
[data-testid="stDataFrame"] td{ color:var(--t1)!important;font-size:.82rem!important; }
[data-testid="stProgressBar"]>div>div{ background:linear-gradient(90deg,var(--amber),var(--warn))!important;border-radius:var(--pill)!important; }
[data-testid="stProgressBar"]>div{ background:var(--surface3)!important;border-radius:var(--pill)!important; }
.stSpinner>div{ border-top-color:var(--amber)!important; }
.page-hero{ padding:2.2rem 0 1.6rem;border-bottom:1px solid var(--bd);margin-bottom:1.6rem; }
.page-eyebrow{ display:inline-flex;align-items:center;gap:7px;border:1px solid rgba(245,180,50,.2);background:rgba(245,180,50,.06);color:var(--amber);font-size:.67rem;font-weight:600;letter-spacing:.12em;text-transform:uppercase;padding:5px 14px;border-radius:var(--pill);margin-bottom:.9rem; }
.ldot{ width:5px;height:5px;background:var(--amber);border-radius:50%;animation:ldot 2.4s ease-in-out infinite; }
@keyframes ldot{ 0%,100%{opacity:1;box-shadow:0 0 0 0 rgba(245,180,50,.7)} 50%{opacity:.4;box-shadow:0 0 0 5px rgba(245,180,50,0)} }
.page-title{ font-size:clamp(1.9rem,4.5vw,3rem);font-weight:700;color:var(--t1);letter-spacing:-.045em;line-height:1.06;margin:0 0 .5rem; }
.page-sub{ font-size:.95rem;color:var(--t2);font-weight:300; }
.wave-sep{ height:1px;background:linear-gradient(90deg,transparent,rgba(245,180,50,.25) 35%,rgba(136,153,187,.2) 65%,transparent);margin:.3rem 0 1.6rem; }
.pred-metrics{ display:grid;grid-template-columns:repeat(4,1fr);gap:9px;margin-bottom:1.4rem; }
.pm{ background:var(--surface);border:1px solid var(--bd);border-radius:var(--r);padding:.9rem 1rem;position:relative;overflow:hidden;transition:border-color .2s,transform .18s; }
.pm:hover{ border-color:var(--bd-hi);transform:translateY(-2px); }
.pm::before{ content:'';position:absolute;top:0;left:0;right:0;height:2px; }
.pm-solar::before{ background:var(--amber); } .pm-wind::before{ background:var(--green); }
.pm-lbl{ font-size:.62rem;font-weight:600;letter-spacing:.09em;text-transform:uppercase;color:var(--t3);margin-bottom:.3rem; }
.pm-val{ font-size:1.3rem;font-weight:700;color:var(--t1);line-height:1; }
.pm-unit{ font-size:.65rem;color:var(--t2);font-weight:400; }
.pm-tag{ display:inline-block;font-size:.6rem;font-weight:600;padding:2px 7px;border-radius:var(--pill);margin-top:4px; }
.tag-solar{ background:var(--amber-dim);color:var(--amber); } .tag-wind{ background:var(--green-dim);color:var(--green); }
.chart-desc{ background:var(--surface);border-left:3px solid var(--amber);border-radius:0 var(--r) var(--r) 0;padding:.6rem 1rem;margin-bottom:.9rem;font-size:.77rem;color:var(--t2);line-height:1.55; }
.chart-desc-wind{ background:var(--surface);border-left:3px solid var(--green);border-radius:0 var(--r) var(--r) 0;padding:.6rem 1rem;margin-bottom:.9rem;font-size:.77rem;color:var(--t2);line-height:1.55; }
.status-banner{ background:var(--surface);border:1px solid var(--bd);border-radius:var(--r);padding:.5rem 1rem;font-size:.78rem;color:var(--t1);margin-bottom:1rem; }
.status-banner .meta{ color:var(--t3);font-size:.72rem; }
.vega-embed{ background:transparent!important; }
.footer{ text-align:center;margin-top:2.5rem;font-size:.63rem;color:var(--t3);letter-spacing:.07em; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# GEOCODING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False, ttl=86400)
def geocode(query: str):
    query = query.strip()
    if not query:
        return None, None, None

    headers = {
        "User-Agent": "RenewableEnergyForecast/3.0 (energy-study@sowi.app)",
        "Accept-Language": "en,es;q=0.9,*;q=0.5",
        "Accept": "application/json",
    }

    nominatim_variants = [
        {"q": query, "format": "json", "limit": 5, "addressdetails": 1,
         "accept-language": "en", "extratags": 1},
        {"q": query, "format": "json", "limit": 5, "addressdetails": 1,
         "accept-language": "en", "countrycodes": ""},
        {"q": query.replace(",", " ").replace("  ", " "), "format": "json",
         "limit": 5, "accept-language": "en"},
    ]
    for params in nominatim_variants:
        try:
            time.sleep(1.1)
            r = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params=params, headers=headers, timeout=15
            )
            r.raise_for_status()
            results = r.json()
            if results:
                best = results[0]
                return float(best["lat"]), float(best["lon"]), best.get("display_name", query)
        except Exception:
            pass

    try:
        r = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": query, "count": 5, "language": "en", "format": "json"},
            timeout=15
        )
        r.raise_for_status()
        data = r.json()
        if data.get("results"):
            best = data["results"][0]
            parts = [best.get("name", "")]
            if best.get("admin1"): parts.append(best["admin1"])
            if best.get("country"): parts.append(best["country"])
            return float(best["latitude"]), float(best["longitude"]), ", ".join(p for p in parts if p)
    except Exception:
        pass

    try:
        r = requests.get(
            "https://photon.komoot.io/api/",
            params={"q": query, "limit": 3, "lang": "en"},
            timeout=15
        )
        r.raise_for_status()
        data = r.json()
        features = data.get("features", [])
        if features:
            coords = features[0]["geometry"]["coordinates"]
            props  = features[0].get("properties", {})
            name   = ", ".join(p for p in [props.get("name",""), props.get("state",""), props.get("country","")] if p) or query
            return float(coords[1]), float(coords[0]), name
    except Exception:
        pass

    return None, None, None


def parse_manual_coords(text: str):
    text = text.strip().replace("°", "").replace(";", ",")
    parts = text.replace(",", " ").split()
    if len(parts) == 2:
        try:
            return float(parts[0]), float(parts[1])
        except ValueError:
            pass
    return None, None


# ══════════════════════════════════════════════════════════════════════════════
# Solar astronomy
# ══════════════════════════════════════════════════════════════════════════════
def solar_window(lat_deg: float, date: datetime):
    lat  = math.radians(lat_deg)
    doy  = date.timetuple().tm_yday
    B    = math.radians((360 / 365) * (doy - 81))
    decl = math.radians(23.45 * math.sin(B))
    cos_ha = -math.tan(lat) * math.tan(decl)
    if cos_ha < -1: return 0.0, 24.0
    if cos_ha >  1: return 0.0, 0.0
    ha = math.degrees(math.acos(cos_ha))
    return 12.0 - ha / 15.0, 12.0 + ha / 15.0


def apply_solar_mask(preds: np.ndarray, dates: list, lat: float) -> np.ndarray:
    out = preds.copy().astype(float)
    for i, dt in enumerate(dates):
        sr, ss = solar_window(lat, dt)
        h = dt.hour + dt.minute / 60.0
        if not (sr - 0.5 <= h <= ss + 0.5):
            out[i] = 0.0
    return np.clip(out, 0, None)


# ══════════════════════════════════════════════════════════════════════════════
# LSTM models
# ══════════════════════════════════════════════════════════════════════════════
class LSTMPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2,
                            batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class StackedLSTMDayAhead(nn.Module):
    def __init__(self, input_size=1, hidden_sizes=(64, 32, 16), dropout=0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size,      hidden_sizes[0], batch_first=True)
        self.lstm2 = nn.LSTM(hidden_sizes[0], hidden_sizes[1], batch_first=True)
        self.lstm3 = nn.LSTM(hidden_sizes[1], hidden_sizes[2], batch_first=True)
        self.drop  = nn.Dropout(dropout)
        self.fc    = nn.Linear(hidden_sizes[2], 1)

    def forward(self, x):
        o, _ = self.lstm1(x); o = self.drop(o)
        o, _ = self.lstm2(o); o = self.drop(o)
        o, _ = self.lstm3(o)
        return self.fc(o[:, -1, :])


class LSTMMultivariate(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2,
                            batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ══════════════════════════════════════════════════════════════════════════════
# Open-Meteo data fetching
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner=False)
def load_openmeteo_data(lat, lon, start_date, end_date, energy_type="Solar"):
    if energy_type == "Solar":
        hourly_vars = ["shortwave_radiation", "temperature_2m",
                       "relative_humidity_2m", "wind_speed_10m", "cloud_cover"]
    else:
        hourly_vars = ["wind_speed_10m", "wind_gusts_10m", "temperature_2m"]

    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start_date.strftime("%Y-%m-%d") if hasattr(start_date, "strftime") else str(start_date),
        "end_date":   end_date.strftime("%Y-%m-%d")   if hasattr(end_date,   "strftime") else str(end_date),
        "hourly":     ",".join(hourly_vars),
        "timezone":   "auto",
    }

    try:
        r = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params=params, timeout=60
        )
        r.raise_for_status()
        data = r.json()
    except requests.exceptions.Timeout:
        raise RuntimeError("Open-Meteo request timed out. Try a shorter date range.")
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Cannot connect to Open-Meteo. Check your internet connection.")
    except Exception as e:
        raise RuntimeError(f"Open-Meteo error: {e}")

    if "hourly" not in data:
        raise RuntimeError(f"Open-Meteo returned no hourly data: {data.get('reason', 'unknown error')}")

    hourly = data["hourly"]
    times  = pd.to_datetime(hourly["time"])
    df_dict = {"Datetime": times}
    for v in hourly_vars:
        if v in hourly:
            df_dict[v] = np.array(hourly[v], dtype=float)

    df = pd.DataFrame(df_dict).set_index("Datetime").sort_index()
    df = df.ffill().bfill().fillna(0)
    df[df < 0] = 0
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════
def make_seed_sequence_solar(df, seq_len, lat):
    if not len(df):
        return np.zeros(seq_len)
    last_day = df.index[-1].date()
    prev_day = last_day - timedelta(days=1)
    recent   = df[df.index.date >= prev_day]
    vals = []
    for dt, row in recent.iterrows():
        sr, ss = solar_window(lat, dt)
        h = dt.hour + dt.minute / 60.0
        if sr - 0.5 <= h <= ss + 0.5:
            vals.append(float(row["shortwave_radiation"]))

    if len(vals) == 0:
        arr = df["shortwave_radiation"].values[-seq_len:].astype(float)
    else:
        arr = np.array(vals, dtype=float)

    if len(arr) < seq_len:
        arr = np.concatenate([np.zeros(seq_len - len(arr)), arr])

    arr = np.nan_to_num(arr, nan=0.0)
    return arr[-seq_len:]


def make_seed_sequence_wind(df, seq_len):
    if not len(df):
        return np.zeros(seq_len)
    arr = df["wind_speed_10m"].values[-seq_len:].astype(float)
    if len(arr) < seq_len:
        arr = np.concatenate([np.zeros(seq_len - len(arr)), arr])
    return np.nan_to_num(arr, nan=0.0)[-seq_len:]


def make_future_dates(last_dt, n_steps):
    cursor = (last_dt + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    dates  = []
    while len(dates) < n_steps:
        dates.append(cursor)
        cursor += timedelta(hours=1)
    return dates


def normalize_data(data):
    data = np.nan_to_num(data, nan=0.0)
    m = float(np.mean(data))
    s = float(np.std(data))
    s = s if s > 1e-8 else 1.0
    return (data - m) / s, m, s


def normalize_features(df):
    means = df.mean()
    stds  = df.std()
    stds_arr = np.where(stds.values > 1e-8, stds.values, 1.0)
    stds_safe = pd.Series(stds_arr, index=stds.index)
    df_norm   = (df - means) / stds_safe
    df_norm   = df_norm.fillna(0.0)
    return df_norm, means, stds_safe


def create_sequences(data, seq_len):
    s, t = [], []
    for i in range(len(data) - seq_len):
        s.append(data[i:i + seq_len])
        t.append(data[i + seq_len])
    return np.array(s), np.array(t)


def create_sequences_mv(features, target, seq_len):
    s, t = [], []
    for i in range(len(features) - seq_len):
        s.append(features[i:i + seq_len])
        t.append(target[i + seq_len])
    return np.array(s), np.array(t)


def train_model(model, X, y, epochs, pb, lr=0.005):
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    fn    = nn.HuberLoss()
    losses = []
    for e in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(X).squeeze()
        loss = fn(pred, y)

        if torch.isnan(loss):
            raise RuntimeError(
                f"NaN loss at epoch {e+1}. Probable cause: NaN in input tensors. "
                "Check that your date range has enough non-zero data."
            )

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        sched.step()
        losses.append(float(loss.item()))
        pb.progress((e + 1) / epochs)
    return model, losses


def predict_univariate(model, seed, n, m, s):
    model.eval()
    preds = []
    cur = seed.view(-1, 1).clone()
    with torch.no_grad():
        for _ in range(n):
            inp = cur.unsqueeze(0)
            val = float(model(inp).item())
            preds.append(val)
            new_val = torch.tensor([[val]], dtype=cur.dtype)
            cur = torch.cat([cur[1:], new_val], dim=0)
    return np.array(preds) * s + m


def predict_multivariate(model, seed, n, target_m, target_s):
    model.eval()
    preds = []
    cur = seed.clone()
    with torch.no_grad():
        for _ in range(n):
            inp = cur.unsqueeze(0)
            v   = float(model(inp).item())
            preds.append(v)
            new_row = cur[-1].clone()
            new_row[0] = v
            cur = torch.cat([cur[1:], new_row.unsqueeze(0)], dim=0)
    return np.array(preds) * float(target_s) + float(target_m)


# ══════════════════════════════════════════════════════════════════════════════
# Map rendering
# ══════════════════════════════════════════════════════════════════════════════
def render_map(lat, lon, location_name=""):
    st.markdown(
        f"<div style='font-size:.68rem;color:#8888a0;margin-bottom:4px'>"
        f"📌 Lat <b style='color:#f5b432'>{lat:.4f}°</b>&nbsp;|&nbsp;"
        f"Lon <b style='color:#f5b432'>{lon:.4f}°</b>"
        + (f"&nbsp;|&nbsp;<span style='color:#eeeef4'>{location_name[:60]}</span>" if location_name else "")
        + "</div>",
        unsafe_allow_html=True
    )
    map_df = pd.DataFrame({"lat": [lat], "lon": [lon]})

    try:
        st.map(map_df, zoom=5, width="stretch", height=260)
        return
    except TypeError:
        pass
    try:
        st.map(map_df, zoom=5, use_container_width=True, height=260)
        return
    except Exception:
        pass

    bbox_pad = 2.5
    osm_url = (
        f"https://www.openstreetmap.org/export/embed.html"
        f"?bbox={lon-bbox_pad:.4f}%2C{lat-bbox_pad:.4f}%2C{lon+bbox_pad:.4f}%2C{lat+bbox_pad:.4f}"
        f"&layer=mapnik&marker={lat:.6f}%2C{lon:.6f}"
    )
    st.markdown(
        f'<iframe width="100%" height="260" frameborder="0" scrolling="no" '
        f'style="border:1px solid rgba(255,255,255,0.08);border-radius:12px;display:block" '
        f'src="{osm_url}" loading="lazy"></iframe>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Altair theme
# ══════════════════════════════════════════════════════════════════════════════
_DARK_THEME_CFG = {
    "background": "#111120", "view": {"stroke": "transparent"},
    "axis": {"domainColor": "#44445a", "gridColor": "#1f1f32", "tickColor": "#44445a",
             "labelColor": "#8888a0", "titleColor": "#8888a0",
             "labelFont": "Sora", "titleFont": "Sora",
             "labelFontSize": 11, "titleFontSize": 11},
    "legend": {"labelColor": "#8888a0", "titleColor": "#8888a0",
               "labelFont": "Sora", "titleFont": "Sora", "labelFontSize": 11},
    "title": {"color": "#eeeef4", "font": "Sora", "fontSize": 13},
}

try:
    @alt.theme.register("sowi_dark", enable=True)
    def _sowi_dark():
        return alt.theme.ThemeConfig({"config": _DARK_THEME_CFG})
except AttributeError:
    alt.themes.register("sowi_dark", lambda: {"config": _DARK_THEME_CFG})
    alt.themes.enable("sowi_dark")


# ══════════════════════════════════════════════════════════════════════════════
# Helpers de groupby compatibles con pandas 2.x y 3.x
# ══════════════════════════════════════════════════════════════════════════════
def groupby_hour(series: pd.Series) -> pd.Series:
    tmp = pd.DataFrame({"val": series.values, "hour": series.index.hour})
    return tmp.groupby("hour")["val"].mean()


# ══════════════════════════════════════════════════════════════════════════════
# Flat-prediction warning
# ══════════════════════════════════════════════════════════════════════════════
def check_flat_prediction(preds: np.ndarray, label: str = "prediction") -> None:
    rng = preds.max() - preds.min()
    if rng > 0 and preds.std() < 0.05 * (rng + 1e-8):
        st.warning(
            f"⚠️ The {label} looks unusually flat (std ≈ {preds.std():.3f}). "
            "Try more epochs, a longer data range, or Day-Ahead mode."
        )


# ══════════════════════════════════════════════════════════════════════════════
# Solar dashboard
# ══════════════════════════════════════════════════════════════════════════════
def render_solar_dashboard(df, predictions, future_dates, lat):
    preds  = np.clip(predictions, 0, None)
    hp     = int((preds > 50).sum())
    ekwh   = preds.sum() / 1000
    mean_w = preds.mean()
    mean_k = mean_w / 1000.0
    cal    = ("Excellent ☀️" if mean_w >= 300 else
              "Good 🌤️"      if mean_w >= 150 else "Moderate ⛅")

    check_flat_prediction(preds, "solar GHI forecast")

    st.markdown(
        f'<div class="pred-metrics">'
        f'<div class="pm pm-solar"><div class="pm-lbl">🌞 Avg GHI</div>'
        f'<div class="pm-val">{mean_w:.1f}<span class="pm-unit"> W/m²</span><br>'
        f'<span style="font-size:.8rem">({mean_k:.3f} kW/m²)</span></div>'
        f'<span class="pm-tag tag-solar">{cal}</span></div>'
        f'<div class="pm pm-solar"><div class="pm-lbl">📈 Max GHI</div>'
        f'<div class="pm-val">{preds.max():.1f}<span class="pm-unit"> W/m²</span></div>'
        f'<span class="pm-tag tag-solar">Peak forecast</span></div>'
        f'<div class="pm pm-solar"><div class="pm-lbl">⚡ Total Energy</div>'
        f'<div class="pm-val">{ekwh:.3f}<span class="pm-unit"> kWh/m²</span></div>'
        f'<span class="pm-tag tag-solar">Accumulated</span></div>'
        f'<div class="pm pm-solar"><div class="pm-lbl">⏱️ Productive Hours</div>'
        f'<div class="pm-val">{hp}<span class="pm-unit"> h</span></div>'
        f'<span class="pm-tag tag-solar">GHI &gt; 50 W/m²</span></div>'
        f'</div>', unsafe_allow_html=True)

    st.subheader("📊 Last 24h + Forecast")
    if len(df) >= 24:
        tmp = df.tail(24).reset_index()
        tmp.columns = ["Datetime"] + list(tmp.columns[1:])
        last24 = tmp[["Datetime", "shortwave_radiation"]].rename(
            columns={"shortwave_radiation": "GHI"})
        last24["Type"] = "Historical (last 24h)"
        combined = pd.concat([
            last24,
            pd.DataFrame({"Datetime": future_dates, "GHI": preds, "Type": "Forecast"})],
            ignore_index=True)
    else:
        combined = pd.DataFrame({"Datetime": future_dates, "GHI": preds, "Type": "Forecast"})

    st.altair_chart(
        alt.Chart(combined).mark_line(strokeWidth=2).encode(
            x=alt.X("Datetime:T", title="Date / Time (local)"),
            y=alt.Y("GHI:Q", title="GHI (W/m²)", scale=alt.Scale(zero=True)),
            color=alt.Color("Type:N",
                scale=alt.Scale(domain=["Historical (last 24h)", "Forecast"],
                                range=["#8899bb", "#f5b432"]),
                legend=alt.Legend(orient="top")),
            tooltip=["Datetime:T", alt.Tooltip("GHI:Q", format=".1f"), "Type:N"]
        ).properties(height=320).interactive(),
        use_container_width=True)

    st.subheader("📊 Historical GHI")
    df_r = df.reset_index()
    dcol = df_r.columns[0]
    hall = df_r[[dcol, "shortwave_radiation"]].copy()
    hall.columns = ["Datetime", "GHI"]
    mn_d, mx_d = hall["Datetime"].min().date(), hall["Datetime"].max().date()
    c1, c2 = st.columns(2)
    with c1:
        fi = st.date_input("From", value=max(mn_d, mx_d - timedelta(days=3)),
                           min_value=mn_d, max_value=mx_d, key="hd1_solar")
    with c2:
        ff = st.date_input("To", value=mx_d,
                           min_value=mn_d, max_value=mx_d, key="hd2_solar")
    hf = hall[(hall["Datetime"].dt.date >= fi) & (hall["Datetime"].dt.date <= ff)]
    st.altair_chart(
        alt.Chart(hf).mark_line(strokeWidth=1.5, color="#8899bb").encode(
            x=alt.X("Datetime:T", title="Date / Time (local)"),
            y=alt.Y("GHI:Q", title="GHI (W/m²)", scale=alt.Scale(zero=True)),
            tooltip=["Datetime:T", alt.Tooltip("GHI:Q", format=".1f")]
        ).properties(height=280).interactive(),
        use_container_width=True)

    st.subheader("📊 Forecast Analysis")
    pd2 = pd.DataFrame({"Hour": future_dates, "GHI": preds,
                        "Hour_num": [d.hour for d in future_dates]})
    t1, t2, t3 = st.tabs(["🌡️ Irradiance curve", "📅 Avg by hour", "📈 Distribution"])

    with t1:
        sr, ss = solar_window(lat, future_dates[0] if future_dates else datetime.today())
        st.markdown(
            f'<div class="chart-desc"><strong>Hourly forecast.</strong> Solar window: '
            f'<b style="color:#f5b432">{sr:.1f}h – {ss:.1f}h</b> (local time).</div>',
            unsafe_allow_html=True)
        area = alt.Chart(pd2).mark_area(
            line={"color": "#f5b432", "strokeWidth": 2},
            color=alt.Gradient(gradient="linear",
                stops=[alt.GradientStop(color="rgba(245,180,50,0)", offset=0),
                       alt.GradientStop(color="rgba(245,180,50,.30)", offset=1)],
                x1=1, x2=1, y1=1, y2=0)
        ).encode(x=alt.X("Hour:T", title="Local time"),
                 y=alt.Y("GHI:Q", title="GHI (W/m²)", scale=alt.Scale(zero=True)),
                 tooltip=["Hour:T", alt.Tooltip("GHI:Q", format=".1f")])
        ln = alt.Chart(pd2).mark_line(color="#f5b432", strokeWidth=2).encode(
            x="Hour:T", y="GHI:Q")
        st.altair_chart((area + ln).properties(height=300).interactive(),
                        use_container_width=True)

    with t2:
        avg_by_h = pd2.groupby("Hour_num")["GHI"].mean().reset_index()
        avg_by_h.columns = ["Hour of day", "Avg GHI"]
        havg = (pd.DataFrame({"Hour of day": list(range(24))})
                .merge(avg_by_h, on="Hour of day", how="left").fillna(0))
        st.altair_chart(
            alt.Chart(havg).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3,
                                     color="#f5b432").encode(
                x=alt.X("Hour of day:O"),
                y=alt.Y("Avg GHI:Q", title="Avg GHI (W/m²)"),
                tooltip=["Hour of day:O", alt.Tooltip("Avg GHI:Q", format=".1f")]
            ).properties(height=300).interactive(), use_container_width=True)
        psh  = havg["Avg GHI"].sum() / 1000.0
        gmax = havg["Avg GHI"].max()
        st.metric("☀️ Avg Peak Sun Hours (PSH)", f"{psh:.2f} h/day")
        rec = ("≥110% DC peak" if gmax > 800 else
               "100-110% standard" if gmax > 500 else "100% DC")
        st.info(f"**Max avg GHI:** {gmax:.1f} W/m² → Inverter sizing: {rec}")

    with t3:
        pd2d = pd2[pd2["GHI"] > 0]
        if len(pd2d):
            st.altair_chart(
                alt.Chart(pd2d).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3,
                                         color="#8899bb").encode(
                    x=alt.X("GHI:Q", bin=alt.Bin(maxbins=20), title="GHI (W/m²)"),
                    y=alt.Y("count()", title="Hours"),
                    tooltip=[alt.Tooltip("GHI:Q", bin=True), "count()"]
                ).properties(height=300).interactive(), use_container_width=True)
        else:
            st.info("No positive GHI values to display.")

    st.subheader("📋 Solar Hourly Profile")
    hist_h = groupby_hour(df["shortwave_radiation"])
    pred_h = pd2.groupby("Hour_num")["GHI"].mean()
    sr, ss = solar_window(lat, future_dates[0] if future_dates else datetime.today())
    rows = []
    for h in range(24):
        hv = float(hist_h.get(h, 0))
        pv = float(pred_h.get(h, 0))
        av = (hv + pv) / 2 if pv else hv
        period = ("🌙 Night"      if h < sr - .5 or h > ss + .5 else
                  "🌅 Dawn"       if h <= sr + 1 else
                  "☀️ Morning"    if h <= 11 else
                  "🔆 Solar noon" if h <= 13 else
                  "🌤️ Afternoon"  if h <= ss - 1 else "🌇 Twilight")
        rows.append({"Hour": f"{h:02d}:00",
                     "Hist GHI (W/m²)":     round(max(hv, 0), 1),
                     "Forecast GHI (W/m²)": round(max(pv, 0), 1),
                     "Avg (W/m²)":          round(max(av, 0), 1),
                     "Period":              period})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=480, hide_index=True)

    with st.expander("📋 Full forecast table"):
        st.dataframe(pd.DataFrame({
            "Date & Time (local)":  [d.strftime("%Y-%m-%d %H:%M") for d in future_dates],
            "Forecast GHI (W/m²)": preds.round(2)}),
            use_container_width=True, height=300)
    return preds


# ══════════════════════════════════════════════════════════════════════════════
# Wind dashboard
# ══════════════════════════════════════════════════════════════════════════════
def render_wind_dashboard(df, predictions, future_dates):
    GC  = "#22c55e"
    GCd = "rgba(34,197,94,.30)"

    preds         = np.clip(predictions, 0, None)
    avg_speed     = preds.mean()
    max_speed     = preds.max()
    power_density = 0.5 * 1.225 * (preds ** 3)
    avg_power     = power_density.mean()
    total_kwh     = power_density.sum() / 1000

    check_flat_prediction(preds, "wind speed forecast")

    st.markdown(
        f'<div class="pred-metrics">'
        f'<div class="pm pm-wind"><div class="pm-lbl">💨 Avg Wind Speed</div>'
        f'<div class="pm-val">{avg_speed:.2f}<span class="pm-unit"> m/s</span></div>'
        f'<span class="pm-tag tag-wind">Forecast</span></div>'
        f'<div class="pm pm-wind"><div class="pm-lbl">📈 Max Wind Speed</div>'
        f'<div class="pm-val">{max_speed:.2f}<span class="pm-unit"> m/s</span></div>'
        f'<span class="pm-tag tag-wind">Peak</span></div>'
        f'<div class="pm pm-wind"><div class="pm-lbl">⚡ Avg Power Density</div>'
        f'<div class="pm-val">{avg_power:.1f}<span class="pm-unit"> W/m²</span></div>'
        f'<span class="pm-tag tag-wind">0.5·ρ·v³</span></div>'
        f'<div class="pm pm-wind"><div class="pm-lbl">🔋 Total Wind Energy</div>'
        f'<div class="pm-val">{total_kwh:.2f}<span class="pm-unit"> kWh/m²</span></div>'
        f'<span class="pm-tag tag-wind">Over {len(preds)} h</span></div>'
        f'</div>', unsafe_allow_html=True)

    st.subheader("📊 Last 24h + Forecast (Wind Speed)")
    if len(df) >= 24:
        tmp = df.tail(24).reset_index()
        tmp.columns = ["Datetime"] + list(tmp.columns[1:])
        last24 = tmp[["Datetime", "wind_speed_10m"]].rename(
            columns={"wind_speed_10m": "WindSpeed"})
        last24["Type"] = "Historical (last 24h)"
        combined = pd.concat([
            last24,
            pd.DataFrame({"Datetime": future_dates, "WindSpeed": preds, "Type": "Forecast"})],
            ignore_index=True)
    else:
        combined = pd.DataFrame({"Datetime": future_dates, "WindSpeed": preds, "Type": "Forecast"})

    st.altair_chart(
        alt.Chart(combined).mark_line(strokeWidth=2).encode(
            x=alt.X("Datetime:T", title="Date / Time (local)"),
            y=alt.Y("WindSpeed:Q", title="Wind Speed (m/s)", scale=alt.Scale(zero=True)),
            color=alt.Color("Type:N",
                scale=alt.Scale(domain=["Historical (last 24h)", "Forecast"],
                                range=["#8899bb", GC]),
                legend=alt.Legend(orient="top")),
            tooltip=["Datetime:T", alt.Tooltip("WindSpeed:Q", format=".2f"), "Type:N"]
        ).properties(height=320).interactive(),
        use_container_width=True)

    st.subheader("📊 Historical Wind Speed")
    df_r = df.reset_index()
    dcol = df_r.columns[0]
    hall = df_r[[dcol, "wind_speed_10m"]].copy()
    hall.columns = ["Datetime", "WindSpeed"]
    mn_d, mx_d = hall["Datetime"].min().date(), hall["Datetime"].max().date()
    c1, c2 = st.columns(2)
    with c1:
        fi = st.date_input("From", value=max(mn_d, mx_d - timedelta(days=3)),
                           min_value=mn_d, max_value=mx_d, key="hd1_wind")
    with c2:
        ff = st.date_input("To", value=mx_d,
                           min_value=mn_d, max_value=mx_d, key="hd2_wind")
    hf = hall[(hall["Datetime"].dt.date >= fi) & (hall["Datetime"].dt.date <= ff)]
    st.altair_chart(
        alt.Chart(hf).mark_line(strokeWidth=1.5, color="#8899bb").encode(
            x=alt.X("Datetime:T", title="Date / Time (local)"),
            y=alt.Y("WindSpeed:Q", title="Wind Speed (m/s)", scale=alt.Scale(zero=True)),
            tooltip=["Datetime:T", alt.Tooltip("WindSpeed:Q", format=".2f")]
        ).properties(height=280).interactive(),
        use_container_width=True)

    st.subheader("📊 Wind Forecast Analysis")
    pd2 = pd.DataFrame({"Hour": future_dates, "WindSpeed": preds,
                        "Hour_num": [d.hour for d in future_dates]})
    t1, t2, t3 = st.tabs(["🌬️ Speed curve", "📅 Avg by hour", "📈 Distribution"])

    with t1:
        st.markdown(
            '<div class="chart-desc-wind"><strong>Hourly wind speed forecast.</strong> '
            'Useful for turbine output estimation.</div>',
            unsafe_allow_html=True)
        area = alt.Chart(pd2).mark_area(
            line={"color": GC, "strokeWidth": 2},
            color=alt.Gradient(gradient="linear",
                stops=[alt.GradientStop(color="rgba(34,197,94,0)", offset=0),
                       alt.GradientStop(color=GCd, offset=1)],
                x1=1, x2=1, y1=1, y2=0)
        ).encode(x=alt.X("Hour:T", title="Local time"),
                 y=alt.Y("WindSpeed:Q", title="Wind Speed (m/s)", scale=alt.Scale(zero=True)),
                 tooltip=["Hour:T", alt.Tooltip("WindSpeed:Q", format=".2f")])
        ln = alt.Chart(pd2).mark_line(color=GC, strokeWidth=2).encode(
            x="Hour:T", y="WindSpeed:Q")
        st.altair_chart((area + ln).properties(height=300).interactive(),
                        use_container_width=True)

    with t2:
        avg_by_h = pd2.groupby("Hour_num")["WindSpeed"].mean().reset_index()
        avg_by_h.columns = ["Hour of day", "Avg Wind (m/s)"]
        havg = (pd.DataFrame({"Hour of day": list(range(24))})
                .merge(avg_by_h, on="Hour of day", how="left").fillna(0))
        st.altair_chart(
            alt.Chart(havg).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3,
                                     color=GC).encode(
                x=alt.X("Hour of day:O"),
                y=alt.Y("Avg Wind (m/s):Q", title="Avg Wind Speed (m/s)"),
                tooltip=["Hour of day:O", alt.Tooltip("Avg Wind (m/s):Q", format=".2f")]
            ).properties(height=300).interactive(), use_container_width=True)
        st.caption("🟩 Average wind speed by hour of day (forecast period)")

    with t3:
        st.altair_chart(
            alt.Chart(pd2).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3,
                                    color=GC).encode(
                x=alt.X("WindSpeed:Q", bin=alt.Bin(maxbins=20), title="Wind Speed (m/s)"),
                y=alt.Y("count()", title="Hours"),
                tooltip=[alt.Tooltip("WindSpeed:Q", bin=True), "count()"]
            ).properties(height=300).interactive(), use_container_width=True)

    st.subheader("📋 Wind Hourly Profile")
    hist_h = groupby_hour(df["wind_speed_10m"])
    pred_h = pd2.groupby("Hour_num")["WindSpeed"].mean()
    rows = []
    for h in range(24):
        hv = float(hist_h.get(h, 0))
        pv = float(pred_h.get(h, 0))
        rows.append({"Hour": f"{h:02d}:00",
                     "Hist Wind (m/s)":     round(hv, 2),
                     "Forecast Wind (m/s)": round(pv, 2),
                     "Avg (m/s)":           round((hv + pv) / 2 if pv else hv, 2)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=480, hide_index=True)

    with st.expander("📋 Full wind forecast table"):
        st.dataframe(pd.DataFrame({
            "Date & Time (local)":  [d.strftime("%Y-%m-%d %H:%M") for d in future_dates],
            "Wind Speed (m/s)":     preds.round(2),
            "Power Density (W/m²)": power_density.round(1)}),
            use_container_width=True, height=300)
    return preds


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        "<div style='text-align:center;padding:.4rem 0 1rem'>"
        "<div style='font-size:2rem;margin-bottom:4px'>🌞⚡</div>"
        "<div style='font-size:.88rem;font-weight:700;color:#eeeef4'>"
        "<span style='color:#f5b432'>Renewable</span> Forecast</div>"
        "<div style='font-size:.68rem;color:#44445a;margin-top:2px'>Open-Meteo · PyTorch LSTM</div>"
        "</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        "<div style='font-size:.7rem;text-transform:uppercase;letter-spacing:.1em;"
        "color:#44445a;margin-bottom:.5rem'>🔋 Energy Type</div>",
        unsafe_allow_html=True)
    energy_type = st.radio(
        "Energy source", options=["Solar", "Wind"],
        key="energy_type_radio", horizontal=True, label_visibility="collapsed")

    st.markdown("---")
    st.markdown(
        "<div style='font-size:.7rem;text-transform:uppercase;letter-spacing:.1em;"
        "color:#44445a;margin-bottom:.4rem'>📍 Location</div>",
        unsafe_allow_html=True)

    loc_mode = st.radio(
        "Location mode",
        options=["🔍 Search by city", "📐 Manual coordinates"],
        key="loc_mode_radio", label_visibility="collapsed")

    for k, v in [("geo_lat", 6.2500), ("geo_lon", -75.5600), ("geo_name", "Medellín, Antioquia, Colombia")]:
        if k not in st.session_state:
            st.session_state[k] = v

    if loc_mode == "🔍 Search by city":
        city_query = st.text_input(
            "City, region or country",
            value=st.session_state.get("city_query_val", "Medellín, Colombia"),
            placeholder="e.g. Tokyo · Dubai · Berlin, Germany · São Paulo",
            key="city_input")
        st.caption("💡 Works worldwide: Tokyo, Dubai, Nairobi, Sydney, New York…")

        col_search, col_clear = st.columns([3, 1])
        with col_search:
            do_search = st.button("🔍 Search", use_container_width=True, type="primary")
        with col_clear:
            if st.button("✕", use_container_width=True, help="Clear"):
                st.session_state["city_query_val"] = ""
                st.rerun()

        if do_search:
            if not city_query.strip():
                st.error("Please enter a location name.")
            else:
                with st.spinner(f"Searching for '{city_query}'…"):
                    fl, fn, fname = geocode(city_query.strip())
                if fl is not None:
                    st.session_state.update({
                        "geo_lat": fl, "geo_lon": fn, "geo_name": fname,
                        "city_query_val": city_query,
                    })
                    st.success(f"📌 Found! {fname[:60]}")
                    st.rerun()
                else:
                    st.error(
                        "❌ Location not found. Tips:\n"
                        "• Add the country: 'Cali, Colombia'\n"
                        "• Use English names: 'Munich' not 'München'\n"
                        "• Try coordinates mode for remote areas"
                    )
    else:
        coord_paste = st.text_input(
            "Paste coordinates (lat, lon)",
            placeholder="e.g. 6.2500, -75.5600", key="coord_paste")
        if coord_paste.strip():
            pl, pn = parse_manual_coords(coord_paste)
            if pl is not None:
                st.caption(f"✅ Parsed: {pl:.4f}°, {pn:.4f}°")

        new_lat = st.number_input("Latitude  (−90 to 90)",
                                  min_value=-90.0, max_value=90.0,
                                  value=float(st.session_state["geo_lat"]),
                                  step=0.0001, format="%.4f", key="manual_lat")
        new_lon = st.number_input("Longitude  (−180 to 180)",
                                  min_value=-180.0, max_value=180.0,
                                  value=float(st.session_state["geo_lon"]),
                                  step=0.0001, format="%.4f", key="manual_lon")

        if st.button("✅ Apply coordinates", use_container_width=True, type="primary"):
            if coord_paste.strip():
                pl, pn = parse_manual_coords(coord_paste)
                if pl is not None:
                    new_lat, new_lon = pl, pn
            if not (-90 <= new_lat <= 90 and -180 <= new_lon <= 180):
                st.error("Coordinates out of valid range.")
            else:
                st.session_state.update({
                    "geo_lat": new_lat, "geo_lon": new_lon,
                    "geo_name": f"Manual: {new_lat:.4f}°, {new_lon:.4f}°"
                })
                st.success("📐 Coordinates applied!")
                st.rerun()

    if st.session_state.get("geo_name"):
        bc = "#f5b432" if loc_mode == "🔍 Search by city" else "#22c55e"
        bb = "rgba(245,180,50,.08)" if loc_mode == "🔍 Search by city" else "rgba(34,197,94,.08)"
        bd = "rgba(245,180,50,.2)"  if loc_mode == "🔍 Search by city" else "rgba(34,197,94,.2)"
        st.markdown(
            f"<div style='background:{bb};border:1px solid {bd};"
            f"border-radius:8px;padding:.4rem .7rem;font-size:.7rem;color:{bc};"
            f"margin:.3rem 0;line-height:1.5;word-break:break-word'>"
            f"📍 {st.session_state['geo_name'][:100]}</div>",
            unsafe_allow_html=True)

    lat = st.session_state["geo_lat"]
    lon = st.session_state["geo_lon"]

    st.markdown("---")
    st.markdown(
        "<div style='font-size:.7rem;text-transform:uppercase;letter-spacing:.1em;"
        "color:#44445a;margin-bottom:.5rem'>📅 Historical Data Range</div>",
        unsafe_allow_html=True)
    om_min        = datetime(1940, 1, 1).date()
    om_max        = (datetime.today() - timedelta(days=5)).date()
    default_start = max((datetime.today() - timedelta(days=365 * 2)).date(), om_min)
    date_start    = st.date_input("Start date", value=default_start,
                                  min_value=om_min, max_value=om_max, key="ds")
    date_end      = st.date_input("End date", value=om_max,
                                  min_value=om_min, max_value=om_max, key="de")
    if date_start >= date_end:
        st.error("⚠️ Start date must be before end date.")
        st.stop()

    st.markdown("---")
    st.markdown(
        "<div style='font-size:.7rem;text-transform:uppercase;letter-spacing:.1em;"
        "color:#44445a;margin-bottom:.5rem'>⚙️ Model Parameters</div>",
        unsafe_allow_html=True)
    seq_len = st.slider("Time window (hours)", 12, 72, 24,
                        help="24h = buen balance velocidad/varianza en CPU")
    epochs  = st.slider("Training epochs", 20, 150, 40,
                        help="40 epochs es suficiente para LSTM small en CPU. Más de 80 es lento sin GPU.")

    st.markdown("---")
    st.markdown(
        "<div style='font-size:.7rem;text-transform:uppercase;letter-spacing:.1em;"
        "color:#44445a;margin-bottom:.5rem'>🔮 Forecast Mode</div>",
        unsafe_allow_html=True)
    modo = st.radio(
        "Forecast mode",
        options=["Standard", "Day-Ahead", "Multivariable"],
        key="forecast_mode_radio", label_visibility="collapsed",
        help="Standard: rápido · Day-Ahead: stacked LSTM 64→32→16 · Multivariable: features climáticas")

    pred_steps      = 48
    da_hidden_sizes = (64, 32, 16)
    use_temp = use_humidity = use_wind_feat = use_cloud = False

    if modo == "Standard":
        pred_steps = st.slider("Hours to forecast", 24, 120, 48, step=24,
                               help="Max 120h recomendado para LSTM en CPU sin degradación")
    elif modo == "Day-Ahead":
        pred_steps = st.slider("Forecast horizon (hours)", 24, 120, 48, step=24)
        da_hidden  = st.select_slider(
            "Model size",
            options=["Small (64→32→16)", "Medium (128→64→32)"],
            value="Small (64→32→16)",
            help="Large eliminado — OOM en CPU de Streamlit Cloud")
        da_hidden_sizes = {
            "Small (64→32→16)":   (64, 32, 16),
            "Medium (128→64→32)": (128, 64, 32),
        }[da_hidden]
    elif modo == "Multivariable":
        pred_steps = st.slider("Hours to forecast", 24, 120, 48, step=24)
        st.markdown(
            "<div style='font-size:.72rem;color:#8888a0;margin:.4rem 0 .3rem'>"
            "Additional features:</div>", unsafe_allow_html=True)
        use_temp = st.checkbox("🌡️ Temperature", value=True)
        if energy_type == "Solar":
            use_humidity  = st.checkbox("💧 Humidity",    value=True)
            use_wind_feat = st.checkbox("💨 Wind speed",  value=False)
            use_cloud     = st.checkbox("☁️ Cloud cover", value=True)
        else:
            st.info("Temperature + wind gusts used automatically.")

    st.markdown("---")
    st.markdown(
        "<div style='background:rgba(245,180,50,.06);border:1px solid rgba(245,180,50,.15);"
        "border-radius:10px;padding:.6rem .8rem;font-size:.72rem;color:#8888a0;line-height:1.55'>"
        "💡 <b style='color:#f5b432'>Tip:</b> 2 años + ventana 24h + 40 epochs da buena varianza en CPU. "
        "Model Small (64→32→16) es el óptimo para Streamlit Cloud.</div>",
        unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("⚡ Run model", use_container_width=True, type="primary")


# ══════════════════════════════════════════════════════════════════════════════
# Hero
# ══════════════════════════════════════════════════════════════════════════════
hero_color    = "#f5b432" if energy_type == "Solar" else "#22c55e"
hero_subtitle = ("Solar irradiance prediction with LSTM" if energy_type == "Solar"
                 else "Wind speed forecasting with LSTM")
st.markdown(
    f'<div class="page-hero">'
    f'<div class="page-eyebrow"><div class="ldot"></div>'
    f'{energy_type} Forecast &nbsp;·&nbsp; Open-Meteo · LSTM</div>'
    f'<h1 class="page-title"><span style="color:{hero_color}">'
    f'{energy_type}</span> Energy Forecast</h1>'
    f'<p class="page-sub" style="font-weight:500;color:#eeeef4;font-size:1rem">'
    f'{hero_subtitle}</p>'
    f'<p class="page-sub">Real-time data · Open-Meteo archive · PyTorch LSTM ⚡</p>'
    f'</div><div class="wave-sep"></div>',
    unsafe_allow_html=True)

col_map, col_info = st.columns([3, 2], gap="large")
with col_map:
    st.markdown(
        "<div style='font-size:.7rem;text-transform:uppercase;letter-spacing:.1em;"
        "color:#44445a;margin-bottom:.6rem'>🗺️ Selected Location</div>",
        unsafe_allow_html=True)
    render_map(lat, lon, st.session_state.get("geo_name", ""))

with col_info:
    accent = "#f5b432" if energy_type == "Solar" else "#22c55e"
    st.markdown(
        "<div style='font-size:.7rem;text-transform:uppercase;letter-spacing:.1em;"
        "color:#44445a;margin-bottom:.6rem'>ℹ️ Model Info</div>",
        unsafe_allow_html=True)

    if energy_type == "Solar":
        sr_now, ss_now = solar_window(lat, datetime.today())
        rise_str = f"{int(sr_now):02d}:{int((sr_now % 1) * 60):02d}"
        set_str  = f"{int(ss_now):02d}:{int((ss_now % 1) * 60):02d}"
        sun_rows = (
            f"<div style='display:flex;justify-content:space-between;font-size:.78rem'>"
            f"<span style='color:#44445a'>Sunrise</span>"
            f"<span style='color:{accent};font-weight:600'>{rise_str} h</span></div>"
            f"<div style='display:flex;justify-content:space-between;font-size:.78rem'>"
            f"<span style='color:#44445a'>Sunset</span>"
            f"<span style='color:{accent};font-weight:600'>{set_str} h</span></div>"
        )
    else:
        sun_rows = ""

    loc_label = (st.session_state.get("geo_name", "") or f"{lat:.4f}°, {lon:.4f}°")[:42]
    info_rows = [
        ("Location",    loc_label,                              "#eeeef4"),
        ("Energy type", energy_type,                            accent),
        ("Mode",        modo,                                   accent),
        ("Horizon",     str(pred_steps) + " h",                "#eeeef4"),
        ("Window",      str(seq_len) + " h",                   "#eeeef4"),
        ("Data range",  str(date_start) + " → " + str(date_end), "#eeeef4"),
    ]
    rows_html = "".join(
        f"<div style='display:flex;justify-content:space-between;font-size:.78rem'>"
        f"<span style='color:#44445a'>{lbl}</span>"
        f"<span style='color:{col};font-weight:600'>{val}</span></div>"
        for lbl, val, col in info_rows)
    source_row = (
        "<div style='display:flex;justify-content:space-between;font-size:.78rem'>"
        "<span style='color:#44445a'>Source</span>"
        "<span style='color:#8899bb;font-weight:600'>Open-Meteo archive</span></div>"
    )
    st.markdown(
        f"<div style='background:var(--surface);border:1px solid var(--bd);"
        f"border-radius:14px;padding:1.1rem 1.2rem;line-height:1.7'>"
        f"<div style='font-size:.8rem;color:#8888a0;margin-bottom:.8rem'>"
        f"Set parameters and press <b style='color:{accent}'>⚡ Run model</b>.</div>"
        f"<div style='display:grid;gap:6px'>"
        f"{rows_html}{sun_rows}{source_row}"
        f"</div></div>",
        unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Main execution
# ══════════════════════════════════════════════════════════════════════════════
if run:
    st.session_state["modelo_ejecutado"] = False

    with st.spinner("🛰️ Fetching historical data from Open-Meteo…"):
        st.caption(f"📅 {date_start} → {date_end}  |  Lat {lat:.4f}° Lon {lon:.4f}°")
        try:
            df = load_openmeteo_data(lat, lon, date_start, date_end, energy_type)
        except RuntimeError as e:
            st.error(str(e))
            st.stop()

    if len(df) < seq_len + 10:
        st.error(
            f"Not enough data ({len(df)} rows). "
            "Please extend the date range or reduce the time window."
        )
        st.stop()

    st.success(f"✅ {len(df):,} records loaded  ·  {df.index.min().date()} → {df.index.max().date()}")

    with st.expander("🔍 Data preview (last 72 h local time)"):
        st.dataframe(df.tail(72), use_container_width=True)

    future_dates = make_future_dates(df.index[-1], pred_steps)

    # ── Training ──────────────────────────────────────────────────────────────
    if energy_type == "Solar":
        target_col = "shortwave_radiation"

        if modo == "Standard":
            st.markdown(
                '<div class="status-banner">⚡ Standard Mode '
                '<span class="meta">· LSTM 2-layer hidden=64</span></div>',
                unsafe_allow_html=True)
            with st.spinner("🧠 Training…"):
                data  = df[target_col].values.ravel()
                dn, m, s = normalize_data(data)
                X, y  = create_sequences(dn, seq_len)
                X = torch.FloatTensor(X).view(-1, seq_len, 1)
                y = torch.FloatTensor(y)
                model = LSTMPredictor(hidden_size=64)
                pb    = st.progress(0)
                try:
                    model, _ = train_model(model, X, y, epochs, pb)
                except RuntimeError as e:
                    st.error(str(e)); st.stop()
            st.success("✅ Model trained")
            seed_n = (make_seed_sequence_solar(df, seq_len, lat) - m) / s
            predictions = predict_univariate(
                model, torch.FloatTensor(seed_n), pred_steps, m, s)

        elif modo == "Day-Ahead":
            st.markdown(
                '<div class="status-banner">🌅 Day-Ahead '
                f'<span class="meta">· Stacked LSTM {da_hidden_sizes}</span></div>',
                unsafe_allow_html=True)
            with st.spinner("🧠 Training stacked LSTM…"):
                data  = df[target_col].values.ravel()
                dn, m, s = normalize_data(data)
                X, y  = create_sequences(dn, seq_len)
                X = torch.FloatTensor(X).view(-1, seq_len, 1)
                y = torch.FloatTensor(y)
                model = StackedLSTMDayAhead(input_size=1, hidden_sizes=da_hidden_sizes)
                pb    = st.progress(0)
                try:
                    model, losses = train_model(model, X, y, epochs, pb)
                except RuntimeError as e:
                    st.error(str(e)); st.stop()
            st.success("✅ Stacked LSTM trained")
            with st.expander("📉 Training loss"):
                st.altair_chart(
                    alt.Chart(pd.DataFrame({"Epoch": range(1, epochs + 1), "Loss": losses}))
                    .mark_line(color="#f5b432", strokeWidth=2)
                    .encode(x="Epoch:Q", y=alt.Y("Loss:Q", title="Huber Loss"))
                    .properties(height=200).interactive(),
                    use_container_width=True)
            seed_n = (make_seed_sequence_solar(df, seq_len, lat) - m) / s
            predictions = predict_univariate(
                model, torch.FloatTensor(seed_n), pred_steps, m, s)

        elif modo == "Multivariable":
            feature_cols = [target_col]
            if use_temp:      feature_cols.append("temperature_2m")
            if use_humidity:  feature_cols.append("relative_humidity_2m")
            if use_wind_feat: feature_cols.append("wind_speed_10m")
            if use_cloud:     feature_cols.append("cloud_cover")
            feature_cols = [c for c in feature_cols if c in df.columns]
            if len(feature_cols) < 2:
                st.warning("⚠️ Select at least one additional feature.")
                st.stop()
            st.markdown(
                f'<div class="status-banner">🔬 Multivariable '
                f'<span class="meta">· {", ".join(feature_cols)}</span></div>',
                unsafe_allow_html=True)
            with st.spinner("🧠 Training multivariable model…"):
                df_f = df[feature_cols].copy()
                df_n, medias, desv = normalize_features(df_f)
                fa = df_n.values.astype(np.float32)
                ta = df_n[target_col].values.astype(np.float32)
                X, y = create_sequences_mv(fa, ta, seq_len)
                X = torch.FloatTensor(X)
                y = torch.FloatTensor(y)
                model = LSTMMultivariate(input_size=len(feature_cols), hidden_size=64)
                pb    = st.progress(0)
                try:
                    model, _ = train_model(model, X, y, epochs, pb)
                except RuntimeError as e:
                    st.error(str(e)); st.stop()
            st.success("✅ Multivariable model trained")
            predictions = predict_multivariate(
                model,
                torch.FloatTensor(fa[-seq_len:]),
                pred_steps,
                float(medias[target_col]),
                float(desv[target_col]))

        min_len = min(len(predictions), len(future_dates))
        predictions, future_dates = predictions[:min_len], future_dates[:min_len]
        predictions = apply_solar_mask(predictions, future_dates, lat)

    else:  # Wind
        target_col = "wind_speed_10m"

        if modo == "Standard":
            st.markdown(
                '<div class="status-banner">⚡ Standard Mode (Wind) '
                '<span class="meta">· LSTM 2-layer hidden=64</span></div>',
                unsafe_allow_html=True)
            with st.spinner("🧠 Training wind model…"):
                data  = df[target_col].values.ravel()
                dn, m, s = normalize_data(data)
                X, y  = create_sequences(dn, seq_len)
                X = torch.FloatTensor(X).view(-1, seq_len, 1)
                y = torch.FloatTensor(y)
                model = LSTMPredictor(hidden_size=64)
                pb    = st.progress(0)
                try:
                    model, _ = train_model(model, X, y, epochs, pb)
                except RuntimeError as e:
                    st.error(str(e)); st.stop()
            st.success("✅ Wind model trained")
            seed_n = (make_seed_sequence_wind(df, seq_len) - m) / s
            predictions = predict_univariate(
                model, torch.FloatTensor(seed_n), pred_steps, m, s)

        elif modo == "Day-Ahead":
            st.markdown(
                '<div class="status-banner">🌅 Day-Ahead (Wind) '
                f'<span class="meta">· Stacked LSTM {da_hidden_sizes}</span></div>',
                unsafe_allow_html=True)
            with st.spinner("🧠 Training stacked LSTM for wind…"):
                data  = df[target_col].values.ravel()
                dn, m, s = normalize_data(data)
                X, y  = create_sequences(dn, seq_len)
                X = torch.FloatTensor(X).view(-1, seq_len, 1)
                y = torch.FloatTensor(y)
                model = StackedLSTMDayAhead(input_size=1, hidden_sizes=da_hidden_sizes)
                pb    = st.progress(0)
                try:
                    model, losses = train_model(model, X, y, epochs, pb)
                except RuntimeError as e:
                    st.error(str(e)); st.stop()
            st.success("✅ Stacked LSTM trained")
            with st.expander("📉 Training loss"):
                st.altair_chart(
                    alt.Chart(pd.DataFrame({"Epoch": range(1, epochs + 1), "Loss": losses}))
                    .mark_line(color="#22c55e", strokeWidth=2)
                    .encode(x="Epoch:Q", y=alt.Y("Loss:Q", title="Huber Loss"))
                    .properties(height=200).interactive(),
                    use_container_width=True)
            seed_n = (make_seed_sequence_wind(df, seq_len) - m) / s
            predictions = predict_univariate(
                model, torch.FloatTensor(seed_n), pred_steps, m, s)

        elif modo == "Multivariable":
            feature_cols = [target_col]
            if use_temp and "temperature_2m" in df.columns:
                feature_cols.append("temperature_2m")
            if "wind_gusts_10m" in df.columns:
                feature_cols.append("wind_gusts_10m")
            feature_cols = [c for c in feature_cols if c in df.columns]
            if len(feature_cols) < 2:
                st.warning("⚠️ No additional features available.")
                st.stop()
            st.markdown(
                f'<div class="status-banner">🔬 Multivariable (Wind) '
                f'<span class="meta">· {", ".join(feature_cols)}</span></div>',
                unsafe_allow_html=True)
            with st.spinner("🧠 Training multivariable wind model…"):
                df_f = df[feature_cols].copy()
                df_n, medias, desv = normalize_features(df_f)
                fa = df_n.values.astype(np.float32)
                ta = df_n[target_col].values.astype(np.float32)
                X, y = create_sequences_mv(fa, ta, seq_len)
                X = torch.FloatTensor(X)
                y = torch.FloatTensor(y)
                model = LSTMMultivariate(input_size=len(feature_cols), hidden_size=64)
                pb    = st.progress(0)
                try:
                    model, _ = train_model(model, X, y, epochs, pb)
                except RuntimeError as e:
                    st.error(str(e)); st.stop()
            st.success("✅ Multivariable wind model trained")
            predictions = predict_multivariate(
                model,
                torch.FloatTensor(fa[-seq_len:]),
                pred_steps,
                float(medias[target_col]),
                float(desv[target_col]))

        predictions = np.clip(predictions, 0, None)
        min_len     = min(len(predictions), len(future_dates))
        predictions, future_dates = predictions[:min_len], future_dates[:min_len]

    st.session_state.update({
        "predictions":      predictions,
        "future_dates":     future_dates,
        "df":               df,
        "lat":              lat,
        "lon":              lon,
        "date_start":       str(date_start),
        "date_end":         str(date_end),
        "modo":             modo,
        "energy_type":      energy_type,
        "modelo_ejecutado": True,
    })


# ══════════════════════════════════════════════════════════════════════════════
# Results
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.get("modelo_ejecutado"):
    res_et = st.session_state["energy_type"]
    st.markdown("---")
    st.subheader("📈 Forecast Results")

    if res_et == "Solar":
        render_solar_dashboard(
            st.session_state["df"], st.session_state["predictions"],
            st.session_state["future_dates"], st.session_state["lat"])
    else:
        render_wind_dashboard(
            st.session_state["df"], st.session_state["predictions"],
            st.session_state["future_dates"])

    # ── CTA banner → Sowi AI (FIXED: direct st.switch_page, no sidebar message) ──
    sowi_color = "#f5b432" if res_et == "Solar" else "#22c55e"
    st.markdown(
        f"<div style='background:linear-gradient(135deg,rgba(245,180,50,.08),rgba(34,197,94,.06));"
        f"border:1px solid rgba(245,180,50,.2);border-radius:16px;"
        f"padding:1.4rem 1.6rem;text-align:center;margin:1.8rem 0 .5rem'>"
        f"<div style='font-size:1.5rem;margin-bottom:.4rem'>🤖</div>"
        f"<div style='font-size:1rem;font-weight:700;color:#eeeef4;margin-bottom:.3rem'>"
        f"Want to analyze these results further?</div>"
        f"<div style='font-size:.85rem;color:#8888a0'>"
        f"Ask <b style='color:{sowi_color}'>Sowi AI</b> — "
        f"answers in ≤ 400 tokens, focused on your forecast.</div>"
        f"</div>", unsafe_allow_html=True)

    _, cc, _ = st.columns([1, 2, 1])
    with cc:
        # ── FIXED: direct navigation via st.switch_page — no fallback message ──
        st.page_link(
            _SOWI_AI_PAGE,
            label="🤖 Ask Sowi AI →",
            use_container_width=True,
        )

    st.markdown(
        "<div class='footer'>⚡ Renewable Energy Forecast &nbsp;·&nbsp; "
        "Open-Meteo &nbsp;·&nbsp; PyTorch LSTM &nbsp;·&nbsp; Sowi AI 🌊</div>",
        unsafe_allow_html=True)

else:
    icon = "🌞" if energy_type == "Solar" else "💨"
    st.markdown(
        f"<div style='text-align:center;padding:2.5rem 1rem;background:var(--surface);"
        f"border:1px dashed var(--bd-hi);border-radius:20px;margin-top:1rem'>"
        f"<div style='font-size:2.2rem;margin-bottom:.6rem;animation:fi 3s ease-in-out infinite'>"
        f"{icon}⚡</div>"
        f"<div style='font-size:1rem;font-weight:600;color:#eeeef4;margin-bottom:.3rem'>"
        f"Ready to forecast</div>"
        f"<div style='font-size:.85rem;color:#8888a0;max-width:380px;margin:0 auto'>"
        f"Configure the parameters in the sidebar and press "
        f"<b style='color:#f5b432'>⚡ Run model</b>.</div>"
        f"</div>"
        f"<style>@keyframes fi{{0%,100%{{transform:translateY(0)}}50%{{transform:translateY(-7px)}}}}</style>",
        unsafe_allow_html=True)
