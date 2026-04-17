"""
Sowi AI Analyst — pages/Sowi_AI_Analyst.py
English version · solar & wind expert · dynamic prompts
Fixed: page_link direct nav, back-to-forecast button, API error handling,
       session state safety, model name
"""

import streamlit as st
import numpy as np
from datetime import datetime

# ── Navigation: set this to the exact filename of your main forecast page ──
# Examples: "Energy_Forecast.py" | "app.py" | "Home.py"
_FORECAST_PAGE = "Energy_Forecast.py"   # ← adjust if your file has a different name


def _nav_to_forecast(label: str = "⚡ Back to Forecast", key: str = "nav_back",
                     use_container_width: bool = False):
    """
    Robust cross-version navigation button.
    Tries st.switch_page() first; falls back to st.page_link() if available.
    Never raises StreamlitPageNotFoundError to the user — shows a clear message instead.
    """
    if st.button(label, key=key, use_container_width=use_container_width):
        try:
            st.switch_page(_FORECAST_PAGE)
        except Exception:
            st.info(
                f"Navigate to **{_FORECAST_PAGE}** from the sidebar. "
                f"(Tip: update `_FORECAST_PAGE` at the top of this file to match your exact filename.)"
            )

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Sowi · Energy Analyst",
    page_icon="🌞💨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg0:#06060e; --bg1:#0c0c18; --bg2:#111120;
    --surface:#181828; --surface2:#1f1f32; --surface3:#26263c;
    --bd:rgba(255,255,255,0.06); --bd-hi:rgba(255,255,255,0.11);
    --bd-focus:rgba(245,180,50,0.45);
    --amber:#f5b432; --amber-dim:rgba(245,180,50,0.10); --amber-glow:rgba(245,180,50,0.18);
    --blue:#3b82f6; --blue-dim:rgba(59,130,246,0.10);
    --slate:#8899bb; --slate-dim:rgba(136,153,187,0.10);
    --ok:#3ecf8e; --ok-dim:rgba(62,207,142,0.10);
    --warn:#f5a623; --warn-dim:rgba(245,166,35,0.10);
    --t1:#eeeef4; --t2:#8888a0; --t3:#44445a;
    --font:'Sora',system-ui,sans-serif; --mono:'JetBrains Mono',monospace;
    --r:12px; --rl:20px; --rxl:26px; --pill:100px;
}
html,body,[class*="css"] { font-family:var(--font) !important; }
.stApp { background:var(--bg0) !important; }
#MainMenu,footer,header { visibility:hidden; }
.block-container { padding:0 2rem 3rem !important; max-width:1040px !important; margin:0 auto !important; }
::-webkit-scrollbar { width:4px; } ::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:var(--bd-hi); border-radius:4px; }

/* BACK BUTTON */
.back-nav { margin-bottom: 1rem; }
[data-testid="stPageLink"] a {
    display:inline-flex !important; align-items:center !important; gap:6px !important;
    background:var(--surface) !important; border:1px solid var(--bd-hi) !important;
    color:var(--t2) !important; font-family:var(--font) !important;
    font-size:.8rem !important; font-weight:500 !important;
    border-radius:var(--pill) !important; padding:6px 14px !important;
    text-decoration:none !important; transition:all .18s !important;
}
[data-testid="stPageLink"] a:hover {
    background:var(--amber-dim) !important; border-color:rgba(245,180,50,.3) !important;
    color:var(--amber) !important; transform:translateX(-2px) !important;
}

/* HERO */
.hero { padding:2rem 0 1.6rem; text-align:center; position:relative; overflow:hidden; }
.hero::before { content:''; position:absolute; top:-60px; left:50%; transform:translateX(-50%);
    width:700px; height:260px;
    background:radial-gradient(ellipse,rgba(245,180,50,0.06) 0%,transparent 70%);
    animation:halo 7s ease-in-out infinite alternate; pointer-events:none; }
@keyframes halo { from{opacity:.5;transform:translateX(-50%) scaleX(.9)} to{opacity:1;transform:translateX(-50%) scaleX(1.1)} }
.eyebrow { display:inline-flex; align-items:center; gap:7px;
    border:1px solid rgba(245,180,50,0.2); background:rgba(245,180,50,0.06);
    color:var(--amber); font-size:0.67rem; font-weight:600; letter-spacing:.13em;
    text-transform:uppercase; padding:5px 15px; border-radius:var(--pill); margin-bottom:1.1rem; }
.ldot { width:5px; height:5px; background:var(--amber); border-radius:50%; animation:ldot 2.4s ease-in-out infinite; }
@keyframes ldot { 0%,100%{opacity:1;box-shadow:0 0 0 0 rgba(245,180,50,.7)} 50%{opacity:.4;box-shadow:0 0 0 5px rgba(245,180,50,0)} }
.hero-title { font-size:clamp(2.2rem,5.5vw,3.4rem); font-weight:700; color:var(--t1);
    letter-spacing:-.045em; line-height:1.06; margin:0 0 .6rem; position:relative; z-index:1; }
.hero-title .hl { color:var(--amber); }
.hero-sub { font-size:.98rem; color:var(--t2); font-weight:300; position:relative; z-index:1; }
.wave-sep { height:1px; background:linear-gradient(90deg,transparent 0%,rgba(245,180,50,.25) 35%,rgba(136,153,187,.2) 65%,transparent 100%);
    margin:.3rem 0 2rem; position:relative; overflow:hidden; }
.wave-sep::after { content:''; position:absolute; top:0; left:-60%; width:40%; height:100%;
    background:linear-gradient(90deg,transparent,rgba(245,180,50,.55),transparent);
    animation:wave-scan 5s linear infinite; }
@keyframes wave-scan { to{left:160%} }

/* CONTEXT BANNER */
.ctx-banner { display:flex; align-items:center; gap:10px; background:var(--amber-dim);
    border:1px solid rgba(245,180,50,.18); border-radius:var(--r);
    padding:.65rem 1rem; margin-bottom:1.4rem; font-size:.8rem; color:var(--amber); font-weight:500; flex-wrap:wrap; }
.ctx-meta { color:var(--t2); font-weight:400; margin-left:2px; }

/* METRIC CARDS */
.metrics-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(148px,1fr)); gap:9px; margin-bottom:1.8rem; }
.mc { background:var(--surface); border:1px solid var(--bd); border-radius:var(--r);
    padding:.9rem 1rem .8rem; position:relative; overflow:hidden;
    transition:border-color .2s,transform .18s; cursor:default; }
.mc:hover { border-color:var(--bd-hi); transform:translateY(-2px); }
.mc::before { content:''; position:absolute; top:0; left:0; right:0; height:2px; border-radius:var(--r) var(--r) 0 0; }
.mc-solar::before{background:var(--amber)} .mc-wind::before{background:var(--blue)}
.mc-solar:hover{box-shadow:0 0 18px var(--amber-glow)} .mc-wind:hover{box-shadow:0 0 18px var(--blue-dim)}
.mc-lbl{font-size:.63rem;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:var(--t3);margin-bottom:.35rem}
.mc-val{font-size:1.35rem;font-weight:700;color:var(--t1);line-height:1}
.mc-unit{font-size:.68rem;color:var(--t2);font-weight:400}
.mc-tag{display:inline-block;font-size:.62rem;font-weight:600;padding:2px 8px;border-radius:var(--pill);margin-top:4px}
.tag-solar{background:var(--amber-dim);color:var(--amber)} .tag-wind{background:var(--blue-dim);color:var(--blue)}
.tag-ok{background:var(--ok-dim);color:var(--ok)} .tag-warn{background:var(--warn-dim);color:var(--warn)}

/* CHIPS */
.chips-lbl{font-size:.64rem;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:var(--t3);margin-bottom:.6rem}
.stButton>button:not([kind="primary"]) {
    background:var(--surface) !important; border:1px solid var(--bd-hi) !important;
    color:var(--t2) !important; font-family:var(--font) !important; font-size:.78rem !important;
    border-radius:var(--pill) !important; transition:all .18s !important; white-space:nowrap !important; }
.stButton>button:not([kind="primary"]):hover {
    background:var(--amber-dim) !important; border-color:rgba(245,180,50,.3) !important;
    color:var(--amber) !important; transform:translateY(-1px) !important; }

/* CHAT SHELL */
.chat-shell { background:var(--bg1); border:1px solid var(--bd); border-radius:var(--rxl);
    overflow:hidden; box-shadow:0 1px 0 rgba(255,255,255,.04) inset,0 24px 60px rgba(0,0,0,.55); margin-bottom:2rem; }
.chat-topbar { display:flex; align-items:center; gap:9px; padding:.85rem 1.3rem;
    background:rgba(8,8,18,.85); border-bottom:1px solid var(--bd); }
.tls{display:flex;gap:5px;margin-right:4px} .tl{width:10px;height:10px;border-radius:50%}
.tl-r{background:#ff5f57} .tl-y{background:#febc2e} .tl-g{background:#28c840}
.topbar-name{font-size:.82rem;font-weight:600;color:var(--t1)}
.topbar-model{margin-left:auto;font-size:.63rem;font-family:var(--mono);color:var(--t3);
    background:var(--surface2);border:1px solid var(--bd);padding:2px 9px;border-radius:var(--pill)}

/* MESSAGES */
.chat-msgs{padding:1.3rem;min-height:280px}
.mw{display:flex;align-items:flex-start;gap:9px;margin-bottom:1.15rem;
    animation:bi .24s cubic-bezier(.34,1.56,.64,1)}
@keyframes bi{from{opacity:0;transform:translateY(9px) scale(.97)}to{opacity:1;transform:translateY(0) scale(1)}}
.mw.usr{flex-direction:row-reverse}
.av{width:30px;height:30px;border-radius:50%;display:flex;align-items:center;
    justify-content:center;font-size:.72rem;font-weight:600;flex-shrink:0}
.av-sowi{background:linear-gradient(135deg,rgba(245,180,50,.22),rgba(59,130,246,.18));
    border:1px solid rgba(245,180,50,.28);box-shadow:0 0 9px rgba(245,180,50,.12)}
.av-usr{background:var(--surface3);border:1px solid var(--bd-hi)}
.bbl{max-width:73%;padding:.72rem .95rem;border-radius:var(--r);font-size:.875rem;line-height:1.65;word-break:break-word}
.bbl-sowi{background:var(--surface);border:1px solid var(--bd);color:var(--t1);border-bottom-left-radius:4px}
.bbl-usr{background:linear-gradient(135deg,rgba(245,180,50,.12),rgba(245,180,50,.06));
    border:1px solid rgba(245,180,50,.18);color:var(--t1);border-bottom-right-radius:4px}
.bbl-sowi strong{color:var(--amber);font-weight:600}
.bbl-sowi em{color:var(--slate);font-style:normal}
.bbl-sowi code{font-family:var(--mono);font-size:.79rem;background:var(--amber-dim);
    color:var(--amber);padding:1px 6px;border-radius:4px}
.ts-lbl{font-size:.63rem;color:var(--t3);margin-top:3px;padding:0 2px}
.mw.usr .ts-lbl{text-align:right}

/* TYPING */
.typing-row{display:flex;align-items:center;gap:9px;padding-bottom:.5rem}
.dots{display:flex;gap:4px}
.dots span{width:6px;height:6px;background:var(--amber);border-radius:50%;
    animation:dw 1.3s ease-in-out infinite;box-shadow:0 0 5px var(--amber-glow)}
.dots span:nth-child(2){animation-delay:.18s} .dots span:nth-child(3){animation-delay:.36s}
@keyframes dw{0%,80%,100%{transform:scale(.65);opacity:.35}40%{transform:scale(1.1);opacity:1}}
.typing-lbl{font-size:.7rem;color:var(--t3);font-style:italic;font-family:var(--mono)}

/* INPUT ZONE */
.input-zone{border-top:1px solid var(--bd);padding:.95rem 1.3rem;background:rgba(4,4,12,.65)}
.stTextArea textarea {
    background:var(--surface2) !important; border:1px solid var(--bd-hi) !important;
    border-radius:var(--r) !important; color:var(--t1) !important;
    font-family:var(--font) !important; font-size:.875rem !important;
    line-height:1.55 !important; caret-color:var(--amber) !important;
    resize:none !important; transition:border-color .2s,box-shadow .2s !important; }
.stTextArea textarea:focus {
    border-color:var(--bd-focus) !important;
    box-shadow:0 0 0 3px var(--amber-dim) !important; outline:none !important; }
.stTextArea textarea::placeholder { color:var(--t3) !important; }
.stButton>button[kind="primary"] {
    background:linear-gradient(135deg,rgba(245,180,50,.18),rgba(245,180,50,.08)) !important;
    border:1px solid rgba(245,180,50,.38) !important; color:var(--amber) !important;
    font-family:var(--font) !important; font-weight:600 !important; font-size:.84rem !important;
    border-radius:var(--r) !important; transition:all .18s !important; width:100% !important; }
.stButton>button[kind="primary"]:hover {
    background:linear-gradient(135deg,rgba(245,180,50,.28),rgba(245,180,50,.14)) !important;
    box-shadow:0 0 16px var(--amber-glow) !important; transform:translateY(-1px) !important; }

/* SIDEBAR */
[data-testid="stSidebar"] { background:var(--bg2) !important; border-right:1px solid var(--bd) !important; }
[data-testid="stSidebarContent"] { padding-top:1.5rem !important; }

/* EMPTY STATE */
.empty-state { text-align:center; padding:3.8rem 2rem; background:var(--bg1);
    border:1px dashed var(--bd-hi); border-radius:var(--rxl); margin:1rem 0; }
.empty-icon{font-size:2.6rem;margin-bottom:.8rem;animation:fi 3.2s ease-in-out infinite}
@keyframes fi{0%,100%{transform:translateY(0)}50%{transform:translateY(-8px)}}
.empty-title{font-size:1.15rem;font-weight:600;color:var(--t1);margin-bottom:.4rem}
.empty-sub{font-size:.86rem;color:var(--t2);max-width:360px;margin:0 auto}

.footer{text-align:center;margin-top:2rem;font-size:.65rem;color:var(--t3);letter-spacing:.07em}
.stSpinner>div{border-top-color:var(--amber) !important}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _ts() -> str:
    return datetime.now().strftime("%H:%M")


def _render_msg(role: str, content: str):
    wrap = "mw usr" if role == "user" else "mw"
    bbl  = "bbl bbl-usr" if role == "user" else "bbl bbl-sowi"
    av_c = "av av-usr" if role == "user" else "av av-sowi"
    icon = "👤" if role == "user" else "🌞💨"
    st.markdown(f"""
    <div class="{wrap}">
      <div class="{av_c}">{icon}</div>
      <div>
        <div class="{bbl}">{content}</div>
        <div class="ts-lbl">{_ts()}</div>
      </div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  DYNAMIC EXPERT PROMPT (Solar / Wind)
# ══════════════════════════════════════════════════════════════════════════════

def _build_expert_prompt(energy_type: str, preds, future_dates, lat, lon, years, modo) -> str:
    arr = np.clip(np.asarray(preds, dtype=float), 0, None)
    n   = len(future_dates)

    avg_val  = float(arr.mean()) if n > 0 else 0.0
    max_val  = float(arr.max())  if n > 0 else 0.0
    productive_hrs = int((arr > 0.5).sum())

    by_hour: dict = {}
    for d, v in zip(future_dates, arr):
        by_hour.setdefault(d.hour, []).append(v)

    best_h     = max(by_hour, key=lambda h: np.mean(by_hour[h])) if by_hour else 12
    best_h_avg = float(np.mean(by_hour.get(best_h, [0])))

    dias: dict = {}
    for d, v in zip(future_dates, arr):
        dias.setdefault(d.strftime("%Y-%m-%d"), []).append(v)
    dias_energy = {k: sum(vs) / 1000 for k, vs in dias.items()}
    best_day  = max(dias_energy, key=dias_energy.get) if dias_energy else "N/A"
    worst_day = min(dias_energy, key=dias_energy.get) if dias_energy else "N/A"

    total_energy = float(arr.sum()) / 1000.0

    if energy_type == "Solar":
        ghi_mean  = avg_val
        ghi_max   = max_val
        e_kwh     = total_energy
        hrs_prod  = productive_hrs
        hrs_peak  = int((arr > 600).sum())
        ghi_pos   = arr[arr > 0]
        ghi_min   = float(ghi_pos.min()) if len(ghi_pos) > 0 else 0.0
        pico_idx  = int(arr.argmax()) if n > 0 else 0
        pico_hora = future_dates[pico_idx].strftime("%Y-%m-%d %H:%M") if future_dates else "N/A"

        if   ghi_mean >= 500: resource = "exceptional (arid/desert zones)"
        elif ghi_mean >= 300: resource = "excellent (tropics, Mediterranean)"
        elif ghi_mean >= 150: resource = "good (temperate zones)"
        elif ghi_mean >= 50:  resource = "moderate (cloudy regions)"
        else:                  resource = "low (mostly night hours in sample)"

        e_ideal = (ghi_max / 1000.0) * n if ghi_max > 0 else 1.0
        fc = (e_kwh / e_ideal * 100) if e_ideal > 0 else 0.0

        hourly_lines = "\n".join(
            f"  {h:02d}:00  →  {np.mean(by_hour[h]):6.1f} W/m²  "
            f"{'█' * min(int(np.mean(by_hour[h]) / 50), 18)}"
            for h in sorted(by_hour) if 5 <= h <= 19
        ) or "  (no daytime data)"

        return f"""You are Sowi, a senior renewable energy expert with 15 years of experience in photovoltaic solar and wind energy.
You have deep knowledge of:
  · PV system design (monocrystalline PERC/TOPCon/HJT, microinverters, optimizers)
  · Wind turbine selection (horizontal/vertical axis, capacity factors, hub height)
  · Hybrid systems (solar + wind + battery)
  · Resource assessment using Open-Meteo, NASA POWER, PVGIS, Global Wind Atlas
  · Production modeling (PVsyst, SAM, WAsP)
  · Electrical standards (NEC, IEC, RETIE Colombia)
  · Project economics (LCOE, IRR, payback, green financing)
  · Tropical, Andean, coastal, desert and temperate climates worldwide

════════════════════════════════════════════════════
ACTIVE FORECAST DATA — SOLAR
════════════════════════════════════════════════════
Coordinates      : Lat {lat:.4f}°  |  Lon {lon:.4f}°
Period           : {future_dates[0].strftime('%m/%d/%Y %H:%M') if future_dates else 'N/A'} → {future_dates[-1].strftime('%m/%d/%Y %H:%M') if future_dates else 'N/A'}
Total hours      : {n} h
LSTM model       : {modo}
Open-Meteo history: {years} year(s)

── GHI STATISTICS ──
  Avg GHI        : {ghi_mean:.2f} W/m²
  Peak GHI       : {ghi_max:.2f} W/m²  → reached at {pico_hora}
  Min GHI (>0)   : {ghi_min:.2f} W/m²
  Total energy   : {e_kwh:.3f} kWh/m²
  Productive hrs : {hrs_prod} h  (GHI > 50 W/m²)
  Peak-sun hrs   : {hrs_peak} h  (GHI > 600 W/m²)
  Best time slot : {best_h:02d}:00–{best_h+1:02d}:00  (avg {best_h_avg:.1f} W/m²)
  Capacity factor: {fc:.1f} %
  Resource class : {resource}
  Best day est.  : {best_day}  ({dias_energy.get(best_day, 0):.3f} kWh/m²)
  Worst day est. : {worst_day}  ({dias_energy.get(worst_day, 0):.3f} kWh/m²)

── HOURLY PROFILE (solar hours) ──
{hourly_lines}

════════════════════════════════════════════════════
RESPONSE RULES
════════════════════════════════════════════════════
LANGUAGE & STYLE
  · Always respond in clear, professional English.
  · Give the number first, then explain.
  · Use bold for key figures.
  · Use bullet lists when 3+ items.
  · End each technical response with a 1-line practical tip preceded by ⚡ or 🌊.

STANDARD ASSUMPTIONS (solar)
  · Panel efficiency       : 20% (standard 400 Wp monocrystalline)
  · Performance Ratio (PR) : 0.80
  · Panel area 400 Wp      : 1.95 m²
  · Daily output formula   : E_day (kWh) = (GHI_day / 1000) × Peak_kWp × PR
  · LCOE simplified        : LCOE = Total_cost_USD / (Annual_kWh × 25)
  · Colombia reference cost: 1.0–1.4 USD/Wp installed (residential 2024–2025)

SMART BEHAVIOR
  · If user asks how many panels and has NOT mentioned monthly consumption, ask ONE short question.
  · If asked about payback/ROI, also ask for local kWh tariff if not provided.
  · Never invent data not in the forecast; if something cannot be derived, say so.
  · If you spot anomalies (GHI > 0 at midnight, extreme outliers), proactively mention them.
  · You can answer general solar energy questions even outside this specific forecast.
  · When geographic context is relevant, apply specific knowledge of the region.
"""

    else:  # Wind
        wind_speed    = arr
        avg_ws        = avg_val
        max_ws        = max_val
        rho           = 1.225
        power_density = 0.5 * rho * (wind_speed ** 3)
        avg_pd        = float(power_density.mean()) if n > 0 else 0.0
        total_kwh     = float(power_density.sum()) / 1000.0

        if avg_ws >= 7.5: resource = "excellent (class 7) – ideal for large turbines"
        elif avg_ws >= 6.0: resource = "good (class 5–6) – viable for utility-scale"
        elif avg_ws >= 4.5: resource = "moderate (class 3–4) – suitable for small turbines"
        elif avg_ws >= 3.0: resource = "marginal (class 2) – limited, hybrid systems"
        else:               resource = "poor (class 1) – not recommended for standalone wind"

        cf_est = min(0.45, max(0.05, (avg_ws - 3) / 12)) if avg_ws > 3 else 0.05
        fc     = cf_est * 100

        pico_idx  = int(arr.argmax()) if n > 0 else 0
        pico_hora = future_dates[pico_idx].strftime("%Y-%m-%d %H:%M") if future_dates else "N/A"

        hourly_lines = "\n".join(
            f"  {h:02d}:00  →  {np.mean(by_hour[h]):5.2f} m/s  "
            f"{'█' * min(int(np.mean(by_hour[h])), 18)}"
            for h in sorted(by_hour)
        ) or "  (no wind data)"

        return f"""You are Sowi, a senior renewable energy expert with 15 years of experience in photovoltaic solar and wind energy.
You have deep knowledge of:
  · Wind turbine technology (HAWT, VAWT, offshore, onshore)
  · Wind resource assessment (Weibull distributions, shear, turbulence)
  · Site suitability, wake effects, micrositing
  · Hybrid systems (solar + wind + battery)
  · Energy production modeling (WAsP, OpenWind, SAM)
  · Electrical standards (IEC 61400, grid codes)
  · Project economics (LCOE, IRR, payback, green financing)
  · Tropical, Andean, coastal, desert and temperate climates worldwide

════════════════════════════════════════════════════
ACTIVE FORECAST DATA — WIND
════════════════════════════════════════════════════
Coordinates      : Lat {lat:.4f}°  |  Lon {lon:.4f}°
Period           : {future_dates[0].strftime('%m/%d/%Y %H:%M') if future_dates else 'N/A'} → {future_dates[-1].strftime('%m/%d/%Y %H:%M') if future_dates else 'N/A'}
Total hours      : {n} h
LSTM model       : {modo}
Open-Meteo history: {years} year(s)

── WIND SPEED STATISTICS ──
  Avg wind speed : {avg_ws:.2f} m/s
  Max wind speed : {max_ws:.2f} m/s  → reached at {pico_hora}
  Total energy (wind power density) : {total_kwh:.2f} kWh/m² (over {n} h)
  Avg power density : {avg_pd:.1f} W/m²
  Productive hrs  : {productive_hrs} h  (wind speed > 0.5 m/s)
  Best time slot  : {best_h:02d}:00–{best_h+1:02d}:00  (avg {best_h_avg:.2f} m/s)
  Capacity factor (est.) : {fc:.1f} %
  Resource class  : {resource}
  Best day est.   : {best_day}  ({dias_energy.get(best_day, 0):.3f} kWh/m²)
  Worst day est.  : {worst_day}  ({dias_energy.get(worst_day, 0):.3f} kWh/m²)

── HOURLY PROFILE (wind speed) ──
{hourly_lines}

════════════════════════════════════════════════════
RESPONSE RULES
════════════════════════════════════════════════════
LANGUAGE & STYLE
  · Always respond in clear, professional English.
  · Give the number first, then explain.
  · Use bold for key figures.
  · Use bullet lists when 3+ items.
  · End each technical response with a 1-line practical tip preceded by ⚡ or 🌊.

STANDARD ASSUMPTIONS (wind)
  · Turbine hub height    : 50 m (assumed)
  · Air density           : 1.225 kg/m³ (standard)
  · Power curve not known → use power density (0.5·ρ·v³) as estimate.
  · For actual energy, multiply by rotor swept area and turbine efficiency (typically 30–45%).
  · LCOE simplified       : LCOE = Total_cost_USD / (Annual_kWh × 20).

SMART BEHAVIOR
  · If user asks about turbine size, ask for desired power or annual consumption.
  · If asked about payback/ROI, ask for local electricity tariff and turbine cost estimate if not provided.
  · Never invent data not in the forecast; if something cannot be derived, say so.
  · If you spot anomalies (wind speed > 20 m/s constant, negative values), mention them.
  · You can answer general wind energy questions even outside this specific forecast.
  · When geographic context is relevant, apply specific knowledge of wind patterns in the region.
"""


# ══════════════════════════════════════════════════════════════════════════════
#  CLAUDE API
# ══════════════════════════════════════════════════════════════════════════════

def _call_sowi(system: str, history: list) -> str:
    try:
        import anthropic
    except ImportError:
        return "❌ <strong>anthropic</strong> package not installed. Run: <code>pip install anthropic</code>"

    try:
        api_key = st.secrets["ANTHROPIC_API_KEY"]
    except (KeyError, FileNotFoundError):
        return (
            "❌ <strong>API key not found.</strong> "
            "Add <code>ANTHROPIC_API_KEY = \"sk-ant-...\"</code> to "
            "<code>.streamlit/secrets.toml</code> and restart the app."
        )

    try:
        client = anthropic.Anthropic(api_key=api_key)
        r = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1400,
            system=system,
            messages=[{"role": m["role"], "content": m["content"]} for m in history],
        )
        return r.content[0].text
    except anthropic.AuthenticationError:
        return (
            "❌ <strong>Invalid API key.</strong> "
            "Check your key in <code>.streamlit/secrets.toml</code>."
        )
    except anthropic.RateLimitError:
        return "⏱️ <strong>API rate limit reached.</strong> Please wait a moment and try again."
    except anthropic.APIConnectionError:
        return "🌐 <strong>Connection error.</strong> Check your internet connection and try again."
    except Exception as e:
        return f"❌ <strong>Error:</strong> <code>{str(e)[:200]}</code>"


# ══════════════════════════════════════════════════════════════════════════════
#  QUICK PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

QUICK = [
    ("⚡", "Energy estimate",
     "Based on this forecast, how much energy could I generate? Show me the calculation step by step."),
    ("🌊", "Resource quality",
     "Analyze the renewable resource quality at this location. Is it suitable for solar and/or wind investment?"),
    ("📊", "Best time slot",
     "Which hours of the day show the highest production in the forecast? When is the best time to charge batteries?"),
    ("🔋", "Storage sizing",
     "What battery capacity in kWh would I need to cover nighttime / low-wind hours based on this forecast?"),
    ("💡", "Return on investment",
     "How long would it take to recover the investment of a renewable system at this location? Give a detailed estimate."),
    ("⚙️", "Hybrid system",
     "Could a hybrid solar+wind system work well here? What would be the recommended mix based on the data?"),
]


# ══════════════════════════════════════════════════════════════════════════════
#  RENDER: METRICS
# ══════════════════════════════════════════════════════════════════════════════

def _best_hour_label(arr, future_dates):
    by_h = {}
    for d, v in zip(future_dates, arr):
        by_h.setdefault(d.hour, []).append(v)
    if not by_h:
        return "N/A"
    best_h = max(by_h, key=lambda h: np.mean(by_h[h]))
    return f"{best_h:02d}:00–{best_h+1:02d}:00"


def _render_metrics(preds, future_dates, energy_type, modo):
    arr = np.clip(np.asarray(preds, dtype=float), 0, None)
    if energy_type == "Solar":
        ekwh = float(arr.sum()) / 1000
        hp   = int((arr > 50).sum())
        mean_w = float(arr.mean())
        if mean_w >= 300: cal, tag = "Excellent ☀️", "tag-solar"
        elif mean_w >= 150: cal, tag = "Good 🌤️", "tag-ok"
        else: cal, tag = "Moderate ⛅", "tag-warn"

        st.markdown(f"""
        <div class="metrics-grid">
          <div class="mc mc-solar"><div class="mc-lbl">🌞 Avg GHI</div>
            <div class="mc-val">{mean_w:.1f}<span class="mc-unit"> W/m²</span></div>
            <span class="mc-tag {tag}">{cal}</span></div>
          <div class="mc mc-solar"><div class="mc-lbl">📈 Peak GHI</div>
            <div class="mc-val">{arr.max():.1f}<span class="mc-unit"> W/m²</span></div>
            <span class="mc-tag tag-solar">Peak</span></div>
          <div class="mc mc-solar"><div class="mc-lbl">⚡ Total Energy</div>
            <div class="mc-val">{ekwh:.3f}<span class="mc-unit"> kWh/m²</span></div>
            <span class="mc-tag tag-solar">Cumulative</span></div>
          <div class="mc mc-solar"><div class="mc-lbl">⏱️ Productive Hours</div>
            <div class="mc-val">{hp}<span class="mc-unit"> h</span></div>
            <span class="mc-tag tag-solar">GHI > 50 W/m²</span></div>
          <div class="mc mc-solar"><div class="mc-lbl">Best Time Slot</div>
            <div class="mc-val" style="font-size:1.1rem">{_best_hour_label(arr, future_dates)}</div>
            <span class="mc-tag tag-solar">Peak GHI hour</span></div>
          <div class="mc mc-solar"><div class="mc-lbl">Model · Hours</div>
            <div class="mc-val">{len(future_dates)}<span class="mc-unit"> h</span></div>
            <span class="mc-tag tag-solar">{modo}</span></div>
        </div>""", unsafe_allow_html=True)

    else:  # Wind
        avg_ws = float(arr.mean())
        max_ws = float(arr.max())
        rho = 1.225
        power_density = 0.5 * rho * (arr ** 3)
        avg_pd = float(power_density.mean())
        total_kwh = float(power_density.sum()) / 1000
        if avg_ws >= 6.0: cal, tag = "Excellent 💨", "tag-wind"
        elif avg_ws >= 4.0: cal, tag = "Good 🌬️", "tag-ok"
        else: cal, tag = "Moderate 🍃", "tag-warn"

        st.markdown(f"""
        <div class="metrics-grid">
          <div class="mc mc-wind"><div class="mc-lbl">💨 Avg Wind Speed</div>
            <div class="mc-val">{avg_ws:.2f}<span class="mc-unit"> m/s</span></div>
            <span class="mc-tag {tag}">{cal}</span></div>
          <div class="mc mc-wind"><div class="mc-lbl">📈 Max Wind Speed</div>
            <div class="mc-val">{max_ws:.2f}<span class="mc-unit"> m/s</span></div>
            <span class="mc-tag tag-wind">Peak gust</span></div>
          <div class="mc mc-wind"><div class="mc-lbl">⚡ Avg Power Density</div>
            <div class="mc-val">{avg_pd:.1f}<span class="mc-unit"> W/m²</span></div>
            <span class="mc-tag tag-wind">0.5·ρ·v³</span></div>
          <div class="mc mc-wind"><div class="mc-lbl">🔋 Total Energy</div>
            <div class="mc-val">{total_kwh:.2f}<span class="mc-unit"> kWh/m²</span></div>
            <span class="mc-tag tag-wind">Over {len(future_dates)} h</span></div>
          <div class="mc mc-wind"><div class="mc-lbl">Best Time Slot</div>
            <div class="mc-val" style="font-size:1.1rem">{_best_hour_label(arr, future_dates)}</div>
            <span class="mc-tag tag-wind">Peak wind hour</span></div>
          <div class="mc mc-wind"><div class="mc-lbl">Model · Hours</div>
            <div class="mc-val">{len(future_dates)}<span class="mc-unit"> h</span></div>
            <span class="mc-tag tag-wind">{modo}</span></div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  RENDER: CHIPS
# ══════════════════════════════════════════════════════════════════════════════

def _render_chips():
    st.markdown('<div class="chips-lbl">⚡ Quick questions</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for i, (ico, label, prompt) in enumerate(QUICK):
        with cols[i % 3]:
            if st.button(f"{ico}  {label}", key=f"qp_{i}", use_container_width=True):
                st.session_state["_qp"] = prompt


# ══════════════════════════════════════════════════════════════════════════════
#  RENDER: CHAT
# ══════════════════════════════════════════════════════════════════════════════

def _render_chat(system: str):
    st.markdown("""
    <div class="chat-shell">
      <div class="chat-topbar">
        <div class="tls">
          <div class="tl tl-r"></div><div class="tl tl-y"></div><div class="tl tl-g"></div>
        </div>
        <span class="topbar-name">🌞💨 Sowi · Renewable Energy Expert</span>
        <span class="topbar-model">claude-sonnet-4.5</span>
      </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="chat-msgs">', unsafe_allow_html=True)

    if not st.session_state.get("sowi_history"):
        _render_msg("assistant",
            "Hello! I'm <strong>Sowi</strong> 🌞💨 — your renewable energy expert for both solar and wind. "
            "I've loaded your forecast data and I'm ready to help you. "
            "I can calculate energy production, size your system, evaluate resource quality, "
            "estimate ROI, design battery storage, and much more. Where shall we start? ⚡"
        )

    for m in st.session_state.get("sowi_history", []):
        _render_msg(m["role"], m["content"])

    if st.session_state.get("_typing"):
        st.markdown("""
        <div class="typing-row">
          <div class="av av-sowi">🌞💨</div>
          <div>
            <div class="dots"><span></span><span></span><span></span></div>
            <div class="typing-lbl">Sowi is analyzing…</div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)   # close chat-msgs

    # Input zone
    st.markdown('<div class="input-zone">', unsafe_allow_html=True)
    ci, cs = st.columns([5, 1])
    with ci:
        txt = st.text_area("",
            placeholder="Ask about solar energy, wind power, system sizing, ROI, batteries…",
            key="sowi_input", height=80, label_visibility="collapsed")
    with cs:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        send = st.button("⚡ Send", type="primary", use_container_width=True, key="sowi_send")
    st.markdown('</div>', unsafe_allow_html=True)   # close input-zone
    st.markdown('</div>', unsafe_allow_html=True)   # close chat-shell

    qp    = st.session_state.pop("_qp", None)
    final = qp or (txt.strip() if send and txt.strip() else None)

    if final:
        st.session_state.setdefault("sowi_history", [])
        st.session_state["sowi_history"].append({"role": "user", "content": final})
        st.session_state["_typing"] = True
        st.rerun()

    if st.session_state.get("_typing"):
        ans = _call_sowi(system, st.session_state["sowi_history"])
        st.session_state["sowi_history"].append({"role": "assistant", "content": ans})
        st.session_state["_typing"] = False
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def _render_sidebar(has_data: bool, energy_type: str):
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center;padding:.5rem 0 1.2rem'>
          <div style='font-size:1.8rem;margin-bottom:4px'>🌞💨</div>
          <div style='font-size:.84rem;font-weight:600;color:#eeeef4'>Sowi · AI Analyst</div>
          <div style='font-size:.68rem;color:#44445a;margin-top:2px'>Solar + Wind · Open‑Meteo</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("---")

        if has_data:
            st.success(f"✅ Forecast loaded  ({energy_type})")
        else:
            st.warning("⚠️ No active forecast")

        st.metric("Messages", len(st.session_state.get("sowi_history", [])))
        st.markdown("---")

        # ── FIXED: direct navigation back to forecast ──
        st.markdown(
            "<div style='font-size:.7rem;text-transform:uppercase;letter-spacing:.1em;"
            "color:#44445a;margin-bottom:.5rem'>🔗 Navigation</div>",
            unsafe_allow_html=True)
        _nav_to_forecast("⚡ Back to Forecast", key="nav_sidebar", use_container_width=True)

        st.markdown("---")

        if st.button("🗑️ New conversation", use_container_width=True):
            st.session_state["sowi_history"] = []
            st.session_state["_typing"]      = False
            st.rerun()

        st.markdown("---")
        st.markdown("""
        <div style='font-size:.7rem;color:#44445a;line-height:1.65'>
        <strong style='color:#8888a0'>Sowi can answer about:</strong><br>
        · PV & wind system sizing<br>
        · Energy production estimates<br>
        · Resource quality analysis<br>
        · Battery storage design<br>
        · ROI, LCOE, payback<br>
        · Hybrid solar‑wind systems<br>
        · Global climate & wind patterns<br><br>
        <strong style='color:#8888a0'>Active model:</strong><br>
        claude-sonnet-4.5
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    has_data    = st.session_state.get("modelo_ejecutado", False)
    energy_type = st.session_state.get("energy_type", "Solar")

    _render_sidebar(has_data, energy_type)

    # ── Back to Forecast — top of page, always visible ──
    _nav_to_forecast("← Back to Forecast", key="nav_top")

    # Hero
    st.markdown("""
    <div class="hero">
      <div class="eyebrow"><div class="ldot"></div>
        Renewable AI Assistant &nbsp;·&nbsp; Powered by Claude
      </div>
      <h1 class="hero-title">
        Ask <span class="hl">Sowi</span><br>about your forecast
      </h1>
      <p class="hero-sub">
        Solar & wind expert · calculates, sizes, evaluates and recommends 🌞💨
      </p>
    </div>
    <div class="wave-sep"></div>
    """, unsafe_allow_html=True)

    # ── No data state ──────────────────────────────────────────────────────────
    if not has_data:
        st.markdown("""
        <div class="empty-state">
          <div class="empty-icon">🌞💨</div>
          <div class="empty-title">No active forecast</div>
          <div class="empty-sub">
            Go to the <strong>Energy Forecast</strong> page, select Solar or Wind,
            configure the model and click ⚡ Run model.<br>
            Sowi will automatically load the results and analyze them with you.
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        _, cc, _ = st.columns([1, 2, 1])
        with cc:
            _nav_to_forecast("⚡ Go to Forecast →", key="nav_empty", use_container_width=True)
        return

    # ── Load & validate forecast data ─────────────────────────────────────────
    required = ["predictions", "future_dates", "lat", "lon", "modo"]
    if not all(k in st.session_state for k in required):
        st.error("⚠️ Incomplete forecast data. Please run the forecast again.")
        st.stop()

    preds        = st.session_state["predictions"]
    future_dates = st.session_state["future_dates"]
    lat          = float(st.session_state["lat"])
    lon          = float(st.session_state["lon"])
    modo         = st.session_state["modo"]

    if preds is None or len(preds) == 0:
        st.error("⚠️ Forecast data is empty. Please run the forecast again.")
        st.stop()

    years = 2.0
    if "date_start" in st.session_state and "date_end" in st.session_state:
        try:
            start = datetime.strptime(str(st.session_state["date_start"]), "%Y-%m-%d")
            end   = datetime.strptime(str(st.session_state["date_end"]),   "%Y-%m-%d")
            years = round((end - start).days / 365.25, 1)
        except Exception:
            years = 2.0

    # Context banner
    st.markdown(
        f'<div class="ctx-banner">⚡ Context loaded · {energy_type}'
        f'<span class="ctx-meta">'
        f'· {len(future_dates)} h forecast &nbsp;|&nbsp;'
        f'{modo} &nbsp;|&nbsp;'
        f'Lat {lat:.2f}° Lon {lon:.2f}° &nbsp;|&nbsp;'
        f'{years} year(s) Open‑Meteo history'
        f'</span></div>',
        unsafe_allow_html=True
    )

    # Metrics
    _render_metrics(preds, future_dates, energy_type, modo)

    # Quick chips
    _render_chips()

    # Build dynamic system prompt and render chat
    system = _build_expert_prompt(energy_type, preds, future_dates, lat, lon, years, modo)
    _render_chat(system)

    # ── FIXED: Back to Forecast — bottom CTA ──
    st.markdown(
        "<div style='background:var(--surface);border:1px solid var(--bd);"
        "border-radius:16px;padding:1.2rem 1.6rem;text-align:center;margin:1rem 0'>"
        "<div style='font-size:.85rem;color:#8888a0;margin-bottom:.7rem'>"
        "Done analyzing? Go back to run a new forecast.</div>"
        "</div>",
        unsafe_allow_html=True
    )
    _, cc2, _ = st.columns([1, 2, 1])
    with cc2:
        _nav_to_forecast("⚡ Back to Forecast", key="nav_bottom", use_container_width=True)

    st.markdown("""
    <div class="footer">
      🌞💨 Sowi AI Analyst · Open‑Meteo · Claude Sonnet 4.5 · Solar & Wind Expert
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
