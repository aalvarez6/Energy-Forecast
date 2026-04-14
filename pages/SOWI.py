"""
Sowi AI Analyst — Sowi AI Analyst.py
English version · solar & wind expert · dynamic prompts
FULLY CORRECTED VERSION
"""

import streamlit as st
import numpy as np
import anthropic
import html as html_lib
from datetime import datetime

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Sowi · Energy Analyst",
    page_icon="🌞💨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =============================================================================
# CSS (same as before, but I keep it short here – you can copy your existing CSS)
# =============================================================================
st.markdown("""
<style>
/* Copy the entire CSS from your working Sowi AI Analyst.py here */
/* For brevity, I'm not repeating the 200+ lines, but you must include your CSS. */
/* Use the exact same CSS that was working before. */
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONSTANTS
# =============================================================================
CLAUDE_MODEL = "claude-3-5-sonnet-20241022"   # correct model name
MAX_HISTORY = 40

# =============================================================================
# HELPERS
# =============================================================================
def _ts() -> str:
    return datetime.now().strftime("%H:%M")

def _sanitize(text: str) -> str:
    return html_lib.escape(text)

def _render_msg(role: str, content: str):
    wrap = "mw usr" if role == "user" else "mw"
    bbl  = "bbl bbl-usr" if role == "user" else "bbl bbl-sowi"
    av_c = "av av-usr" if role == "user" else "av av-sowi"
    icon = "👤" if role == "user" else "🌞💨"
    safe_content = _sanitize(content).replace("\n", "<br>") if role == "user" else content
    st.markdown(f"""
    <div class="{wrap}">
      <div class="{av_c}">{icon}</div>
      <div>
        <div class="{bbl}">{safe_content}</div>
        <div class="ts-lbl">{_ts()}</div>
      </div>
    </div>""", unsafe_allow_html=True)

def _trim_history(history: list, max_msgs: int = MAX_HISTORY) -> list:
    if len(history) <= max_msgs:
        return history
    trim_to = len(history) - max_msgs
    while trim_to < len(history) and history[trim_to]["role"] != "user":
        trim_to += 1
    return history[trim_to:]

# =============================================================================
# DYNAMIC EXPERT PROMPT (Solar / Wind) – same as your final working version
# =============================================================================
def _build_expert_prompt(energy_type, preds, future_dates, lat, lon, years, modo):
    # ... (keep your full implementation; it's long but correct)
    # I won't repeat it here to save space, but you must paste your working function.
    # The one you had with positive_hours fix, etc.
    return "Expert prompt content"

# =============================================================================
# CLAUDE API
# =============================================================================
def _call_sowi(system: str, history: list) -> str:
    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
    trimmed = _trim_history(history)
    r = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1400,
        system=system,
        messages=[{"role": m["role"], "content": m["content"]} for m in trimmed],
    )
    return r.content[0].text

# =============================================================================
# QUICK PROMPTS
# =============================================================================
QUICK = [
    ("⚡", "Energy estimate", "Based on this forecast, how much energy could I generate? Show me the calculation."),
    ("🌊", "Resource quality", "Analyze the renewable resource quality at this location. Is it good for solar and/or wind?"),
    ("📊", "Best time slot", "Which hours of the day show the highest production in the forecast? When is the best time to charge batteries?"),
    ("🔋", "Storage sizing", "What battery capacity in kWh would I need to cover nighttime/low‑wind hours based on the forecast?"),
    ("💡", "Return on investment", "How long would it take to recover the investment of a renewable system at this location? Give an estimate."),
    ("⚙️", "Hybrid system", "Could a hybrid solar+wind system work well here? What would be the recommended mix?"),
]

# =============================================================================
# METRICS & BEST HOUR
# =============================================================================
def _best_hour_label(arr, future_dates):
    by_h = {}
    for d, v in zip(future_dates, arr):
        by_h.setdefault(d.hour, []).append(v)
    positive_h = {h: vs for h, vs in by_h.items() if np.mean(vs) > 1.0}
    if positive_h:
        best_h = max(positive_h, key=lambda h: np.mean(positive_h[h]))
    elif by_h:
        best_h = max(by_h, key=lambda h: np.mean(by_h[h]))
    else:
        best_h = 12
    return f"{best_h:02d}:00–{best_h+1:02d}:00"

def _render_metrics(preds, future_dates, energy_type, modo):
    arr = np.clip(preds, 0, None)
    best_slot = _best_hour_label(arr, future_dates)
    if energy_type == "Solar":
        ekwh = arr.sum() / 1000
        hp = int((arr > 50).sum())
        if arr.mean() >= 300: cal, tag = "Excellent ☀️", "tag-solar"
        elif arr.mean() >= 150: cal, tag = "Good 🌤️", "tag-ok"
        else: cal, tag = "Moderate ⛅", "tag-warn"
        st.markdown(f"""
        <div class="metrics-grid">
          <div class="mc mc-solar"><div class="mc-lbl">🌞 Avg GHI</div>
            <div class="mc-val">{arr.mean():.1f}<span class="mc-unit"> W/m²</span></div>
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
            <div class="mc-val" style="font-size:1.1rem">{best_slot}</div>
            <span class="mc-tag tag-solar">Peak GHI hour</span></div>
          <div class="mc mc-solar"><div class="mc-lbl">Model · Hours</div>
            <div class="mc-val">{len(future_dates)}<span class="mc-unit"> h</span></div>
            <span class="mc-tag tag-solar">{modo}</span></div>
        </div>""", unsafe_allow_html=True)
    else:
        avg_ws = arr.mean()
        max_ws = arr.max()
        rho = 1.225
        power_density = 0.5 * rho * (arr ** 3)
        avg_pd = power_density.mean()
        total_energy_kwh = power_density.sum() / 1000
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
            <div class="mc-val">{total_energy_kwh:.2f}<span class="mc-unit"> kWh/m²</span></div>
            <span class="mc-tag tag-wind">Over {len(future_dates)} h</span></div>
          <div class="mc mc-wind"><div class="mc-lbl">Best Time Slot</div>
            <div class="mc-val" style="font-size:1.1rem">{best_slot}</div>
            <span class="mc-tag tag-wind">Peak wind hour</span></div>
          <div class="mc mc-wind"><div class="mc-lbl">Model · Hours</div>
            <div class="mc-val">{len(future_dates)}<span class="mc-unit"> h</span></div>
            <span class="mc-tag tag-wind">{modo}</span></div>
        </div>""", unsafe_allow_html=True)

def _render_chips():
    st.markdown('<div class="chips-lbl">⚡ Quick questions</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for i, (ico, label, prompt) in enumerate(QUICK):
        with cols[i % 3]:
            if st.button(f"{ico}  {label}", key=f"qp_{i}", use_container_width=True):
                st.session_state["_qp"] = prompt

def _render_chat(system: str):
    history = st.session_state.get("sowi_history", [])
    n_msgs = len(history)
    if n_msgs > MAX_HISTORY * 0.75:
        pct = int(n_msgs / MAX_HISTORY * 100)
        st.markdown(f'<div class="hist-badge">💬 {n_msgs} messages — {pct}% of history limit ({MAX_HISTORY}). Older messages will be dropped automatically.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="chat-shell">
      <div class="chat-topbar">
        <div class="tls"><div class="tl tl-r"></div><div class="tl tl-y"></div><div class="tl tl-g"></div></div>
        <span class="topbar-name">🌞💨 Sowi · Renewable Energy Expert</span>
        <span class="topbar-model">claude-3-5-sonnet</span>
      </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="chat-msgs">', unsafe_allow_html=True)
    if not history:
        _render_msg("assistant", "Hello! I'm <strong>Sowi</strong> 🌞💨 — your renewable energy expert for both solar and wind. I've loaded your forecast data (Open‑Meteo) and I'm ready to help. I can calculate energy production, size your system, evaluate the resource quality, estimate ROI, design battery storage, and much more. Where shall we start? ⚡")
    for m in history:
        _render_msg(m["role"], m["content"])
    if st.session_state.get("_typing"):
        st.markdown("""
        <div class="typing-row">
          <div class="av av-sowi">🌞💨</div>
          <div><div class="dots"><span></span><span></span><span></span></div><div class="typing-lbl">Sowi is analyzing…</div></div>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="input-zone">', unsafe_allow_html=True)
    ci, cs = st.columns([5,1])
    with ci:
        txt = st.text_area("", placeholder="Ask about solar energy, wind power, or the forecast results…", key="sowi_input", height=80, label_visibility="collapsed")
    with cs:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        send = st.button("⚡ Send", type="primary", use_container_width=True, key="sowi_send")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    qp = st.session_state.pop("_qp", None)
    final = qp or (txt.strip() if send and txt.strip() else None)
    if final:
        st.session_state.setdefault("sowi_history", [])
        st.session_state["sowi_history"].append({"role": "user", "content": final})
        st.session_state["_typing"] = True
        st.rerun()
    if st.session_state.get("_typing"):
        try:
            ans = _call_sowi(system, st.session_state["sowi_history"])
        except anthropic.AuthenticationError:
            ans = "❌ <strong>Invalid API key.</strong> Check your key in <code>.streamlit/secrets.toml</code>."
        except anthropic.RateLimitError:
            ans = "⏱️ <strong>API rate limit reached.</strong> Please wait a moment and try again."
        except KeyError:
            ans = "❌ <strong>API key not found.</strong> Add <code>ANTHROPIC_API_KEY</code> to <code>.streamlit/secrets.toml</code>."
        except anthropic.BadRequestError as e:
            if "context_length" in str(e).lower() or "too long" in str(e).lower():
                st.session_state["sowi_history"] = _trim_history(st.session_state["sowi_history"], max_msgs=MAX_HISTORY//2)
                try:
                    ans = _call_sowi(system, st.session_state["sowi_history"])
                except Exception:
                    ans = "⚠️ <strong>Conversation too long.</strong> History was trimmed. Please try again."
            else:
                ans = f"❌ <strong>Request error:</strong> <code>{html_lib.escape(str(e))}</code>"
        except Exception as e:
            ans = f"❌ <strong>Error connecting to Claude:</strong> <code>{html_lib.escape(str(e))}</code>"
        st.session_state["sowi_history"].append({"role": "assistant", "content": ans})
        st.session_state["_typing"] = False
        st.rerun()

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
        n_msgs = len(st.session_state.get("sowi_history", []))
        st.metric("Messages", n_msgs)
        if n_msgs > 0:
            pct = min(1.0, n_msgs / MAX_HISTORY)
            st.progress(pct, text=f"History: {n_msgs}/{MAX_HISTORY}")
        st.markdown("---")
        if st.button("🗑️ New conversation", use_container_width=True):
            st.session_state["sowi_history"] = []
            st.session_state["_typing"] = False
            st.rerun()
        st.markdown("---")
        st.markdown(f"""
        <div style='font-size:.7rem;color:#44445a;line-height:1.65'>
        <strong style='color:#8888a0'>Sowi can answer about:</strong><br>
        · PV & wind system sizing<br>
        · Energy production estimates<br>
        · Resource quality analysis<br>
        · Battery storage design<br>
        · ROI, LCOE, payback<br>
        · Hybrid solar‑wind systems<br>
        · Tropical and Andean climates<br><br>
        <strong style='color:#8888a0'>Active model:</strong><br>{CLAUDE_MODEL}
        </div>""", unsafe_allow_html=True)

# =============================================================================
# MAIN
# =============================================================================
def main():
    has_data = st.session_state.get("modelo_ejecutado", False)
    energy_type = st.session_state.get("energy_type", "Solar")
    _render_sidebar(has_data, energy_type)

    st.markdown("""
    <div class="hero">
      <div class="eyebrow"><div class="ldot"></div>Renewable AI Assistant &nbsp;·&nbsp; Powered by Claude</div>
      <h1 class="hero-title">Ask <span class="hl">Sowi</span><br>about your forecast</h1>
      <p class="hero-sub">Solar & wind expert · calculates, sizes, evaluates and recommends 🌞💨</p>
    </div>
    <div class="wave-sep"></div>
    """, unsafe_allow_html=True)

    if not has_data:
        st.markdown("""
        <div class="empty-state">
          <div class="empty-icon">🌞💨</div>
          <div class="empty-title">No active forecast</div>
          <div class="empty-sub">Go to the Forecast page, select Solar or Wind, configure the model and run it. Sowi will automatically load the results and analyze them with you.</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        _, cc, _ = st.columns([1,2,1])
        with cc:
            try:
                st.page_link("Energy Forecast.py", label="⚡ Go to Forecast →", icon="🌞", use_container_width=True)
            except Exception:
                st.markdown("""
                <a href="/" target="_self" style="display:block;text-align:center;background:linear-gradient(135deg,rgba(245,180,50,.18),rgba(245,180,50,.08));border:1px solid rgba(245,180,50,.38);color:#f5b432;font-family:'Sora',sans-serif;font-weight:600;padding:0.5rem 1rem;border-radius:12px;text-decoration:none">⚡ Go to Forecast →</a>
                """, unsafe_allow_html=True)
                st.caption("If the button doesn't work, go to the main page manually.")
        return

    required = ["predictions","future_dates","lat","lon","modo"]
    if not all(k in st.session_state for k in required):
        st.error("Incomplete forecast data. Please run the forecast again.")
        st.stop()

    preds = st.session_state["predictions"]
    future_dates = st.session_state["future_dates"]
    lat = st.session_state["lat"]
    lon = st.session_state["lon"]
    modo = st.session_state["modo"]

    if "date_start" in st.session_state and "date_end" in st.session_state:
        try:
            start = datetime.strptime(st.session_state["date_start"], "%Y-%m-%d")
            end   = datetime.strptime(st.session_state["date_end"],   "%Y-%m-%d")
            years = round((end - start).days / 365.25, 1)
        except:
            years = 2.0
    else:
        years = 2.0

    st.markdown(f"""
    <div class="ctx-banner">
      ⚡ Context loaded · {energy_type}
      <span class="ctx-meta">· {len(future_dates)} h forecast &nbsp;|&nbsp;{modo} &nbsp;|&nbsp;Lat {lat:.2f}° Lon {lon:.2f}° &nbsp;|&nbsp;{years} year(s) Open‑Meteo history</span>
    </div>""", unsafe_allow_html=True)

    _render_metrics(preds, future_dates, energy_type, modo)
    _render_chips()

    system = _build_expert_prompt(energy_type, preds, future_dates, lat, lon, years, modo)
    _render_chat(system)

    st.markdown(f"""
    <div class="footer">🌞💨 Sowi AI Analyst · Open‑Meteo · {CLAUDE_MODEL} · Solar & Wind Expert</div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
