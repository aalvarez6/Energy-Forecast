"""
Sowi AI Analyst — Sowi AI Analyst.py
English version · solar & wind expert · dynamic prompts
FIXED VERSION — bugs corregidos y optimizado
"""

import streamlit as st
import numpy as np
import anthropic
import html as html_lib
from datetime import datetime

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
#  CSS (igual que antes, no lo repito para no alargar)
# ══════════════════════════════════════════════════════════════════════════════
# ... (mantén el CSS que ya tenías) ...


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS - MODELO CORRECTO DE CLAUDE
# ══════════════════════════════════════════════════════════════════════════════
# FIX #5: modelo correcto (el anterior "claude-sonnet-4-5" no existe)
CLAUDE_MODEL   = "claude-3-5-sonnet-20241022"
MAX_HISTORY    = 40


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS (igual que antes, pero con la corrección de _sanitize)
# ══════════════════════════════════════════════════════════════════════════════
def _ts() -> str:
    return datetime.now().strftime("%H:%M")

def _sanitize(text: str) -> str:
    return html_lib.escape(text)

def _render_msg(role: str, content: str):
    # ... igual que antes ...

def _trim_history(history: list, max_msgs: int = MAX_HISTORY) -> list:
    # ... igual que antes ...


# ══════════════════════════════════════════════════════════════════════════════
#  DYNAMIC EXPERT PROMPT (Solar / Wind)
# ══════════════════════════════════════════════════════════════════════════════
def _build_expert_prompt(...):
    # ... igual que antes, pero asegúrate de que la línea de "Best time slot"
    # no tenga errores (ya lo corregiste con positive_hours) ...
    # Mantenlo tal cual como en tu versión final, que ya estaba bien.


# ══════════════════════════════════════════════════════════════════════════════
#  CLAUDE API
# ══════════════════════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN (con la corrección de redirección a la página principal)
# ══════════════════════════════════════════════════════════════════════════════
def main():
    has_data = st.session_state.get("modelo_ejecutado", False)
    energy_type = st.session_state.get("energy_type", "Solar")
    _render_sidebar(has_data, energy_type)

    # Hero
    st.markdown("...")  # igual que antes

    if not has_data:
        st.markdown("...")  # empty state
        _, cc, _ = st.columns([1,2,1])
        with cc:
            # REDIRECCIÓN CORREGIDA: apunta al archivo principal con espacios
            try:
                st.page_link("Energy Forecast.py", label="⚡ Go to Forecast →", icon="🌞", use_container_width=True)
            except Exception:
                st.markdown("""
                <a href="/" target="_self" style="display:block;text-align:center;
                background:linear-gradient(135deg,rgba(245,180,50,.18),rgba(245,180,50,.08));
                border:1px solid rgba(245,180,50,.38);color:#f5b432;font-family:'Sora',sans-serif;
                font-weight:600;padding:0.5rem 1rem;border-radius:12px;text-decoration:none">
                ⚡ Go to Forecast →</a>
                """, unsafe_allow_html=True)
                st.caption("If the button doesn't work, go to the main page manually.")
        return

    # Resto del código (cargar datos, métricas, chat, etc.) igual que antes
    # ... pero asegúrate de que el modelo de Claude sea el correcto (ya lo cambiamos)
    # ...


if __name__ == "__main__":
    main()
