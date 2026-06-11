"""
╔══════════════════════════════════════════════════════════════════════╗
║        Wandr — AI Travel Concierge  |  app.py                       ║
║  Tech Stack:                                                         ║
║  • Frontend  : Streamlit                                             ║
║  • Backend   : Python + LangChain (langchain-groq, langchain-core)  ║
║  • LLM       : Groq  (llama-3.1-8b-instant)                         ║
║  • Document  : PyPDF2                                                ║
║  • RAG       : Simple Retrieval-Augmented Generation                 ║
║  • Weather   : OpenWeatherMap REST API                               ║
║  • HTTP      : requests                                              ║
║  • Config    : python-dotenv                                         ║
║  • Deployment: Streamlit Cloud                                       ║
╚══════════════════════════════════════════════════════════════════════╝
"""

# ── STANDARD LIBRARY ─────────────────────────────────────────────────────────
import io
import os
import random
import re
import sqlite3
from datetime import datetime

# ── THIRD-PARTY: HTTP + CONFIG ────────────────────────────────────────────────
import requests
from dotenv import load_dotenv

# ── THIRD-PARTY: UI ───────────────────────────────────────────────────────────
import streamlit as st
st.set_page_config(
    page_title="Wandr AI · Travel Concierge",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)
# ── THIRD-PARTY: DOCUMENT PROCESSING ─────────────────────────────────────────
import PyPDF2

# ── THIRD-PARTY: LLM / LANGCHAIN ─────────────────────────────────────────────
from langchain_groq import ChatGroq
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

# ── LOAD ENV VARS ─────────────────────────────────────────────────────────────
load_dotenv()

def _get_secret(key: str) -> str:
    return os.getenv(key, "")

GROQ_API_KEY    = _get_secret("GROQ_API_KEY")
WEATHER_API_KEY = _get_secret("WEATHER_API_KEY")

# SQLite Database
conn = sqlite3.connect("travel.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS searches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT,
    response TEXT,
    timestamp TEXT
)
""")
conn.commit()

# ══════════════════════════════════════════════════════════════════════════════
# CITY IMAGES
# ══════════════════════════════════════════════════════════════════════════════
UNSPLASH_FALLBACK = [
    "https://images.unsplash.com/photo-1476514525535-07fb3b4ae5f1?w=1600&q=80",
    "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=1600&q=80",
    "https://images.unsplash.com/photo-1469854523086-cc02fe5d8800?w=1600&q=80",
    "https://images.unsplash.com/photo-1488085061387-422e29b40080?w=1600&q=80",
]

CITY_IMAGES = {
    "paris":     "https://images.unsplash.com/photo-1502602898657-3e91760cbb34?w=1600&q=80",
    "london":    "https://images.unsplash.com/photo-1513635269975-59663e0ac1ad?w=1600&q=80",
    "tokyo":     "https://images.unsplash.com/photo-1540959733332-eab4deabeeaf?w=1600&q=80",
    "new york":  "https://images.unsplash.com/photo-1496442226666-8d4d0e62e6e9?w=1600&q=80",
    "dubai":     "https://images.unsplash.com/photo-1512453979798-5ea266f8880c?w=1600&q=80",
    "rome":      "https://images.unsplash.com/photo-1552832230-c0197dd311b5?w=1600&q=80",
    "barcelona": "https://images.unsplash.com/photo-1583422409516-2895a77efded?w=1600&q=80",
    "sydney":    "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=1600&q=80",
    "bali":      "https://images.unsplash.com/photo-1537996194471-e657df975ab4?w=1600&q=80",
    "singapore": "https://images.unsplash.com/photo-1525625293386-3f8f99389edd?w=1600&q=80",
    "maldives":  "https://images.unsplash.com/photo-1514282401047-d79a71a590e8?w=1600&q=80",
    "istanbul":  "https://images.unsplash.com/photo-1524231757912-21f4fe3a7200?w=1600&q=80",
    "amsterdam": "https://images.unsplash.com/photo-1534351590666-13e3e96b5017?w=1600&q=80",
    "santorini": "https://images.unsplash.com/photo-1507501336603-6760ed5e3174?w=1600&q=80",
    "kyoto":     "https://images.unsplash.com/photo-1493976040374-85c8e12f0c0e?w=1600&q=80",
    "bangkok":   "https://images.unsplash.com/photo-1508009603885-50cf7c8dd0d5?w=1600&q=80",
    "cairo":     "https://images.unsplash.com/photo-1572252009286-268acec5ca0a?w=1600&q=80",
    "venice":    "https://images.unsplash.com/photo-1514890547357-a9ee288728e0?w=1600&q=80",
    "miami":     "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=1600&q=80",
    "goa":       "https://images.unsplash.com/photo-1614082242765-7c98ca0f3df3?w=1600&q=80",
    "mumbai":    "https://images.unsplash.com/photo-1529253355930-ddbe423a2ac7?w=1600&q=80",
    "delhi":     "https://images.unsplash.com/photo-1587474260584-136574528ed5?w=1600&q=80",
    "jaipur":    "https://images.unsplash.com/photo-1477587458883-47145ed6a2e0?w=1600&q=80",
    "kerala":    "https://images.unsplash.com/photo-1602216056096-3b40cc0c9944?w=1600&q=80",
}


def get_city_image(city: str) -> str:
    if city:
        for key, url in CITY_IMAGES.items():
            if key in city.lower():
                return url
        clean = city.replace(" ", "%20")
        return f"https://source.unsplash.com/1600x900/?{clean},travel,city"
    return random.choice(UNSPLASH_FALLBACK)


# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
def inject_css(bg_url: str):
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=DM+Sans:wght@300;400;500;600&family=Space+Mono:wght@400;700&display=swap');

:root {{
  --primary:  #FF6B35;
  --accent:   #FFD166;
  --sky:      #5eb8ff;
  --teal:     #2dd4bf;
  --success:  #06D6A0;
  --bg-dark:  #0A0E1A;
  --bg-card:  #111827;
  --bg-card2: #1a2235;
  --text:     #F8F9FA;
  --muted:    #8899AA;
  --border:   rgba(255,107,53,0.25);
  --gold:     #d4a853;
  --gold-lt:  #f0c96b;
  --radius:   16px;
}}

html, body, [class*="css"] {{
  font-family: 'DM Sans', sans-serif !important;
  background-color: var(--bg-dark) !important;
  color: var(--text) !important;
}}

.stApp {{
  background:
    linear-gradient(160deg,rgba(10,14,26,0.92) 0%,rgba(10,14,26,0.80) 50%,rgba(10,14,26,0.95) 100%),
    url("{bg_url}") center/cover fixed;
  font-family: 'DM Sans', sans-serif;
  color: var(--text);
}}

#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding: 2rem 3rem 4rem; max-width: 1280px; }}

/* ── HIDE ALL NATIVE STREAMLIT SIDEBAR TOGGLE BUTTONS ── */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"],
button[kind="header"],
[data-testid="stSidebar"] > div:first-child > div:first-child > button {{
  display: none !important;
  visibility: hidden !important;
  opacity: 0 !important;
  pointer-events: none !important;
}}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {{
  background: linear-gradient(180deg,rgba(8,8,14,0.97) 0%,rgba(18,16,28,0.97) 100%) !important;
  border-right: 1px solid var(--border) !important;
  transition: transform 0.3s ease, width 0.3s ease !important;
}}
[data-testid="stSidebar"] .stMarkdown h3 {{
  color: var(--accent) !important;
  font-family: 'Playfair Display', serif;
}}

/* ── SIDEBAR COLLAPSED STATE (driven by checkbox) ── */
#wandr-sidebar-toggle:checked ~ * [data-testid="stSidebar"],
body.sidebar-collapsed [data-testid="stSidebar"] {{
  width: 0 !important;
  min-width: 0 !important;
  overflow: hidden !important;
  transform: translateX(-100%) !important;
}}

/* ── HAMBURGER MENU BUTTON ── */
#wandr-hamburger-label {{
  position: fixed;
  top: 14px;
  left: 14px;
  z-index: 99999;
  background: linear-gradient(135deg, #1a2235, #0d1a30);
  border: 1px solid rgba(212,168,83,0.45);
  border-radius: 10px;
  width: 42px;
  height: 42px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 5px;
  cursor: pointer;
  box-shadow: 0 4px 18px rgba(0,0,0,0.45);
  transition: all 0.2s ease;
  user-select: none;
}}
#wandr-hamburger-label:hover {{
  background: linear-gradient(135deg, #223050, #122040);
  border-color: rgba(212,168,83,0.75);
  transform: translateY(-1px);
  box-shadow: 0 6px 22px rgba(0,0,0,0.55);
}}
#wandr-hamburger-label .bar {{
  width: 20px;
  height: 2px;
  background: #d4a853;
  border-radius: 2px;
  transition: all 0.25s ease;
  display: block;
}}

/* ── HERO ── */
.hero-title {{
  font-family: 'Playfair Display', serif;
  font-size: 3.2rem;
  font-weight: 700;
  color: var(--gold);
  letter-spacing: -1px;
  line-height: 1;
}}
.hero-sub {{
  font-size: 0.75rem;
  color: rgba(240,236,228,0.4);
  letter-spacing: 4px;
  text-transform: uppercase;
}}
.hero-divider {{
  width: 60px; height: 2px;
  background: linear-gradient(90deg, var(--gold), transparent);
  margin: 10px 0 20px;
}}

/* ── DESTINATION BANNER ── */
.dest-banner {{
  background: linear-gradient(135deg,rgba(212,168,83,0.15),rgba(94,184,255,0.10));
  border: 1px solid rgba(212,168,83,0.3);
  border-radius: var(--radius);
  padding: 16px 22px;
  margin-bottom: 18px;
  display: flex; align-items: center; gap: 14px;
}}
.dest-banner-city {{
  font-family: 'Playfair Display', serif;
  font-size: 1.35rem;
  color: var(--gold-lt);
  font-style: italic;
}}
.dest-banner-meta {{
  font-size: 0.72rem;
  color: rgba(240,236,228,0.4);
  letter-spacing: 2px;
  text-transform: uppercase;
}}

/* ── WEATHER CARD ── */
.weather-card {{
  background: linear-gradient(135deg,#1a2235,#0d2040);
  border: 1px solid rgba(0,78,137,0.4);
  border-radius: 20px;
  padding: 1.6rem;
  text-align: center;
  margin-bottom: 1rem;
}}
.weather-temp  {{ font-family:'Space Mono',monospace; font-size:3.2rem; font-weight:700; color:var(--accent); line-height:1; }}
.weather-city  {{ font-family:'Playfair Display',serif; font-size:1.4rem; font-weight:700; color:var(--text); margin-bottom:4px; }}
.weather-meta  {{ display:flex; justify-content:center; gap:1.4rem; margin-top:.9rem; font-size:.83rem; color:var(--muted); }}
.map-link {{
  display:inline-flex; align-items:center; gap:6px;
  background:rgba(45,212,191,0.12); border:1px solid rgba(45,212,191,0.3);
  border-radius:50px; padding:5px 15px; font-size:.78rem; font-weight:500;
  color:var(--teal) !important; text-decoration:none; margin-top:8px;
}}

/* ── CARDS ── */
.card {{
  background:var(--bg-card); border:1px solid var(--border);
  border-radius:var(--radius); padding:1.5rem; margin-bottom:1rem;
}}

/* ── FEATURE PILLS ── */
.feature-pill {{
  display:inline-block;
  background:rgba(255,107,53,0.12); border:1px solid rgba(255,107,53,0.35);
  color:var(--primary); border-radius:50px; padding:.28rem .9rem;
  font-size:.78rem; font-weight:600; margin:.2rem; letter-spacing:.04em;
}}

/* ── STATUS BADGES ── */
.status-badge {{ display:inline-flex; align-items:center; gap:.4rem; border-radius:50px; padding:.28rem .95rem; font-size:.76rem; font-weight:600; }}
.status-pdf    {{ background:rgba(212,168,83,0.12); border:1px solid rgba(212,168,83,0.3); color:var(--gold); }}
.status-travel {{ background:rgba(6,214,160,0.12);  border:1px solid rgba(6,214,160,0.35); color:var(--success); }}
.status-dot    {{ width:7px; height:7px; background:var(--success); border-radius:50%; animation:pulse 1.5s ease-in-out infinite; }}
@keyframes pulse {{ 0%,100%{{opacity:1;transform:scale(1)}} 50%{{opacity:.5;transform:scale(1.3)}} }}

/* ── METRICS ── */
.metric-box   {{ background:var(--bg-card2); border:1px solid var(--border); border-radius:12px; padding:1.1rem; text-align:center; margin-bottom:.6rem; }}
.metric-val   {{ font-family:'Space Mono',monospace; font-size:1.9rem; font-weight:700; color:var(--accent); }}
.metric-label {{ font-size:.75rem; color:var(--muted); text-transform:uppercase; letter-spacing:.1em; margin-top:.15rem; }}

/* ── CHAT ── */
.chat-container {{
  max-height:500px; overflow-y:auto; padding:1rem;
  background:var(--bg-card); border-radius:var(--radius);
  border:1px solid var(--border); margin-bottom:1rem;
  scrollbar-width:thin; scrollbar-color:var(--primary) transparent;
}}
.msg-user {{ display:flex; justify-content:flex-end; margin:.65rem 0; }}
.msg-user .bubble {{
  background:linear-gradient(135deg,var(--primary),#e85d2a);
  color:white; border-radius:18px 18px 4px 18px;
  padding:.7rem 1.1rem; max-width:76%; font-size:.93rem;
  box-shadow:0 4px 14px rgba(255,107,53,.3);
}}
.msg-ai {{ display:flex; justify-content:flex-start; margin:.65rem 0; gap:.6rem; align-items:flex-start; }}
.ai-avatar {{
  width:34px; height:34px;
  background:linear-gradient(135deg,#004e89,#0077b6);
  border-radius:50%; display:flex; align-items:center; justify-content:center;
  font-size:.95rem; flex-shrink:0;
}}
.msg-ai .bubble {{
  background:var(--bg-card2); color:var(--text);
  border-radius:4px 18px 18px 18px;
  padding:.7rem 1.1rem; max-width:80%; font-size:.93rem;
  border:1px solid rgba(255,255,255,.06); line-height:1.65;
}}

/* ── BUTTONS ── */
.stButton > button {{
  background:linear-gradient(135deg,var(--primary),#e85d2a) !important;
  color:white !important; border:none !important; border-radius:50px !important;
  font-family:'DM Sans',sans-serif !important; font-weight:600 !important;
  font-size:.87rem !important; letter-spacing:.03em !important;
  padding:.55rem 1.5rem !important;
  box-shadow:0 4px 14px rgba(255,107,53,.3) !important;
  transition:all .2s ease !important;
}}
.stButton > button:hover {{ transform:translateY(-2px) !important; box-shadow:0 8px 24px rgba(255,107,53,.5) !important; }}

/* ── INPUTS ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {{
  background:var(--bg-card2) !important; border:1px solid var(--border) !important;
  border-radius:10px !important; color:var(--text) !important;
  font-family:'DM Sans',sans-serif !important;
}}
.stTextInput input::placeholder, .stTextArea textarea::placeholder {{ color:#B8C0CC !important; opacity:1 !important; }}
.stTextInput input, .stTextArea textarea {{ color:white !important; font-weight:500 !important; }}
.stSelectbox > div > div {{ background:var(--bg-card2) !important; border:1px solid var(--border) !important; border-radius:10px !important; color:var(--text) !important; }}
[data-testid="stFileUploader"] {{ background:var(--bg-card) !important; border:1px dashed rgba(212,168,83,.35) !important; border-radius:10px !important; padding:10px !important; }}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {{ background:var(--bg-card) !important; border-radius:12px !important; padding:.28rem !important; gap:.28rem !important; border:1px solid var(--border) !important; }}
.stTabs [data-baseweb="tab"] {{ background:transparent !important; color:var(--muted) !important; border-radius:8px !important; font-family:'DM Sans',sans-serif !important; font-weight:500 !important; font-size:.88rem !important; padding:.45rem 1.1rem !important; }}
.stTabs [aria-selected="true"] {{ background:linear-gradient(135deg,var(--primary),#e85d2a) !important; color:white !important; }}

/* ── MISC ── */
.stSuccess {{ background:rgba(6,214,160,.1) !important; border-left:3px solid var(--success) !important; border-radius:8px !important; }}
.stError   {{ background:rgba(255,107,107,.1) !important; border-left:3px solid #FF6B6B !important; border-radius:8px !important; }}
.stInfo    {{ background:rgba(0,78,137,.15) !important; border-left:3px solid #004e89 !important; border-radius:8px !important; }}
.stWarning {{ background:rgba(255,209,102,.1) !important; border-left:3px solid var(--accent) !important; border-radius:8px !important; }}
[data-testid="stExpander"] {{ background:var(--bg-card) !important; border:1px solid var(--border) !important; border-radius:10px !important; }}
hr {{ border-color:var(--border) !important; margin:1.4rem 0 !important; }}
label,
.stMarkdown,
.stMarkdown p,
.stMarkdown div,
.stMarkdown span,
.stMarkdown li {{
    color: white !important;
}}
.stMarkdown h1 {{ color:var(--accent) !important; }}
.stMarkdown h2 {{ color:var(--primary) !important; }}
.stMarkdown h3, .stMarkdown h4 {{ color:#d4a853 !important; font-family:'Playfair Display',serif !important; }}
.stDownloadButton > button {{
    color: black !important;
    font-weight: 600 !important;
}}
::-webkit-scrollbar {{ width:4px; }}
::-webkit-scrollbar-thumb {{ background:var(--primary); border-radius:2px; }}
</style>

<!-- ── HAMBURGER: Pure JS approach that polls for the button reliably on deployed apps ── -->
<div id="wandr-hamburger-label" onclick="wandrToggleSidebar()" title="Toggle Menu">
  <span class="bar"></span>
  <span class="bar"></span>
  <span class="bar"></span>
</div>

<script>
(function() {{
  var isOpen = true;

  function wandrToggleSidebar() {{
    var btn = document.getElementById('wandr-hamburger-label');

    // Strategy 1: Try all known Streamlit sidebar button selectors
    var selectors = [
      '[data-testid="collapsedControl"]',
      '[data-testid="stSidebarCollapseButton"]',
      'button[aria-label="Close sidebar"]',
      'button[aria-label="Open sidebar"]',
      'button[aria-label="collapse sidebar"]',
      'button[aria-label="expand sidebar"]',
      '[data-testid="stSidebar"] > div > button',
      'section[data-testid="stSidebar"] button:first-of-type',
    ];

    var clicked = false;
    for (var i = 0; i < selectors.length; i++) {{
      var el = document.querySelector(selectors[i]);
      if (el) {{
        el.click();
        clicked = true;
        break;
      }}
    }}

    // Strategy 2: If no native button found, directly hide/show sidebar via style
    if (!clicked) {{
      var sidebar = document.querySelector('[data-testid="stSidebar"]');
      if (sidebar) {{
        if (isOpen) {{
          sidebar.style.cssText += '; width: 0 !important; min-width: 0 !important; overflow: hidden !important; transform: translateX(-300px) !important; transition: all 0.3s ease !important;';
          isOpen = false;
        }} else {{
          sidebar.style.cssText = sidebar.style.cssText
            .replace(/width:[^;]+;/g, '')
            .replace(/min-width:[^;]+;/g, '')
            .replace(/overflow:[^;]+;/g, '')
            .replace(/transform:[^;]+;/g, '');
          sidebar.style.removeProperty('width');
          sidebar.style.removeProperty('min-width');
          sidebar.style.removeProperty('overflow');
          sidebar.style.removeProperty('transform');
          isOpen = true;
        }}
      }}
    }}

    // Animate hamburger bars
    if (btn) {{
      btn.classList.toggle('open');
      var bars = btn.querySelectorAll('.bar');
      if (btn.classList.contains('open')) {{
        if (bars[0]) bars[0].style.cssText = 'transform: translateY(7px) rotate(45deg);';
        if (bars[1]) bars[1].style.cssText = 'opacity: 0; transform: scaleX(0);';
        if (bars[2]) bars[2].style.cssText = 'transform: translateY(-7px) rotate(-45deg);';
      }} else {{
        if (bars[0]) bars[0].style.cssText = '';
        if (bars[1]) bars[1].style.cssText = '';
        if (bars[2]) bars[2].style.cssText = '';
      }}
    }}
  }}

  window.wandrToggleSidebar = wandrToggleSidebar;

  // Also sync hamburger state when sidebar is toggled by other means
  // (e.g. user resizes window, presses Escape, etc.)
  function observeSidebarState() {{
    var sidebar = document.querySelector('[data-testid="stSidebar"]');
    if (!sidebar) {{
      setTimeout(observeSidebarState, 500);
      return;
    }}
    var observer = new MutationObserver(function(mutations) {{
      mutations.forEach(function(m) {{
        if (m.type === 'attributes' && m.attributeName === 'aria-expanded') {{
          var expanded = sidebar.getAttribute('aria-expanded');
          var btn = document.getElementById('wandr-hamburger-label');
          if (btn) {{
            if (expanded === 'false') {{
              btn.classList.add('open');
              isOpen = false;
            }} else {{
              btn.classList.remove('open');
              isOpen = true;
            }}
          }}
        }}
      }});
    }});
    observer.observe(sidebar, {{ attributes: true, attributeFilter: ['aria-expanded', 'style'] }});
  }}

  // Wait for DOM to be ready
  if (document.readyState === 'loading') {{
    document.addEventListener('DOMContentLoaded', observeSidebarState);
  }} else {{
    observeSidebarState();
  }}
}})();
</script>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ── BACKEND: OPENWEATHERMAP API
# ══════════════════════════════════════════════════════════════════════════════
def get_weather(city: str) -> dict | None:
    if not WEATHER_API_KEY:
        return None
    try:
        url = (
            "http://api.openweathermap.org/data/2.5/weather"
            f"?q={city}&appid={WEATHER_API_KEY}&units=metric"
        )
        res = requests.get(url, timeout=6).json()
        if res.get("cod") != 200:
            return None
        desc = res["weather"][0]["description"]
        return {
            "city":        res["name"],
            "country":     res["sys"]["country"],
            "temp":        round(res["main"]["temp"]),
            "feels_like":  round(res["main"]["feels_like"]),
            "description": desc,
            "humidity":    res["main"]["humidity"],
            "wind":        round(res["wind"]["speed"] * 3.6),
            "pressure":    res["main"].get("pressure", "N/A"),
            "clouds":      res["clouds"].get("all", "N/A"),
            "visibility":  round(res.get("visibility", 0) / 1000, 1),
            "icon":        _weather_emoji(desc),
        }
    except Exception:
        return None


def _weather_emoji(desc: str) -> str:
    d = desc.lower()
    if "clear" in d or "sunny" in d:    return "☀️"
    if "cloud" in d:                    return "⛅"
    if "rain" in d or "drizzle" in d:   return "🌧️"
    if "storm" in d or "thunder" in d:  return "⛈️"
    if "snow" in d:                     return "❄️"
    if "fog" in d or "mist" in d:       return "🌫️"
    return "🌤️"


# ══════════════════════════════════════════════════════════════════════════════
# ── BACKEND: LANGCHAIN + GROQ
# ══════════════════════════════════════════════════════════════════════════════
def _build_llm(temperature: float = 0.7, max_tokens: int = 1200) -> ChatGroq:
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
        temperature=temperature,
        max_tokens=max_tokens,
    )


def detect_city_from_text(text: str) -> str | None:
    if not GROQ_API_KEY:
        return None
    try:
        llm = _build_llm(temperature=0, max_tokens=20)
        messages = [
            SystemMessage(content="Extract ONLY the city or place name from the sentence. If none, return NONE. Return just the name."),
            HumanMessage(content=text),
        ]
        result = llm.invoke(messages)
        city = result.content.strip().split(",")[0].replace(".", "").strip()
        return None if city.upper() == "NONE" else city
    except Exception:
        return None


def get_travel_response(user_input: str, history: list, context: str = "") -> str:
    if not GROQ_API_KEY:
        return "⚠️ GROQ_API_KEY is not configured. Check your .env file."
    try:
        llm = _build_llm(temperature=0.7, max_tokens=1200)

        system_content = (
            "You are Wandr, a luxury travel concierge with deep expertise "
            "in crafting unforgettable journeys. Be vivid, helpful, and inspiring."
        )
        if context:
            system_content += (
                f"\n\nAnswer using ONLY this document context:\n{context}\n"
                "If the document doesn't contain the answer, say so politely."
            )

        messages: list = [SystemMessage(content=system_content)]
        for m in history[-6:]:
            if m["role"] == "user":
                messages.append(HumanMessage(content=m["content"]))
            else:
                messages.append(AIMessage(content=m["content"]))
        messages.append(HumanMessage(content=user_input))

        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        return f"LLM Error: {e}"


def get_itinerary(destination: str, days: int, budget: str,
                  style: str, interests: list, weather_info: str) -> str:
    if not GROQ_API_KEY:
        return "⚠️ GROQ_API_KEY is not configured."
    try:
        llm = _build_llm(temperature=0.8, max_tokens=2000)
        interests_str = ", ".join(interests) if interests else "general sightseeing"
        prompt = f"""You are Wandr, a luxury travel concierge. Create an immersive, detailed {days}-day itinerary for {destination}.

Travel style: {style} | Budget: {budget} | Interests: {interests_str}
{f"Current weather: {weather_info}" if weather_info else ""}

Structure:
**Overview** — inspiring 2-3 sentence intro.

**Day-by-Day Itinerary**
Day N: [Theme Title]
Morning / Afternoon / Evening — specific activities and why they're special.
Stay: [Hotel recommendation]

**Practical Travel Intelligence**
- 💰 Budget Estimate
- 🗓️ Best Time to Visit
- ✈️ Getting There
- 🎒 Packing Essentials (destination-specific)
- 🤝 Cultural Tips
- ⚡ Insider Secret

Write with vivid, evocative language."""

        messages = [HumanMessage(content=prompt)]
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        return f"Error generating itinerary: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# ── RAG: DOCUMENT PROCESSING
# ══════════════════════════════════════════════════════════════════════════════
def index_pdf(file_bytes: bytes) -> list[str]:
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        raw_pages: list[str] = []
        for page in reader.pages:
            text = page.extract_text()
            if text and text.strip():
                raw_pages.append(text)

        if not raw_pages:
            return []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        full_text = "\n\n".join(raw_pages)
        chunks = splitter.split_text(full_text)
        return chunks

    except Exception:
        return []


def search_pdf_context(chunks: list[str], query: str, top_k: int = 3) -> str:
    if not chunks:
        return ""
    query_words = set(query.lower().split())
    scored = []
    for chunk in chunks:
        score = sum(1 for w in query_words if w in chunk.lower())
        scored.append((score, chunk))
    scored.sort(key=lambda x: -x[0])
    return "\n\n---\n\n".join(chunk for _, chunk in scored[:top_k])


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
_defaults = {
    "chat_messages":      [],
    "pdf_chunks":         None,
    "pdf_page_count":     0,
    "current_place":      None,
    "bg_url":             random.choice(UNSPLASH_FALLBACK),
    "weather_data":       None,
    "itinerary_result":   None,
    "itinerary_dest":     None,
    "history":            [],
    "saved_itineraries":  [],
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Apply CSS + Hamburger ─────────────────────────────────────────────────────
inject_css(st.session_state.bg_url)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
<div style="text-align:center;padding:20px 0 10px;">
  <div style="font-family:'Playfair Display',serif;font-size:2rem;color:#d4a853;font-weight:700;">✈ Wandr</div>
  <div style="font-size:.68rem;color:rgba(240,236,228,.35);letter-spacing:3px;text-transform:uppercase;margin-top:4px;">AI Travel Concierge</div>
</div>
<div style="display:flex;justify-content:center;margin-bottom:12px;">
  <span style="display:inline-flex;align-items:center;gap:6px;background:rgba(6,214,160,.12);border:1px solid rgba(6,214,160,.35);color:#06D6A0;border-radius:50px;padding:4px 13px;font-size:.73rem;font-weight:600;">
    <span style="width:7px;height:7px;background:#06D6A0;border-radius:50%;display:inline-block;"></span>Online & Ready
  </span>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    # ── PDF UPLOAD
    st.markdown("### 📄 Travel Document (PDF)")
    uploaded_file = st.file_uploader(
        "Upload PDF (itinerary, visa docs, guides…)",
        type="pdf",
        label_visibility="collapsed",
    )
    if uploaded_file and st.session_state.pdf_chunks is None:
        with st.spinner("📄 Indexing PDF with PyPDF2 + LangChain RAG…"):
            chunks = index_pdf(uploaded_file.read())
            if chunks:
                st.session_state.pdf_chunks     = chunks
                st.session_state.pdf_page_count = len(chunks)
                st.success(f"✅ Indexed {len(chunks)} chunks (RAG ready)")
            else:
                st.error("❌ Could not extract text from PDF")

    if st.session_state.pdf_chunks:
        st.markdown(
            f'<span class="status-badge status-pdf">📄 PDF Loaded — '
            f'{st.session_state.pdf_page_count} RAG chunks</span>',
            unsafe_allow_html=True,
        )
        if st.button("🗑️ Remove PDF", use_container_width=True):
            st.session_state.pdf_chunks     = None
            st.session_state.pdf_page_count = 0
            st.rerun()

    st.markdown("---")

    # ── QUICK WEATHER
    st.markdown("### 🌍 Quick Weather Check")
    quick_city = st.text_input("City name", placeholder="e.g. Tokyo", key="sidebar_city")
    if st.button("🌤️ Check Weather", use_container_width=True):
        if not WEATHER_API_KEY:
            st.warning("⚠️ Add WEATHER_API_KEY to your .env file")
        elif quick_city.strip():
            with st.spinner("Fetching from OpenWeatherMap…"):
                wdata = get_weather(quick_city)
            if wdata:
                st.session_state.weather_data  = wdata
                st.session_state.bg_url        = get_city_image(quick_city)
                st.session_state.current_place = quick_city.title()
                st.rerun()
            else:
                st.error("City not found!")
        else:
            st.warning("Enter a city name first")

    st.markdown("---")

    # ── QUICK DESTINATIONS
    st.markdown("### 🎯 Quick Destinations")
    quick_cities = [
        ("Paris 🗼", "paris"), ("Tokyo 🌸", "tokyo"), ("Bali 🌴", "bali"),
        ("Dubai 🏙️", "dubai"), ("Santorini 🌊", "santorini"), ("Kyoto ⛩️", "kyoto"),
    ]
    for label, city_key in quick_cities:
        if st.button(label, use_container_width=True, key=f"quick_{city_key}"):
            st.session_state.current_place = city_key.title()
            st.session_state.bg_url        = get_city_image(city_key)
            prompt = f"Plan a 5-day trip to {city_key.title()}"
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            if GROQ_API_KEY:
                with st.spinner("Planning your trip…"):
                    reply = get_travel_response(prompt, [])
                st.session_state.chat_messages.append({"role": "assistant", "content": reply})
                st.session_state.history.append({
                    "query":     prompt,
                    "response":  reply,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                })
            st.rerun()

    st.markdown("---")

    # ── STATS
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div class="metric-box"><div class="metric-val">{len(st.session_state.history)}</div><div class="metric-label">Searches</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-box"><div class="metric-val">{len(st.session_state.saved_itineraries)}</div><div class="metric-label">Saved</div></div>', unsafe_allow_html=True)

    if st.button("🧹 Clear All Session Data", use_container_width=True):
        for k in ["chat_messages", "pdf_chunks", "current_place",
                  "weather_data", "itinerary_result", "itinerary_dest", "history"]:
            st.session_state[k] = [] if k in ["chat_messages", "history"] else None
        st.session_state.bg_url = random.choice(UNSPLASH_FALLBACK)
        st.rerun()

    st.markdown(f"""
<div style="font-size:.68rem;color:rgba(240,236,228,.28);text-align:center;letter-spacing:1px;margin-top:10px;">
  WANDR AI · {datetime.now().strftime('%B %Y')}
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════

# ── HEADER
st.markdown("""
<div style="display:flex;align-items:baseline;gap:12px;margin-bottom:4px;">
  <div class="hero-title">Wandr</div>
  <div class="hero-sub">AI Travel Concierge</div>
</div>
<div class="hero-divider"></div>
""", unsafe_allow_html=True)

# ── DESTINATION BANNER
if st.session_state.current_place:
    st.markdown(f"""
<div class="dest-banner">
  <span style="font-size:2rem;">🌍</span>
  <div>
    <div class="dest-banner-city">{st.session_state.current_place.title()}</div>
    <div class="dest-banner-meta">Current Destination</div>
  </div>
</div>""", unsafe_allow_html=True)

# ── MODE BADGE
if st.session_state.pdf_chunks:
    st.markdown('<span class="status-badge status-pdf">📄 RAG Mode — Answering from your PDF document</span>', unsafe_allow_html=True)
else:
    st.markdown('<span class="status-badge status-travel"><span class="status-dot"></span> Travel Mode — Ask me anything about your trip</span>', unsafe_allow_html=True)

# ── TECH PILLS
st.markdown("""
<div style="text-align:center;margin:14px 0 22px;">
  <span class="feature-pill">🌤️ OpenWeatherMap</span>
  <span class="feature-pill">🗺️ Itineraries</span>
  <span class="feature-pill">✈️ Travel Tips</span>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_chat, tab_itin, tab_weather, tab_history = st.tabs(
    ["💬 Chat Assistant", "🗺️ Itinerary Planner", "🌤️ Weather Explorer", "📜 History"]
)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — CHAT ASSISTANT
# ─────────────────────────────────────────────────────────────────────────────
with tab_chat:
    col_chat, col_tips = st.columns([2, 1])

    with col_chat:
        st.markdown("### 💬 Ask Anything About Travel")

        chat_html = '<div class="chat-container">'
        if not st.session_state.chat_messages:
            chat_html += """
<div style="text-align:center;padding:2.5rem;color:#8899AA;">
  <div style="font-size:3rem;margin-bottom:1rem;">🌍</div>
  <div style="font-family:'Playfair Display',serif;font-size:1.25rem;color:#F8F9FA;">Where would you like to go?</div>
  <div style="font-size:.88rem;margin-top:.5rem;">Ask about destinations, packing, visas, budgets — or upload a PDF to chat with it via RAG.</div>
</div>"""
        else:
            for msg in st.session_state.chat_messages:
                safe = (msg["content"]
                        .replace("&", "&amp;")
                        .replace("<", "&lt;")
                        .replace("\n", "<br>"))
                if msg["role"] == "user":
                    chat_html += f'<div class="msg-user"><div class="bubble">{safe}</div></div>'
                else:
                    chat_html += f'<div class="msg-ai"><div class="ai-avatar">✈️</div><div class="bubble">{safe}</div></div>'
        chat_html += "</div>"
        st.markdown(chat_html, unsafe_allow_html=True)

        user_input = st.text_input(
            "Question",
            placeholder="e.g. What are the top 5 things to do in Bali? / Ask about your uploaded PDF…",
            label_visibility="collapsed",
            key="chat_input",
        )
        col_send, col_clear = st.columns([3, 1])
        with col_send:
            send_btn = st.button("✈️ Send Message", use_container_width=True)
        with col_clear:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.chat_messages = []
                st.rerun()

        if send_btn and user_input.strip():
            if not GROQ_API_KEY:
                st.error("⚠️ GROQ_API_KEY is not set in your .env or Streamlit secrets!")
            else:
                st.session_state.chat_messages.append({"role": "user", "content": user_input})

                city = detect_city_from_text(user_input)
                if city:
                    st.session_state.current_place = city.title()
                    st.session_state.bg_url        = get_city_image(city)
                    if WEATHER_API_KEY:
                        wdata = get_weather(city)
                        if wdata:
                            st.session_state.weather_data = wdata

                context = ""
                if st.session_state.pdf_chunks:
                    context = search_pdf_context(
                        st.session_state.pdf_chunks, user_input
                    )

                with st.spinner("✈️ Crafting your answer with Groq LLM…"):
                    reply = get_travel_response(
                        user_input,
                        st.session_state.chat_messages[:-1],
                        context=context,
                    )

                st.session_state.chat_messages.append({"role": "assistant", "content": reply})
                st.session_state.history.append({
                    "query":     user_input,
                    "response":  reply,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                })
                cursor.execute(
                    "INSERT INTO searches(query, response, timestamp) VALUES (?, ?, ?)",
                    (user_input, reply, datetime.now().strftime("%Y-%m-%d %H:%M"))
                )
                conn.commit()
                st.rerun()

        if st.session_state.weather_data:
            w = st.session_state.weather_data
            st.markdown(f"""
<div class="weather-card">
  <div style="font-size:2.8rem;">{w['icon']}</div>
  <div class="weather-city">📍 {w['city']}, {w['country']}</div>
  <div class="weather-temp">{w['temp']}°C</div>
  <div style="color:#8899AA;font-size:.83rem;text-transform:capitalize;margin-top:4px;">{w['description']}</div>
  <div class="weather-meta">
    <span>🌡️ Feels {w['feels_like']}°C</span>
    <span>💧 {w['humidity']}%</span>
    <span>💨 {w['wind']} km/h</span>
  </div>
</div>
<a class="map-link" href="https://www.google.com/maps/search/?api=1&query={w['city']}" target="_blank">🗺️ Open in Google Maps</a>
""", unsafe_allow_html=True)

    with col_tips:
        st.markdown("### 💡 Quick Prompts")
        quick_prompts = [
            ("🏖️", "Best beaches in Southeast Asia?"),
            ("💰", "Budget travel tips for Europe?"),
            ("🎒", "Essential packing list for 2 weeks?"),
            ("🛂", "Visa requirements for Japan?"),
            ("🍜", "Street food guide for Thailand?"),
            ("⛰️", "Best trekking destinations in Asia?"),
            ("🏨", "How to find budget hostels?"),
            ("✈️", "Tips for long-haul flights?"),
        ]
        for icon, prompt in quick_prompts:
            if st.button(f"{icon} {prompt[:28]}", key=f"qp_{prompt[:10]}", use_container_width=True):
                if not GROQ_API_KEY:
                    st.error("GROQ_API_KEY not set!")
                else:
                    st.session_state.chat_messages.append({"role": "user", "content": prompt})
                    with st.spinner("Thinking…"):
                        reply = get_travel_response(prompt, st.session_state.chat_messages[:-1])
                    st.session_state.chat_messages.append({"role": "assistant", "content": reply})
                    st.session_state.history.append({"query": prompt, "response": reply, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")})
                    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — ITINERARY PLANNER
# ─────────────────────────────────────────────────────────────────────────────
with tab_itin:
    st.markdown("### 🗺️ Build Your Perfect Itinerary")
    col_form, col_result = st.columns([1, 1.6])

    with col_form:
        destination = st.text_input("📍 Destination", placeholder="e.g. Paris, France", key="itin_dest")
        days        = st.slider("📅 Duration (days)", 1, 21, 7)

        cb, cs = st.columns(2)
        with cb:
            budget = st.selectbox("💰 Budget", ["Mid-range", "Luxury"])
        with cs:
            style = st.selectbox("🎭 Style", ["Culture & History", "Adventure", "Food & Nightlife", "Relaxation", "Family", "Backpacker"])

        interests = st.multiselect(
            "🎯 Interests",
            ["Museums", "Beaches", "Mountains", "Street Food", "Nightlife",
             "Shopping", "Architecture", "Nature", "Photography", "Local Markets"],
        )

        weather_info = ""
        if WEATHER_API_KEY and destination.strip():
            wdata = get_weather(destination.split(",")[0])
            if wdata:
                weather_info = f"{wdata['temp']}°C, {wdata['description']}"
                st.markdown(f"""
<div style="background:rgba(255,209,102,.08);border:1px solid rgba(255,209,102,.2);border-radius:8px;padding:.65rem;font-size:.83rem;margin-top:.4rem;">
  {wdata['icon']} Live weather: <b style="color:#FFD166;">{wdata['temp']}°C</b>, {wdata['description'].title()}
</div>""", unsafe_allow_html=True)

        gen_btn = st.button("🗺️ Generate Itinerary", use_container_width=True)

    with col_result:
        if gen_btn:
            if not GROQ_API_KEY:
                st.error("⚠️ GROQ_API_KEY is not set!")
            elif not destination.strip():
                st.warning("Please enter a destination!")
            else:
                st.session_state.current_place = destination.split(",")[0].strip().title()
                st.session_state.bg_url        = get_city_image(destination.split(",")[0].strip())
                with st.spinner(f"✈️ Crafting your {days}-day {destination} adventure with Groq LLM…"):
                    itin = get_itinerary(destination, days, budget, style, interests, weather_info)
                st.session_state.itinerary_result = itin
                st.session_state.itinerary_dest   = destination
                st.session_state.history.append({
                    "query":     f"Itinerary: {destination} ({days}d, {budget}, {style})",
                    "response":  itin,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                })

        if st.session_state.itinerary_result:
            st.markdown(f"""
<div class="card">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;">
    <div style="font-family:'Playfair Display',serif;font-size:1.3rem;font-weight:700;">{st.session_state.itinerary_dest}</div>
    <span class="feature-pill">✨ AI Generated</span>
  </div>""", unsafe_allow_html=True)
            st.markdown(st.session_state.itinerary_result)
            st.markdown("</div>", unsafe_allow_html=True)

            cs1, cs2 = st.columns(2)
            with cs1:
                if st.button("💾 Save Itinerary", use_container_width=True):
                    st.session_state.saved_itineraries.append({
                        "destination": st.session_state.itinerary_dest,
                        "content":     st.session_state.itinerary_result,
                        "days":        days,
                        "created_at":  datetime.now().strftime("%Y-%m-%d %H:%M"),
                    })
                    st.success("✅ Saved!")
            with cs2:
                if st.button("🗑️ Clear", use_container_width=True, key="clear_itin"):
                    st.session_state.itinerary_result = None
                    st.rerun()
            st.download_button(
                label="📥 Download Itinerary",
                data=st.session_state.itinerary_result,
                file_name="travel_itinerary.txt",
                mime="text/plain",
                use_container_width=True
            )
        else:
            st.markdown("""
<div class="card" style="text-align:center;padding:3rem;min-height:340px;display:flex;flex-direction:column;align-items:center;justify-content:center;">
  <div style="font-size:4rem;margin-bottom:1rem;">🗺️</div>
  <div style="font-family:'Playfair Display',serif;font-size:1.25rem;color:#F8F9FA;margin-bottom:.5rem;">Your itinerary will appear here</div>
  <div style="color:#8899AA;font-size:.88rem;">Fill in the details and click Generate</div>
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — WEATHER EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
with tab_weather:
    st.markdown("### 🌤️ Weather Explorer")
    cw1, cw2 = st.columns([1, 2])

    with cw1:
        city_input     = st.text_input("🏙️ Search City", placeholder="e.g. New York", key="wx_city")
        search_weather = st.button("🔍 Get Weather", use_container_width=True)

        st.markdown("#### 🌍 Popular Destinations")
        popular = [
            ("Tokyo 🇯🇵", "Tokyo"), ("Paris 🇫🇷", "Paris"),
            ("Dubai 🇦🇪", "Dubai"), ("Singapore 🇸🇬", "Singapore"),
            ("New York 🇺🇸", "New York"), ("London 🇬🇧", "London"),
            ("Bali 🇮🇩", "Bali"), ("Sydney 🇦🇺", "Sydney"),
        ]
        for lbl, city_name in popular:
            if st.button(lbl, key=f"pop_{city_name}", use_container_width=True):
                if not WEATHER_API_KEY:
                    st.warning("WEATHER_API_KEY not set!")
                else:
                    with st.spinner(f"Fetching {city_name} weather from OpenWeatherMap…"):
                        wdata = get_weather(city_name)
                    if wdata:
                        st.session_state.weather_data  = wdata
                        st.session_state.bg_url        = get_city_image(city_name)
                        st.session_state.current_place = city_name
                    else:
                        st.error("Could not fetch weather")

    with cw2:
        if search_weather and city_input.strip():
            if not WEATHER_API_KEY:
                st.error("WEATHER_API_KEY not set in .env!")
            else:
                with st.spinner("Fetching weather data from OpenWeatherMap API…"):
                    wdata = get_weather(city_input)
                if wdata:
                    st.session_state.weather_data  = wdata
                    st.session_state.bg_url        = get_city_image(city_input)
                    st.session_state.current_place = city_input.title()
                else:
                    st.error("❌ City not found!")

        if st.session_state.weather_data:
            w = st.session_state.weather_data
            st.markdown(f"""
<div class="weather-card" style="margin-bottom:1rem;">
  <div style="font-size:4.5rem;line-height:1;">{w['icon']}</div>
  <div class="weather-city" style="margin-top:.4rem;">{w['city']}, {w['country']}</div>
  <div class="weather-temp">{w['temp']}°C</div>
  <div style="color:#8899AA;font-size:.95rem;margin-top:.3rem;">{w['description'].title()}</div>
</div>
<a class="map-link" href="https://www.google.com/maps/search/?api=1&query={w['city']}" target="_blank">🗺️ Open in Google Maps</a>
""", unsafe_allow_html=True)

            details = [
                ("🌡️", "Feels Like",  f"{w['feels_like']}°C"),
                ("💧", "Humidity",    f"{w['humidity']}%"),
                ("💨", "Wind Speed",  f"{w['wind']} km/h"),
                ("📊", "Pressure",    f"{w['pressure']} hPa"),
                ("☁️", "Cloudiness",  f"{w['clouds']}%"),
                ("👁️", "Visibility",  f"{w['visibility']} km"),
            ]
            cols3 = st.columns(3)
            for i, (icon, label, val) in enumerate(details):
                with cols3[i % 3]:
                    st.markdown(f"""
<div class="metric-box">
  <div style="font-size:1.3rem;">{icon}</div>
  <div class="metric-val" style="font-size:1.35rem;">{val}</div>
  <div class="metric-label">{label}</div>
</div>""", unsafe_allow_html=True)

            if GROQ_API_KEY and st.button("🤖 Get AI Travel Advice for This Weather", use_container_width=True):
                with st.spinner("Generating advice with Groq LLM…"):
                    advice = get_travel_response(
                        f"Give 3 quick, practical travel tips for visiting {w['city']} right now. "
                        f"Current weather: {w['temp']}°C, {w['description']}.",
                        [],
                    )
                st.info(f"**🤖 AI Travel Advice (powered by Groq LLM):**\n\n{advice}")
        else:
            st.markdown("""
<div class="card" style="text-align:center;padding:3rem;">
  <div style="font-size:4rem;">🌍</div>
  <div style="font-family:'Playfair Display',serif;font-size:1.2rem;margin-top:1rem;">Search a city to see live weather</div>
  <div style="color:#8899AA;font-size:.83rem;margin-top:.4rem;">Powered by OpenWeatherMap API · via requests library</div>
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — HISTORY
# ─────────────────────────────────────────────────────────────────────────────
with tab_history:
    st.markdown("### 📜 Search History & Saved Itineraries")
    ht1, ht2 = st.tabs(["🔍 Recent Searches", "🗺️ Saved Itineraries"])

    with ht1:
        history_rev = list(reversed(st.session_state.history))
        if history_rev:
            for item in history_rev[:20]:
                with st.expander(f"🔍 {item['query'][:65]} — {item['timestamp']}"):
                    st.markdown(item["response"])
        else:
            st.markdown("""
<div class="card" style="text-align:center;padding:2rem;">
  <div style="font-size:2.5rem;">📭</div>
  <div style="color:#8899AA;margin-top:.5rem;">No searches yet. Start exploring!</div>
</div>""", unsafe_allow_html=True)

    with ht2:
        saved_rev = list(reversed(st.session_state.saved_itineraries))
        if saved_rev:
            for idx, it in enumerate(saved_rev):
                with st.expander(f"🗺️ {it['destination']} — {it['days']} days — {it['created_at']}"):
                    st.markdown(it["content"])
                    if st.button("📋 Copy as Text", key=f"copy_{idx}"):
                        st.code(it["content"])
        else:
            st.markdown("""
<div class="card" style="text-align:center;padding:2rem;">
  <div style="font-size:2.5rem;">🗺️</div>
  <div style="color:#8899AA;margin-top:.5rem;">No saved itineraries yet. Generate one in the Itinerary tab!</div>
</div>""", unsafe_allow_html=True)
