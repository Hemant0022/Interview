"""
app.py
======
AI Hiring Intelligence Platform — single entrypoint.

One script with sidebar tab navigation (not Streamlit's automatic
pages/ multipage mechanism) so we keep full control over layout and,
importantly, over the interview webcam's lifecycle when the user
switches tabs — same reasoning the old face.py used.

Run with:
    streamlit run app.py
"""
from dotenv import load_dotenv
import os

load_dotenv()

import streamlit as st

from modules import home_page, resume_page, interview_page, roadmap_page, recruiter_dashboard
try:
    hf_token = st.secrets.get("HF_TOKEN")
except Exception:
    hf_token = None

if hf_token:
    os.environ["HF_TOKEN"] = hf_token
st.set_page_config(
    page_title="AI Hiring Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Color theme — Blue primary / Green good / Orange warning / Red weak / White bg
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    #MainMenu, footer {visibility: hidden;}
    .block-container {padding-top: 1.6rem;}
    :root {
        --primary: #4F46E5;   /* blue */
        --good: #12B76A;      /* green */
        --warn: #F59E0B;      /* orange */
        --bad: #F04438;       /* red */
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        border-radius: 10px; padding: 6px 10px; margin-bottom: 2px;
    }
    div.stButton > button[kind="primary"] { background-color: var(--primary); border-color: var(--primary); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Navigation tree (matches the spec's sidebar)
# ---------------------------------------------------------------------------
NAV_TREE = {
    "🏠 Home": ["Home"],
    "👤 Candidate": ["Resume Analysis", "AI Mock Interview", "Interview Report", "Learning Roadmap"],
    "👨‍💼 Recruiter": ["Dashboard", "Reports"],
    "⚙️ Settings": ["Settings"],
}

# Several sidebar labels map onto the same underlying page, mirroring how
# the resume module already renders upload + analysis + job-match together,
# and the interview module renders both the live interview and its report.
PAGE_MAP = {
    "Home": home_page.render,
    "Resume Analysis": resume_page.render,
    "AI Mock Interview": interview_page.render,
    "Interview Report": interview_page.render,
    "Learning Roadmap": roadmap_page.render,
    "Dashboard": recruiter_dashboard.render,
    "Reports": recruiter_dashboard.render,
    "Settings": None,  # handled inline below
}

if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"

# Allow other pages to programmatically navigate (e.g. "Continue to AI Mock Interview →")
if "nav_target" in st.session_state:
    st.session_state.current_page = st.session_state.pop("nav_target")

with st.sidebar:
    st.markdown("## 🧠 AI Hiring Intelligence")
    st.divider()
    for group, items in NAV_TREE.items():
        st.markdown(f"**{group}**")
        for item in items:
            selected = st.session_state.current_page == item
            if st.button(item, key=f"nav_{item}", use_container_width=True,
                         type="primary" if selected else "secondary"):
                st.session_state.current_page = item
                st.rerun()
        st.write("")

# ---------------------------------------------------------------------------
# Camera lifecycle: leaving the AI Interview page always tears down the
# webcam thread, exactly like the old face.py's "leaving Live Monitoring"
# guard — prevents an orphaned camera thread running in the background.
# ---------------------------------------------------------------------------
if st.session_state.current_page != "AI Mock Interview" and st.session_state.get("camera_running"):
    interview_page.stop_session()

# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------
page = st.session_state.current_page

if page == "Settings":
    st.title("⚙️ Settings")
    st.write("**LLM Evaluation**")
    import os
    has_key = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    st.write(f"GEMINI_API_KEY configured: {'✅ Yes' if has_key else '❌ No — set this env var to enable LLM-graded interview scoring and personalized roadmaps.'}")
    st.write("**Candidate Data**")
    from backend import candidate_store
    st.write(f"Candidates on file: {len(candidate_store.list_candidates())}")
    if st.button("🗑️ Clear all candidate data", type="secondary"):
        import shutil
        shutil.rmtree(candidate_store.DATA_DIR, ignore_errors=True)
        st.success("Cleared.")
else:
    page_renderer = PAGE_MAP.get(page)
    if page_renderer is not None:
        page_renderer()
    else:
        st.title(page)
        st.info("This page is not wired up yet.")

