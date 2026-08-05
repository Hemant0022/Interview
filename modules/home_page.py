import streamlit as st
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

def render():
    st.markdown(
        """
        <div style="text-align:center; padding: 30px 0 10px;">
            <h1 style="margin-bottom:4px;">🧠 AI Hiring Intelligence</h1>
            <p style="color:#64748B; font-size:16px;">
                Resume screening, job matching, AI Mock interviews, and a recruiter dashboard — in one flow.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    features = [
        ("📄", "Resume Screening", "ATS-style scoring against a job description."),
        ("🎯", "Job Matching", "Skill, experience and education fit, explained."),
        ("🎥", "AI Mock Interview", "Webcam + voice interview with live LLM evaluation."),
        ("📊", "Recruiter Dashboard", "Rank candidates and export PDF reports."),
    ]
    for col, (icon, title, desc) in zip(cols, features):
        with col:
            st.markdown(
                f"""
                <div style="background:#F8F9FC; border:1px solid #E7E9F3; border-radius:16px;
                            padding:20px; text-align:center; height:170px;">
                    <div style="font-size:28px;">{icon}</div>
                    <div style="font-weight:700; margin-top:8px;">{title}</div>
                    <div style="font-size:12.5px; color:#64748B; margin-top:6px;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")
    st.write("")
    c = st.columns([1, 1, 1])
    with c[1]:
        if st.button("🚀 Start Assessment", type="primary", use_container_width=True):
            st.session_state.nav_target = "Resume Analysis"
            st.rerun()
