"""
roadmap_page.py
================
Renders the Candidate Improvement Roadmap — pulls weak areas from the
resume analysis (backend.resume_engine) and the interview evaluation
(backend.llm_evaluator), then builds a 4-week plan via
backend.roadmap_engine with real clickable resources.
"""
import streamlit as st
import os

from backend import roadmap_engine, candidate_store

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

def render():
    st.title("⭐ Candidate Roadmap")
    st.caption("Your personalized 4-week plan, built from your resume and interview results.")

    gemini_ready = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))

    resume_result = st.session_state.get("resume_result")
    interview_report = st.session_state.get("interview_report")

    if not resume_result and not interview_report:
        st.info("Complete **Resume Analysis** and/or the **AI Interview** first so the roadmap can be tailored to you.")
        return

    strengths, weak_areas, missing_skills, interview_weak = [], [], [], []

    if resume_result:
        sw = resume_result.get("strengths_weaknesses", {})
        strengths = sw.get("strengths", [])
        weak_areas = sw.get("weaknesses", [])
        missing_skills = resume_result.get("skill_match", {}).get("missing", [])

    if interview_report:
        interview_weak = [
            dim for dim, score in interview_report.get("aggregate_scores", {}).items() if score < 70
        ]

    if st.button("Generate Roadmap", type="primary"):
        with st.spinner("Building your plan…"):
            roadmap = roadmap_engine.build_roadmap(
                strengths=strengths, weak_areas=weak_areas,
                missing_jd_skills=missing_skills, interview_weak_dimensions=interview_weak,
            )
        st.session_state.roadmap = roadmap.to_dict()

        candidate_id = st.session_state.get("candidate_id")
        if candidate_id:
            candidate_store.upsert_candidate(candidate_id, "", {"roadmap": st.session_state.roadmap})

    roadmap = st.session_state.get("roadmap")
    if not roadmap:
        return

    if gemini_ready and roadmap.get("method") == "heuristic":
        with st.spinner("Gemini key detected — rebuilding your learning roadmap…"):
            roadmap_obj = roadmap_engine.build_roadmap(
                strengths=strengths, weak_areas=weak_areas,
                missing_jd_skills=missing_skills, interview_weak_dimensions=interview_weak,
            )
            roadmap = roadmap_obj.to_dict()
            st.session_state.roadmap = roadmap

            candidate_id = st.session_state.get("candidate_id")
            if candidate_id:
                candidate_store.upsert_candidate(candidate_id, "", {"roadmap": st.session_state.roadmap})

    if roadmap.get("method") == "heuristic":
        st.caption(
            "No GEMINI_API_KEY configured — using a rules-based plan instead of an LLM-personalized one."
    )

    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown("**✅ Strengths**")
        for s in roadmap["strengths"]:
            st.success(s)
    with sc2:
        st.markdown("**⚠️ Weak Areas**")
        for w in roadmap["weak_areas"]:
            st.warning(w)

    st.divider()
    st.markdown("### AI Generated Learning Plan")

    kind_icon = {"docs": "📄", "course": "🎓", "practice": "💻", "tutorial": "📘", "tool": "🛠️", "guide": "🧭"}

    for week in roadmap["weeks"]:
        with st.expander(f"Week {week['week']} — {week['title']}", expanded=(week["week"] == 1)):
            if not week["tasks"]:
                st.write("_No tasks this week — nice work._")
                continue
            for task in week["tasks"]:
                st.markdown(f"**{task['topic'].title()}**")
                if task.get("why"):
                    st.caption(task["why"])
                for res in task.get("resources", []):
                    icon = kind_icon.get(res.get("kind"), "🔗")
                    st.markdown(f"{icon} [{res['title']}]({res['url']})")
                st.write("")

    st.divider()
    if st.button("Continue to Recruiter Dashboard →"):
        st.session_state.nav_target = "Recruiter Dashboard"
        st.rerun()