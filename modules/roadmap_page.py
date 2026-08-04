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
        agg_scores = interview_report.get("aggregate_scores", {})
        interview_weak = [dim for dim, score in agg_scores.items() if score < 70]
        # Previously only weak dimensions were ever extracted from the
        # interview, so a candidate with no resume analysis (interview-only
        # flow) always saw an empty "Strengths" column even though
        # aggregate_scores clearly has dimensions scoring >= 70.
        interview_strengths = [dim for dim, score in agg_scores.items() if score >= 70]
        strengths = strengths + [s for s in interview_strengths if s not in strengths]

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

    total_weeks = len(roadmap["weeks"])
    total_tasks = sum(len(w["tasks"]) for w in roadmap["weeks"])
    method_label = "AI-Personalized (Gemini)" if roadmap.get("method") == "llm" else "Rules-Based"

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Strengths Identified", len(roadmap["strengths"]))
    m2.metric("Focus Areas", len(roadmap["weak_areas"]))
    m3.metric("Total Tasks", total_tasks)
    # m4.metric("Plan Type", method_label)

    st.divider()

    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown("#### ✅ Strengths")
        if roadmap["strengths"]:
            for s in roadmap["strengths"]:
                st.success(str(s).strip().title())
        else:
            st.caption("_No strengths captured yet._")
    with sc2:
        st.markdown("#### ⚠️ Focus Areas")
        if roadmap["weak_areas"]:
            for w in roadmap["weak_areas"]:
                st.warning(str(w).strip().title())
        else:
            st.caption("_No focus areas identified._")

    st.divider()
    st.markdown("### 📅 4-Week Learning Plan")

    kind_icon = {"docs": "📄", "course": "🎓", "practice": "💻", "tutorial": "📘", "tool": "🛠️", "guide": "🧭"}

    for week in roadmap["weeks"]:
        task_count = len(week["tasks"])
        subtitle = f"{task_count} task{'s' if task_count != 1 else ''}" if task_count else "No tasks — nice work"
        with st.expander(f"**Week {week['week']} — {week['title']}**  ·  {subtitle}", expanded=(week["week"] == 1)):
            if not week["tasks"]:
                st.write("_Nothing to work on this week. Keep up the momentum!_")
                continue
            for idx, task in enumerate(week["tasks"], start=1):
                with st.container(border=True):
                    st.markdown(f"**{idx}. {task['topic'].strip().title()}**")
                    if task.get("why"):
                        st.caption(task["why"])
                    resources = task.get("resources", [])
                    if resources:
                        res_cols = st.columns(len(resources)) if len(resources) <= 3 else [st] * len(resources)
                        for col, res in zip(res_cols, resources):
                            icon = kind_icon.get(res.get("kind"), "🔗")
                            col.markdown(f"{icon} [{res['title']}]({res['url']})")

    st.divider()
    if st.button("Continue to Recruiter Dashboard →", type="primary"):
        st.session_state.nav_target = "Recruiter Dashboard"
        st.rerun()