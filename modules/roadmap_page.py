"""
roadmap_page.py
================
Renders the Candidate Improvement Roadmap — pulls weak areas from the
resume analysis (backend.resume_engine) and the interview evaluation
(backend.llm_evaluator), then builds a 4-week plan via
backend.roadmap_engine with real clickable resources.

--------------------------------------------------------------------------
Fix notes (why the roadmap looked identical for every candidate)
--------------------------------------------------------------------------
The roadmap was only rebuilt when the "Generate Roadmap" button was
clicked. `st.session_state.roadmap` otherwise just kept whatever was
built for the *previous* candidate/resume/interview in this browser
session — so navigating here again for a different candidate without
re-clicking the button silently redisplayed the old plan.

Fixed: the inputs that actually drive the roadmap (strengths, weak areas,
missing JD skills, weak interview dimensions) are now hashed into a
signature. Any time that signature changes from what's stored in session
state, the roadmap is rebuilt automatically — the button becomes an
optional manual "regenerate anyway" action rather than the only trigger.

Also added a small diagnostic expander showing exactly what weak areas /
missing skills / interview dimensions were detected for this candidate,
so it's easy to tell at a glance whether upstream (resume_engine /
llm_evaluator) actually produced candidate-specific data, or whether the
roadmap is falling back to the generic ["projects", "communication"]
default because nothing was detected.
"""
import hashlib
import json
import os

import streamlit as st

from backend import roadmap_engine, candidate_store

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


def _inputs_signature(strengths, weak_areas, missing_skills, interview_weak) -> str:
    payload = json.dumps(
        {
            "strengths": sorted(strengths),
            "weak_areas": sorted(weak_areas),
            "missing_skills": sorted(missing_skills),
            "interview_weak": sorted(interview_weak),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _build_and_store(strengths, weak_areas, missing_skills, interview_weak, signature):
    roadmap = roadmap_engine.build_roadmap(
        strengths=strengths, weak_areas=weak_areas,
        missing_jd_skills=missing_skills, interview_weak_dimensions=interview_weak,
    )
    st.session_state.roadmap = roadmap.to_dict()
    st.session_state.roadmap_inputs_signature = signature

    candidate_id = st.session_state.get("candidate_id")
    if candidate_id:
        candidate_store.upsert_candidate(candidate_id, "", {"roadmap": st.session_state.roadmap})


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

    with st.expander("🔍 What we detected for this candidate", expanded=False):
        st.write("**Strengths:**", strengths or "_none detected_")
        st.write("**Resume weak areas:**", weak_areas or "_none detected_")
        st.write("**Missing JD skills:**", missing_skills or "_none detected_")
        st.write("**Weak interview dimensions:**", interview_weak or "_none detected_")
        if not (weak_areas or missing_skills or interview_weak):
            st.caption(
                "Nothing candidate-specific was detected, so the roadmap will use a generic "
                "'projects / communication' fallback. If this looks wrong, check that "
                "resume_engine's strengths_weaknesses()/skill_match() and the interview "
                "aggregate_scores are actually populated for this candidate."
            )

    current_signature = _inputs_signature(strengths, weak_areas, missing_skills, interview_weak)
    stored_signature = st.session_state.get("roadmap_inputs_signature")

    # Auto-regenerate whenever the underlying candidate data has changed,
    # so switching candidates never silently shows a stale roadmap.
    if st.session_state.get("roadmap") and current_signature != stored_signature:
        with st.spinner("Candidate data changed — rebuilding your plan…"):
            _build_and_store(strengths, weak_areas, missing_skills, interview_weak, current_signature)

    button_label = "Regenerate Roadmap" if st.session_state.get("roadmap") else "Generate Roadmap"
    if st.button(button_label, type="primary"):
        with st.spinner("Building your plan…"):
            _build_and_store(strengths, weak_areas, missing_skills, interview_weak, current_signature)

    roadmap = st.session_state.get("roadmap")
    if not roadmap:
        return

    if gemini_ready and roadmap.get("method") == "heuristic":
        with st.spinner("Gemini key detected — rebuilding your learning roadmap…"):
            _build_and_store(strengths, weak_areas, missing_skills, interview_weak, current_signature)
            roadmap = st.session_state.roadmap

    if roadmap.get("method") == "heuristic":
        st.caption(f"📋 {roadmap.get('note') or 'Using a rules-based plan instead of an LLM-personalized one.'}")
    else:
        st.caption(f"✨ {roadmap.get('note') or 'Personalized with an LLM.'}")

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
        st.session_state.nav_target = "Dashboard"
        st.rerun()