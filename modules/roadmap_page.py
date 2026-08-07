"""
roadmap_page.py
================
Renders the Candidate Improvement Roadmap. This is now driven entirely by
the AI Interview evaluation (backend.llm_evaluator), not by resume/ATS
weaknesses -- resume gaps are already covered on the Resume Analysis page,
so duplicating them here just produced a bare, unhelpful keyword list.

--------------------------------------------------------------------------
Fix notes
--------------------------------------------------------------------------
1. (Earlier fix) The roadmap was only rebuilt on button click, so
   navigating here for a different candidate without clicking
   "Regenerate" silently showed the previous candidate's stale plan.
   Fixed by hashing the inputs that drive the roadmap and auto-rebuilding
   whenever that signature changes.

2. (This fix) The "Weak Areas" panel was just a flat list of raw topic
   strings/keywords ("aws", "technical_accuracy", "grammar", ...) with no
   explanation of what was actually wrong or how to fix it -- not
   something a candidate could act on.

   Fixed, and refocused per product direction: the roadmap now ignores
   resume weaknesses / missing JD skills entirely (that analysis already
   lives on the Resume Analysis page) and is built solely from the
   interview evaluation:
     - Each weak interview *dimension* (technical accuracy, communication,
       confidence, grammar, completeness, relevance) is shown with the
       candidate's real score, the actual recurring "improvements"
       feedback the evaluator generated for the answers that scored low
       on that dimension (not a generic canned tip), and one concrete
       example question/answer where it showed up most clearly.
     - A new "Question-by-question feedback" section lists every
       interview answer with its full per-dimension scores, feedback,
       strengths and improvements -- the level of detail an actual human
       coach would walk a candidate through, not just a keyword.
     - The 4-week plan's "why" text for interview-related tasks now
       quotes that same real per-answer feedback instead of a canned
       "focused practice here should show up clearly" line for everyone.
"""
import hashlib
import json
import os

import streamlit as st

from backend import roadmap_engine, candidate_store

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

KIND_ICON = {"docs": "📄", "course": "🎓", "practice": "💻", "tutorial": "📘", "tool": "🛠️", "guide": "🧭"}


def _inputs_signature(strengths, interview_weak, evaluations) -> str:
    payload = json.dumps(
        {
            "strengths": sorted(strengths),
            "interview_weak": sorted(interview_weak),
            # Hash the actual answers too, not just which dimensions are
            # weak -- so redoing the interview with different answers
            # (even if the same dimensions end up weak) still rebuilds
            # the plan instead of silently reusing stale per-answer tips.
            "evaluations": [
                {"question": e.get("question", ""), "transcript": e.get("transcript", "")}
                for e in evaluations
            ],
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _interview_weak_area_details(weak_dims: dict, evaluations: list):
    """Builds one structured, explained card per weak interview dimension:
    the real score, the evaluator's own recurring feedback for that
    dimension (not a generic tip), and a concrete example answer -- so a
    candidate sees exactly what was wrong and what to do differently,
    not just a dimension name."""
    details = []
    for dim, score in weak_dims.items():
        label = dim.replace("_", " ").title()
        tips = roadmap_engine.improvements_for_dimension(dim, evaluations)
        example = roadmap_engine.worst_example_for_dimension(dim, evaluations)

        improvement = roadmap_engine.how_to_improve(label, "interview_dim", extra_tips=tips)

        details.append({
            "label": label,
            "score": score,
            "improvement": improvement,
            "has_real_feedback": bool(tips),
            "example": example,
            "dim_key": dim,
            "resource_topic": dim.replace("_", " "),
        })
    # Weakest first, so the candidate tackles the biggest gap first.
    return sorted(details, key=lambda d: d["score"])


def _build_and_store(strengths, interview_weak, evaluations, signature):
    roadmap = roadmap_engine.build_roadmap(
        strengths=strengths, weak_areas=[], missing_jd_skills=[],
        interview_weak_dimensions=interview_weak, interview_evaluations=evaluations,
    )
    st.session_state.roadmap = roadmap.to_dict()
    st.session_state.roadmap_inputs_signature = signature

    candidate_id = st.session_state.get("candidate_id")
    if candidate_id:
        candidate_store.upsert_candidate(candidate_id, "", {"roadmap": st.session_state.roadmap})


def render():
    st.title("⭐ Candidate Roadmap")
    st.caption("Your personalized improvement plan, built from how you actually performed in the AI Interview.")

    gemini_ready = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))

    resume_result = st.session_state.get("resume_result")
    interview_report = st.session_state.get("interview_report")

    if not interview_report:
        st.info(
            "Complete the **AI Mock Interview** first — this roadmap is built from how you actually "
            "answered, so it can give you specific, real feedback instead of generic tips. "
            "(Resume gaps are already covered on the **Resume Analysis** page.)"
        )
        return

    evaluations = interview_report.get("evaluations", [])
    agg_scores = interview_report.get("aggregate_scores", {})
    weak_dims = {dim: score for dim, score in agg_scores.items() if score < 70}
    interview_weak = list(weak_dims.keys())

    # Resume strengths are still nice context for the "what's working"
    # panel, but no resume weaknesses / missing skills feed the plan.
    strengths = []
    if resume_result:
        strengths = resume_result.get("strengths_weaknesses", {}).get("strengths", [])

    with st.expander("🔍 What we detected for this candidate", expanded=False):
        st.write("**Overall interview score:**", f"{interview_report.get('overall_score', 0)}%")
        st.write("**Per-dimension scores:**", agg_scores or "_none detected_")
        st.write("**Weak dimensions (below 70%):**", interview_weak or "_none — solid across the board_")
        st.write("**Questions evaluated:**", len(evaluations))
        if not interview_weak:
            st.caption(
                "No dimension scored below 70%, so the roadmap below defaults to general "
                "interview-polish tasks rather than inventing weaknesses that weren't there."
            )

    current_signature = _inputs_signature(strengths, interview_weak, evaluations)
    stored_signature = st.session_state.get("roadmap_inputs_signature")

    # Auto-regenerate whenever the underlying candidate data has changed,
    # so switching candidates (or redoing the interview) never silently
    # shows a stale plan.
    if st.session_state.get("roadmap") and current_signature != stored_signature:
        with st.spinner("Interview data changed — rebuilding your plan…"):
            _build_and_store(strengths, interview_weak, evaluations, current_signature)

    button_label = "Regenerate Roadmap" if st.session_state.get("roadmap") else "Generate Roadmap"
    if st.button(button_label, type="primary"):
        with st.spinner("Building your plan…"):
            _build_and_store(strengths, interview_weak, evaluations, current_signature)

    roadmap = st.session_state.get("roadmap")
    if not roadmap:
        return

    if gemini_ready and roadmap.get("method") == "heuristic":
        with st.spinner("Gemini key detected — rebuilding your learning roadmap…"):
            _build_and_store(strengths, interview_weak, evaluations, current_signature)
            roadmap = st.session_state.roadmap

    if roadmap.get("method") == "heuristic":
        st.caption(f"📋 {roadmap.get('note') or 'Using a rules-based plan instead of an LLM-personalized one.'}")
    else:
        st.caption(f"✨ {roadmap.get('note') or 'Personalized with an LLM.'}")

    # ---------------- Strengths ----------------
    st.markdown("**✅ What's working**")
    all_strengths = list(dict.fromkeys(strengths + [f"Strong {d.replace('_',' ')}" for d, s in agg_scores.items() if s >= 85]))
    if all_strengths:
        for s in all_strengths:
            st.success(s)
    else:
        st.caption("_None detected yet — everything below is worth focused practice._")

    # ---------------- Weak interview dimensions, explained ----------------
    st.markdown("**⚠️ What to improve, based on your interview**")
    st.caption("Your real score on each dimension, why it matters, and what to actually change — not just a label.")

    weak_details = _interview_weak_area_details(weak_dims, evaluations)
    if not weak_details:
        st.success("Every dimension scored 70% or above in your last interview — nice work. Use the weekly plan below to keep sharpening.")
    for d in weak_details:
        with st.container(border=True):
            st.markdown(f"**🎤 {d['label']} — {d['score']}%** _(target: 70%+)_")
            st.write(f"**How to improve:** {d['improvement']}")
            if not d["has_real_feedback"]:
                st.caption("No recurring pattern found across your answers yet — this is general guidance for the dimension.")
            if d["example"]:
                ex = d["example"]
                with st.expander(f"See the answer where this showed up most clearly (scored {ex['scores'].get(d['dim_key'], 0)}%)"):
                    st.markdown(f"**Q:** {ex.get('question', '')}")
                    if ex.get("transcript"):
                        st.caption(f"Your answer: “{ex['transcript'][:400]}{'…' if len(ex['transcript']) > 400 else ''}”")
                    if ex.get("feedback"):
                        st.write(ex["feedback"])
            resources = roadmap_engine.resources_for(d["resource_topic"])
            if resources:
                links = "  ·  ".join(
                    f"{KIND_ICON.get(r.get('kind'), '🔗')} [{r['title']}]({r['url']})" for r in resources
                )
                st.caption(links)

    st.divider()

    # ---------------- Question-by-question feedback ----------------
    st.markdown("### 📋 Question-by-question feedback")
    st.caption("Every answer from your interview, with the specific scores and coaching notes behind them.")
    if not evaluations:
        st.caption("_No answers recorded._")
    for i, e in enumerate(evaluations, start=1):
        overall = e.get("overall", 0)
        flag = "🔴" if overall < 60 else ("🟡" if overall < 75 else "🟢")
        with st.expander(f"{flag} Q{i}: {e.get('question', '')} — {overall}%"):
            scores = e.get("scores", {})
            cols = st.columns(len(scores) or 1)
            for col, (dim, val) in zip(cols, scores.items()):
                col.metric(dim.replace("_", " ").title(), f"{val}%")
            if e.get("transcript"):
                st.caption(f"Your answer: “{e['transcript'][:500]}{'…' if len(e['transcript']) > 500 else ''}”")
            if e.get("feedback"):
                st.write(f"**Feedback:** {e['feedback']}")
            if e.get("strengths"):
                st.write("**What worked:**")
                for s in e["strengths"]:
                    st.write(f"- {s}")
            if e.get("improvements"):
                st.write("**What to change:**")
                for imp in e["improvements"]:
                    st.write(f"- {imp}")

    st.divider()
    st.markdown("### AI Generated Learning Plan")

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
                    icon = KIND_ICON.get(res.get("kind"), "🔗")
                    st.markdown(f"{icon} [{res['title']}]({res['url']})")
                st.write("")

    st.divider()
    if st.button("Continue to Recruiter Dashboard →"):
        st.session_state.nav_target = "Dashboard"
        st.rerun()