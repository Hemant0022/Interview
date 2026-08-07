"""
roadmap_engine.py
==================
Builds the "Candidate Improvement Roadmap" -- the flagship feature in the
spec. Combines:
  - resume_engine's strengths/weaknesses + missing JD skills
  - the interview evaluator's weak scoring dimensions
into a 4-week plan, each item linked to a real, curated resource
(official docs / a well-known course) so it's actually actionable instead
of a bare list of topic names.

Falls back to a rules-based plan (no LLM) if no API key is set; the LLM
path only reorders/prioritizes and writes the "why" text -- resource links
always come from the curated RESOURCE_LIBRARY below, never invented by the
model, so links are never hallucinated.

--------------------------------------------------------------------------
Fix notes (why every candidate was getting the "same" roadmap)
--------------------------------------------------------------------------
1. The heuristic (no-LLM) path previously gave every single task the exact
   same hardcoded "why" string, regardless of topic or candidate:
       "Identified as a weak area to strengthen before your next interview."
   Since the LLM path only ever runs when GEMINI_API_KEY/GOOGLE_API_KEY is
   set *and* the call succeeds, most deployments spend all their time on
   this heuristic path -- so "why" text looked identical for everyone.
   Fixed: each topic is now tagged with where it came from (a resume
   weakness / a missing JD skill / a weak interview dimension) and gets a
   template that names the topic and explains *that* reason specifically.

2. Failures inside `_get_client()` / `_plan_with_llm()` were caught with a
   bare `except Exception: return None` and silently discarded -- so a bad
   API key, network error, or bad model name looked identical to "no key
   configured at all", with no way to tell them apart. Fixed: the last
   error is now captured in `_last_llm_error` and surfaced on the
   returned Roadmap's `.note` field so the UI can show *why* it fell back.
"""


from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
try:
    from google import genai
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False


MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")

# ---------------------------------------------------------------------------
# Curated resource library: topic -> list of {title, url, kind}
# Kept small and hand-picked (official docs / widely-used free courses)
# rather than trying to cover everything.
# ---------------------------------------------------------------------------
RESOURCE_LIBRARY: Dict[str, List[Dict]] = {
    "python": [
        {"title": "Official Python Tutorial", "url": "https://docs.python.org/3/tutorial/", "kind": "docs"},
        {"title": "Python Interview Practice — LeetCode", "url": "https://leetcode.com/", "kind": "practice"},
    ],
    "sql": [
        {"title": "SQL Tutorial — Mode Analytics", "url": "https://mode.com/sql-tutorial/", "kind": "tutorial"},
        {"title": "SQL Practice — HackerRank", "url": "https://www.hackerrank.com/domains/sql", "kind": "practice"},
    ],
    "machine learning": [
        {"title": "Machine Learning Crash Course — Google", "url": "https://developers.google.com/machine-learning/crash-course", "kind": "course"},
        {"title": "scikit-learn User Guide", "url": "https://scikit-learn.org/stable/user_guide.html", "kind": "docs"},
    ],
    "power bi": [
        {"title": "Power BI Official Documentation", "url": "https://learn.microsoft.com/en-us/power-bi/", "kind": "docs"},
        {"title": "Power BI Guided Learning — Microsoft", "url": "https://learn.microsoft.com/en-us/power-bi/guided-learning/", "kind": "course"},
    ],
    "tableau": [
        {"title": "Tableau Training & Tutorials", "url": "https://www.tableau.com/learn/training", "kind": "course"},
    ],
    "tensorflow": [
        {"title": "TensorFlow Official Tutorials", "url": "https://www.tensorflow.org/tutorials", "kind": "docs"},
    ],
    "nlp": [
        {"title": "Hugging Face NLP Course", "url": "https://huggingface.co/learn/nlp-course", "kind": "course"},
    ],
    "statistics": [
        {"title": "Statistics and Probability — Khan Academy", "url": "https://www.khanacademy.org/math/statistics-probability", "kind": "course"},
    ],
    "communication": [
        {"title": "Toastmasters — Improve Public Speaking", "url": "https://www.toastmasters.org/", "kind": "practice"},
    ],
    "eye contact": [
        {"title": "Mock Interview Practice — Big Interview", "url": "https://www.biginterview.com/", "kind": "practice"},
    ],
    "projects": [
        {"title": "Build in Public — GitHub Explore", "url": "https://github.com/explore", "kind": "practice"},
    ],
    "resume": [
        {"title": "Resume Worded — free resume scan", "url": "https://resumeworded.com/", "kind": "tool"},
    ],
    "github": [
        {"title": "GitHub Docs — Building a strong profile", "url": "https://docs.github.com/en/account-and-profile", "kind": "docs"},
    ],
    "linkedin": [
        {"title": "LinkedIn Learning — Profile optimization", "url": "https://www.linkedin.com/learning/", "kind": "course"},
    ],
    "coding": [
        {"title": "LeetCode — Coding Interview Practice", "url": "https://leetcode.com/", "kind": "practice"},
    ],
    "behavioral questions": [
        {"title": "STAR Method Guide — Indeed Career Guide", "url": "https://www.indeed.com/career-advice/interviewing/how-to-use-the-star-interview-response-technique", "kind": "guide"},
    ],
    "aws": [
        {"title": "AWS Training and Certification", "url": "https://skillbuilder.aws/", "kind": "course"},
    ],
    "gcp": [
        {"title": "Google Cloud Training and Certification", "url": "https://cloud.google.com/training", "kind": "course"},
    ],
}

GENERIC_RESOURCE = {"title": "Search Coursera for this topic", "url": "https://www.coursera.org/search", "kind": "course"}


def _resources_for(topic: str) -> List[Dict]:
    key = topic.lower().strip()
    if key in RESOURCE_LIBRARY:
        return RESOURCE_LIBRARY[key]
    for k, v in RESOURCE_LIBRARY.items():
        if k in key or key in k:
            return v
    return [GENERIC_RESOURCE]


# Public alias -- pages outside this module (e.g. roadmap_page.py) should
# use this instead of reaching into the private _resources_for().
def resources_for(topic: str) -> List[Dict]:
    return _resources_for(topic)


# ---------------------------------------------------------------------------
# Personalized "why" text for the heuristic (no-LLM) path.
#
# Each focus-area topic is tagged with the source it came from -- a resume
# weakness, a JD skill missing from the resume, or a low-scoring interview
# dimension -- and gets a template that mentions the topic by name and
# explains *that specific* reason, instead of one identical sentence for
# every task regardless of candidate or topic.
# ---------------------------------------------------------------------------
WHY_TEMPLATES = {
    "missing_skill": "\"{topic}\" is listed in the job description but doesn't show up in your resume — recruiters will likely screen for it directly.",
    "resume_weak": "Your resume review flagged \"{topic}\" as a weaker area — sharpening this will make your application noticeably stronger.",
    "interview_dim": "Your interview scored below target on \"{topic}\" — focused practice here should show up clearly in your next round.",
    "default": "Strengthening \"{topic}\" will directly improve how you come across to recruiters and interviewers.",
}

# Fixed week-4 "polish" topics. These are the same category of task for
# every candidate by design (interview polish + profile hygiene), but each
# one now gets its own distinct reasoning instead of sharing one sentence.
POLISH_WHY = {
    "mock interview": "A timed mock interview surfaces pacing and delivery issues that are hard to notice on your own before the real thing.",
    "behavioral questions": "Behavioral questions come up in almost every loop — having STAR-structured answers ready removes a common source of rambling.",
    "resume": "A final resume pass catches formatting/keyword issues that can cause an otherwise-strong resume to get filtered out automatically.",
    "linkedin": "Recruiters and interviewers routinely check LinkedIn before/after a conversation — keeping it aligned with your resume avoids inconsistencies.",
}


def _why_for(topic: str, source: str) -> str:
    if topic in POLISH_WHY:
        return POLISH_WHY[topic]
    template = WHY_TEMPLATES.get(source, WHY_TEMPLATES["default"])
    return template.format(topic=topic)


# Public alias -- pages outside this module should use this rather than
# reaching into the private _why_for().
def why_for_topic(topic: str, source: str) -> str:
    return _why_for(topic, source)


# ---------------------------------------------------------------------------
# "How to improve" text -- distinct from WHY_TEMPLATES above. WHY explains
# *why* a topic was flagged; this explains the concrete next action to take,
# so the Weak Areas view can show both instead of a bare topic keyword.
# ---------------------------------------------------------------------------
HOW_TEMPLATES = {
    "missing_skill": "Get hands-on with {topic} — even a small side project counts — and add it explicitly to your Skills or Experience section so it isn't missed by a recruiter or ATS scan.",
    "resume_weak": "Rework this section of your resume before you apply again — be specific, quantify impact where you can, and make sure it's clearly labeled so an ATS parses it correctly.",
    "interview_dim": "Run a focused mock interview on {topic}, review the recording afterward, and note one concrete change to try in the next round.",
    "default": "Work through the resource below, then practice explaining {topic} out loud before your next interview.",
}

# Dimension-specific improvement tips for interview scoring dimensions --
# more actionable than the generic interview_dim template above.
INTERVIEW_DIM_TIPS = {
    "technical_accuracy": "Revisit the core concepts behind the questions you missed and practice explaining them out loud, not just re-reading notes silently.",
    "communication": "Practice structuring answers with a clear beginning, middle, and end (e.g. the STAR method) before your next mock interview.",
    "confidence": "Record yourself answering a few practice questions and review your tone and pacing — most confidence issues come from delivery, not content.",
    "grammar": "Read your answers back out loud, or run written practice answers through a grammar checker, to catch recurring phrasing issues.",
    "completeness": "Make sure you directly answer every part of the question asked — practice pausing to check you haven't skipped a sub-question.",
    "relevance": "Tie your examples more explicitly back to the question asked, and trim any background detail that isn't doing work for your answer.",
}


def how_to_improve(topic: str, source: str, extra_tips: Optional[List[str]] = None) -> str:
    if source == "interview_dim" and topic.lower().replace(" ", "_") in INTERVIEW_DIM_TIPS:
        base = INTERVIEW_DIM_TIPS[topic.lower().replace(" ", "_")]
    elif topic in POLISH_WHY:
        # Week-4 polish topics already have their own actionable framing.
        return POLISH_WHY[topic]
    else:
        template = HOW_TEMPLATES.get(source, HOW_TEMPLATES["default"])
        base = template.format(topic=topic)
    if extra_tips:
        return base + " Specific to your actual answers: " + " ".join(f"({i+1}) {t}" for i, t in enumerate(extra_tips))
    return base


def improvements_for_dimension(dim_key: str, evaluations: Optional[List[Dict]], top_n: int = 3) -> List[str]:
    """Pulls the most common 'improvements' bullets the answer-evaluator
    already generated for questions where this specific dimension scored
    below target (<70), so the roadmap can surface real, per-answer
    coaching instead of a generic canned tip. dim_key uses the underscore
    form used in aggregate_scores, e.g. 'technical_accuracy'."""
    if not evaluations:
        return []
    from collections import Counter
    bullets = []
    for e in evaluations:
        if e.get("scores", {}).get(dim_key, 100) < 70:
            bullets.extend(b.strip() for b in e.get("improvements", []) if b and b.strip())
    if not bullets:
        return []
    counts = Counter(bullets)
    return [b for b, _ in counts.most_common(top_n)]


def worst_example_for_dimension(dim_key: str, evaluations: Optional[List[Dict]]) -> Optional[Dict]:
    """Returns the single evaluation (question/answer/feedback) where this
    dimension scored the lowest, so the roadmap can point to a concrete
    example instead of speaking only in the abstract."""
    candidates = [e for e in (evaluations or []) if dim_key in e.get("scores", {})]
    if not candidates:
        return None
    return min(candidates, key=lambda e: e["scores"].get(dim_key, 100))


@dataclass
class RoadmapWeek:
    week: int
    title: str
    focus_areas: List[str] = field(default_factory=list)
    tasks: List[Dict] = field(default_factory=list)  # {topic, why, resources:[...]}


@dataclass
class Roadmap:
    strengths: List[str]
    weak_areas: List[str]
    weeks: List[RoadmapWeek]
    method: str = "heuristic"
    note: str = ""  # human-readable explanation of *why* this method was used

    def to_dict(self) -> Dict:
        return {
            "strengths": self.strengths,
            "weak_areas": self.weak_areas,
            "weeks": [asdict(w) for w in self.weeks],
            "method": self.method,
            "note": self.note,
        }


_last_llm_error: Optional[str] = None


def build_roadmap(
    strengths: List[str],
    weak_areas: List[str],
    missing_jd_skills: Optional[List[str]] = None,
    interview_weak_dimensions: Optional[List[str]] = None,
    interview_evaluations: Optional[List[Dict]] = None,
) -> Roadmap:
    """strengths / weak_areas typically come from resume_engine's
    strengths_and_weaknesses(); missing_jd_skills from skill_match;
    interview_weak_dimensions from the LLM evaluator's aggregate_scores
    (any dimension scoring < 70), as underscore-form keys (e.g.
    'technical_accuracy'). interview_evaluations, if provided, is the
    per-question list from evaluate_full_interview()['evaluations'] --
    passing it lets the plan quote the evaluator's own real, per-answer
    'improvements' feedback instead of a generic canned tip for that
    dimension."""
    global _last_llm_error
    _last_llm_error = None

    missing_jd_skills = missing_jd_skills or []
    interview_weak_dimensions = interview_weak_dimensions or []

    # Interview dimensions arrive as underscore keys ("technical_accuracy").
    # Use a readable label as the topic everywhere it's displayed, but keep
    # the reverse mapping so we can still look up real per-answer feedback
    # for that dimension in interview_evaluations.
    dim_label_to_key = {d.replace("_", " "): d for d in interview_weak_dimensions}
    interview_weak_labels = list(dim_label_to_key.keys())

    interview_tips: Dict[str, List[str]] = {}
    if interview_evaluations:
        for label, key in dim_label_to_key.items():
            tips = improvements_for_dimension(key, interview_evaluations)
            if tips:
                interview_tips[label] = tips

    # Tag each topic with the source it came from, first-occurrence wins,
    # so the "why" text can reference the real reason instead of a
    # one-size-fits-all sentence. dict.fromkeys-style de-dup, preserving order.
    topic_source: Dict[str, str] = {}
    for t in weak_areas:
        topic_source.setdefault(t, "resume_weak")
    for t in missing_jd_skills:
        topic_source.setdefault(t, "missing_skill")
    for t in interview_weak_labels:
        topic_source.setdefault(t, "interview_dim")

    all_focus = list(topic_source.keys())
    if not all_focus:
        all_focus = ["projects", "communication"]
        topic_source = {"projects": "default", "communication": "default"}

    client = _get_client()
    if client is not None:
        plan = _plan_with_llm(client, strengths, all_focus, interview_tips)
        if plan is not None:
            return _attach_resources(plan, strengths, all_focus, method="llm",
                                      note=f"Personalized by {MODEL_NAME}.")
        # LLM was configured but the call/parse failed -- fall through to
        # heuristic, but say so explicitly instead of pretending no key
        # was ever configured.
        return _plan_heuristic(
            strengths, all_focus, topic_source, interview_tips,
            note=f"Gemini call failed, used rules-based plan instead. ({_last_llm_error or 'unknown error'})",
        )

    reason = "no GEMINI_API_KEY/GOOGLE_API_KEY configured" if not _GENAI_AVAILABLE or not (
        os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    ) else (_last_llm_error or "LLM client unavailable")
    return _plan_heuristic(strengths, all_focus, topic_source, interview_tips, note=f"Rules-based plan ({reason}).")


def _get_client():
    global _last_llm_error
    if not _GENAI_AVAILABLE:
        _last_llm_error = "google-genai package not installed"
        return None
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        _last_llm_error = f"client init failed: {e}"
        return None


def _plan_with_llm(client, strengths: List[str], focus_areas: List[str],
                    interview_tips: Optional[Dict[str, List[str]]] = None) -> Optional[List[Dict]]:
    import json
    import re
    global _last_llm_error

    tips_block = ""
    if interview_tips:
        lines = [f"- {topic}: {'; '.join(tips)}" for topic, tips in interview_tips.items()]
        tips_block = (
            "\n\nReal, specific issues pulled from the candidate's actual interview answers "
            "(use these to make the \"why\" text concrete, not generic):\n" + "\n".join(lines)
        )

    prompt = f"""A candidate has these strengths: {strengths}
And these weak areas to improve before their next interview: {focus_areas}{tips_block}

Design a 4-week improvement plan. Distribute the weak areas across weeks
sensibly (foundational topics earlier, mock-interview/polish later). For
each week give: a short title, and a list of tasks, each with a "topic"
(must be one of the weak areas given -- do not invent new topics) and a
1-2 sentence "why" explaining why it matters for this candidate -- if real
issues were given above for a topic, reference them specifically instead
of writing something generic.

Respond with ONLY JSON, no prose, no markdown fences:
{{"weeks": [{{"week": 1, "title": "...", "tasks": [{{"topic": "...", "why": "..."}}]}}]}}
"""
    try:
        resp = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        text = resp.text or ""
        text = re.sub(r"^```(?:json)?|```$", "", text.strip()).strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            _last_llm_error = "model response did not contain JSON"
            return None
        parsed = json.loads(match.group(0))
        weeks = parsed.get("weeks")
        if not weeks:
            _last_llm_error = "model response JSON had no 'weeks' key"
            return None
        return weeks
    except Exception as e:
        _last_llm_error = f"{type(e).__name__}: {e}"
        return None


def _attach_resources(weeks_raw: List[Dict], strengths: List[str], focus_areas: List[str],
                       method: str, note: str = "") -> Roadmap:
    weeks = []
    for w in weeks_raw:
        tasks = []
        for t in w.get("tasks", []):
            topic = t.get("topic", "")
            tasks.append({"topic": topic, "why": t.get("why", ""), "resources": _resources_for(topic)})
        weeks.append(RoadmapWeek(week=w.get("week", len(weeks) + 1), title=w.get("title", f"Week {len(weeks)+1}"),
                                  focus_areas=[t["topic"] for t in tasks], tasks=tasks))
    return Roadmap(strengths=strengths, weak_areas=focus_areas, weeks=weeks, method=method, note=note)


def _plan_heuristic(strengths: List[str], focus_areas: List[str], topic_source: Dict[str, str],
                     interview_tips: Optional[Dict[str, List[str]]] = None, note: str = "") -> Roadmap:
    """Simple round-robin distribution across 4 weeks, front-loading
    foundational topics and ending with interview polish. Each task now
    gets a "why" that names the topic and its real source (resume /
    missing JD skill / weak interview dimension) instead of one shared
    generic sentence for every task. For interview-dimension topics, when
    interview_tips has real per-answer feedback for that dimension, it's
    appended so the reasoning is concrete rather than canned."""
    chunks = [[] for _ in range(4)]
    for i, topic in enumerate(focus_areas):
        chunks[min(i % 3, 2)].append(topic)  # weeks 1-3 get topics
    chunks[3] = ["mock interview", "behavioral questions", "resume", "linkedin"]

    titles = ["Foundations", "Skill Building", "Applied Practice", "Interview & Profile Polish"]
    weeks = []
    for i, topics in enumerate(chunks):
        tasks = []
        for t in topics:
            source = topic_source.get(t, "default")
            why = _why_for(t, source)
            tips = (interview_tips or {}).get(t) if source == "interview_dim" else None
            if tips:
                why = why + " From your actual answers: " + " ".join(f"({j+1}) {tip}" for j, tip in enumerate(tips))
            tasks.append({"topic": t, "why": why, "resources": _resources_for(t)})
        weeks.append(RoadmapWeek(week=i + 1, title=titles[i], focus_areas=topics, tasks=tasks))

    return Roadmap(strengths=strengths, weak_areas=focus_areas, weeks=weeks, method="heuristic", note=note)