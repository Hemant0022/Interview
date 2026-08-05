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
) -> Roadmap:
    """strengths / weak_areas typically come from resume_engine's
    strengths_and_weaknesses(); missing_jd_skills from skill_match;
    interview_weak_dimensions from the LLM evaluator's aggregate_scores
    (any dimension scoring < 70)."""
    global _last_llm_error
    _last_llm_error = None

    missing_jd_skills = missing_jd_skills or []
    interview_weak_dimensions = interview_weak_dimensions or []

    # Tag each topic with the source it came from, first-occurrence wins,
    # so the "why" text can reference the real reason instead of a
    # one-size-fits-all sentence. dict.fromkeys-style de-dup, preserving order.
    topic_source: Dict[str, str] = {}
    for t in weak_areas:
        topic_source.setdefault(t, "resume_weak")
    for t in missing_jd_skills:
        topic_source.setdefault(t, "missing_skill")
    for t in interview_weak_dimensions:
        topic_source.setdefault(t, "interview_dim")

    all_focus = list(topic_source.keys())
    if not all_focus:
        all_focus = ["projects", "communication"]
        topic_source = {"projects": "default", "communication": "default"}

    client = _get_client()
    if client is not None:
        plan = _plan_with_llm(client, strengths, all_focus)
        if plan is not None:
            return _attach_resources(plan, strengths, all_focus, method="llm",
                                      note=f"Personalized by {MODEL_NAME}.")
        # LLM was configured but the call/parse failed -- fall through to
        # heuristic, but say so explicitly instead of pretending no key
        # was ever configured.
        return _plan_heuristic(
            strengths, all_focus, topic_source,
            note=f"Gemini call failed, used rules-based plan instead. ({_last_llm_error or 'unknown error'})",
        )

    reason = "no GEMINI_API_KEY/GOOGLE_API_KEY configured" if not _GENAI_AVAILABLE or not (
        os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    ) else (_last_llm_error or "LLM client unavailable")
    return _plan_heuristic(strengths, all_focus, topic_source, note=f"Rules-based plan ({reason}).")


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


def _plan_with_llm(client, strengths: List[str], focus_areas: List[str]) -> Optional[List[Dict]]:
    import json
    import re
    global _last_llm_error

    prompt = f"""A candidate has these strengths: {strengths}
And these weak areas to improve before their next interview: {focus_areas}

Design a 4-week improvement plan. Distribute the weak areas across weeks
sensibly (foundational topics earlier, mock-interview/polish later). For
each week give: a short title, and a list of tasks, each with a "topic"
(must be one of the weak areas given -- do not invent new topics) and a
1-sentence "why" explaining why it matters for this candidate.

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
                     note: str = "") -> Roadmap:
    """Simple round-robin distribution across 4 weeks, front-loading
    foundational topics and ending with interview polish. Each task now
    gets a "why" that names the topic and its real source (resume /
    missing JD skill / weak interview dimension) instead of one shared
    generic sentence for every task."""
    chunks = [[] for _ in range(4)]
    for i, topic in enumerate(focus_areas):
        chunks[min(i % 3, 2)].append(topic)  # weeks 1-3 get topics
    chunks[3] = ["mock interview", "behavioral questions", "resume", "linkedin"]

    titles = ["Foundations", "Skill Building", "Applied Practice", "Interview & Profile Polish"]
    weeks = []
    for i, topics in enumerate(chunks):
        tasks = [
            {
                "topic": t,
                "why": _why_for(t, topic_source.get(t, "default")),
                "resources": _resources_for(t),
            }
            for t in topics
        ]
        weeks.append(RoadmapWeek(week=i + 1, title=titles[i], focus_areas=topics, tasks=tasks))

    return Roadmap(strengths=strengths, weak_areas=focus_areas, weeks=weeks, method="heuristic", note=note)