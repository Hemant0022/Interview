"""
llm_evaluator.py
=================
LLM-based interview intelligence. Two jobs:

1. QUESTION GENERATION  -> pulls a small set of interview questions out of the
   resume/JD context (technical + behavioral), instead of a static question bank.
2. ANSWER EVALUATION    -> scores a candidate's spoken answer (transcript) on
   Technical Accuracy, Communication, Confidence, Grammar, Completeness and
   Relevance, matching the "AI Evaluation" panel in the product spec.

Uses the Google Gemini API (GEMINI_API_KEY env var). If no key is configured,
or the call fails for any reason, everything degrades to a transparent
heuristic scorer rather than crashing the app -- the UI should always show
`result["method"]` so it's clear whether a score is LLM-graded or
heuristic-graded.

Nothing here touches resume_engine.py / detect.py; this is purely additive.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

try:
    from google import genai
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False

# Override with GEMINI_MODEL env var if you want a different Gemini model.
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")

FILLER_WORDS = [
    "um", "uh", "like", "you know", "sort of", "kind of", "basically",
    "actually", "literally", "i mean", "so yeah", "right?",
]


def _get_client() -> Optional["genai.Client"]:
    if not _GENAI_AVAILABLE:
        return None
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception:
        return None


def _generate_text(client, prompt: str, max_output_tokens: int = 1000) -> str:
    """Calls Gemini and returns the plain text response."""
    resp = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    return resp.text or ""


def _extract_json(text: str) -> Optional[Dict]:
    """LLMs sometimes wrap JSON in prose/code fences. Pull the first
    top-level {...} or [...] block out and parse it."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# 1. QUESTION GENERATION
# ---------------------------------------------------------------------------

DEFAULT_QUESTION_BANK = [
    {"question": "Tell me about yourself and your background.", "type": "behavioral", "skill": "general"},
    {"question": "Explain a project you're most proud of and the impact it had.", "type": "behavioral", "skill": "general"},
    {"question": "What is the difference between supervised and unsupervised learning?", "type": "technical", "skill": "machine learning"},
    {"question": "Explain how you would optimize a slow SQL query.", "type": "technical", "skill": "sql"},
    {"question": "Describe a time you disagreed with a teammate. How did you resolve it?", "type": "behavioral", "skill": "communication"},
]


def generate_questions(resume_text: str, jd_text: str, num_questions: int = 5) -> List[Dict]:
    """Returns a list of {question, type, skill} dicts tailored to the
    candidate's resume and the job description. Falls back to a generic
    bank (still resume/JD-aware via simple skill lookup) if no LLM is
    configured."""
    client = _get_client()
    if client is None:
        return _fallback_questions(resume_text, jd_text, num_questions)

    prompt = f"""You are an experienced technical interviewer preparing questions for a candidate.

RESUME (excerpt):
{resume_text[:3000]}

JOB DESCRIPTION (excerpt):
{jd_text[:2000]}

Generate exactly {num_questions} interview questions: a mix of technical
questions grounded in the skills/JD overlap, and 1-2 behavioral questions.
Order them from easier to harder.

Respond with ONLY a JSON array, no prose, no markdown fences, in this shape:
[{{"question": "...", "type": "technical" | "behavioral", "skill": "the specific skill or topic being probed"}}]
"""
    try:
        text = _generate_text(client, prompt, max_output_tokens=1000)
        parsed = _extract_json(text)
        if isinstance(parsed, list) and parsed:
            return parsed[:num_questions]
    except Exception:
        pass

    return _fallback_questions(resume_text, jd_text, num_questions)


def _fallback_questions(resume_text: str, jd_text: str, num_questions: int) -> List[Dict]:
    try:
        from backend.resume_engine import find_skills
        jd_skills = find_skills(jd_text) if jd_text else []
    except Exception:
        jd_skills = []

    skill_qs = []
    templates = {
        "python": "Walk me through how you'd debug a memory leak in a Python service.",
        "sql": "Explain how you would optimize a slow SQL query.",
        "machine learning": "Explain the difference between bias and variance in a model.",
        "power bi": "How would you design a Power BI dashboard for a non-technical stakeholder?",
        "tensorflow": "Walk me through building and training a simple neural network in TensorFlow.",
        "communication": "Describe a time you had to explain a technical concept to a non-technical audience.",
    }
    for skill in jd_skills:
        if skill in templates:
            skill_qs.append({"question": templates[skill], "type": "technical", "skill": skill})

    combined = skill_qs + DEFAULT_QUESTION_BANK
    seen, deduped = set(), []
    for q in combined:
        if q["question"] not in seen:
            seen.add(q["question"])
            deduped.append(q)
    return deduped[:num_questions]


# ---------------------------------------------------------------------------
# 2. ANSWER EVALUATION
# ---------------------------------------------------------------------------

EVAL_DIMENSIONS = ["technical_accuracy", "communication", "confidence", "grammar", "completeness", "relevance"]


@dataclass
class AnswerEvaluation:
    question: str
    transcript: str
    scores: Dict[str, float] = field(default_factory=dict)
    overall: float = 0.0
    feedback: str = ""
    strengths: List[str] = field(default_factory=list)
    improvements: List[str] = field(default_factory=list)
    filler_word_count: int = 0
    method: str = "heuristic"  # "llm" or "heuristic"

    def to_dict(self) -> Dict:
        return asdict(self)


def count_filler_words(transcript: str) -> int:
    lowered = f" {transcript.lower()} "
    return sum(lowered.count(f" {w} ") for w in FILLER_WORDS)


def evaluate_answer(question: str, transcript: str, skill: str = "") -> AnswerEvaluation:
    """Scores one Q/A pair. transcript should be the raw speech-to-text
    output for that answer."""
    filler_count = count_filler_words(transcript)

    if not transcript.strip():
        return AnswerEvaluation(
            question=question, transcript=transcript,
            scores={d: 0.0 for d in EVAL_DIMENSIONS}, overall=0.0,
            feedback="No answer was captured for this question.",
            filler_word_count=0, method="heuristic",
        )

    client = _get_client()
    if client is not None:
        result = _evaluate_with_llm(client, question, transcript, skill, filler_count)
        if result is not None:
            return result

    return _evaluate_heuristic(question, transcript, filler_count)


def _evaluate_with_llm(client, question: str, transcript: str, skill: str, filler_count: int) -> Optional[AnswerEvaluation]:
    prompt = f"""You are grading one interview answer. Be fair but rigorous -- this
feeds a real hiring decision, so don't inflate scores.

QUESTION ({skill or 'general'}): {question}

CANDIDATE'S ANSWER (verbatim transcript, may include speech artifacts):
\"\"\"{transcript[:3000]}\"\"\"

Score each dimension 0-100:
- technical_accuracy: correctness/depth of technical content (if the question is behavioral, score based on relevance of the example instead)
- communication: clarity and structure of the answer
- confidence: how assured the phrasing is (hedging, filler words, rambling reduce this)
- grammar: grammatical correctness of the transcribed answer
- completeness: did they fully answer what was asked
- relevance: how on-topic the answer is

Respond with ONLY JSON, no prose, no markdown fences:
{{"scores": {{"technical_accuracy": 0, "communication": 0, "confidence": 0, "grammar": 0, "completeness": 0, "relevance": 0}},
"feedback": "2-3 sentence overall assessment",
"strengths": ["short bullet", "short bullet"],
"improvements": ["short bullet", "short bullet"]}}
"""
    try:
        text = _generate_text(client, prompt, max_output_tokens=700)
        parsed = _extract_json(text)
        if not parsed or "scores" not in parsed:
            return None

        scores = {d: float(parsed["scores"].get(d, 0)) for d in EVAL_DIMENSIONS}
        overall = round(sum(scores.values()) / len(scores), 1)

        return AnswerEvaluation(
            question=question, transcript=transcript, scores=scores, overall=overall,
            feedback=parsed.get("feedback", ""),
            strengths=parsed.get("strengths", []),
            improvements=parsed.get("improvements", []),
            filler_word_count=filler_count, method="llm",
        )
    except Exception:
        return None


def _evaluate_heuristic(question: str, transcript: str, filler_count: int) -> AnswerEvaluation:
    """Transparent, deterministic fallback used when no API key is set.
    Not a substitute for the LLM grader -- just keeps the UI functional."""
    words = transcript.split()
    word_count = len(words)

    completeness = min(100.0, round((word_count / 60) * 100, 1))  # ~60 words = "complete" answer
    grammar = max(40.0, 100.0 - 3 * transcript.count("  "))  # crude signal only
    confidence = max(20.0, 100.0 - filler_count * 8)
    communication = max(30.0, min(100.0, 100 - abs(word_count - 90) * 0.4))
    q_words = set(re.findall(r"[a-zA-Z]{4,}", question.lower()))
    a_words = set(re.findall(r"[a-zA-Z]{4,}", transcript.lower()))
    overlap = len(q_words & a_words) / max(1, len(q_words))
    relevance = round(40 + 60 * overlap, 1)
    technical_accuracy = relevance  # can't verify correctness without an LLM

    scores = {
        "technical_accuracy": round(technical_accuracy, 1),
        "communication": round(communication, 1),
        "confidence": round(confidence, 1),
        "grammar": round(grammar, 1),
        "completeness": completeness,
        "relevance": relevance,
    }
    overall = round(sum(scores.values()) / len(scores), 1)

    feedback = (
        "Heuristic score (no LLM configured) based on answer length, keyword overlap "
        "with the question, and filler-word count. Configure GEMINI_API_KEY for a "
        "real content-quality assessment."
    )
    strengths = ["Answered within a reasonable length"] if word_count > 20 else []
    improvements = []
    if filler_count > 3:
        improvements.append("Reduce filler words (um, like, you know)")
    if word_count < 30:
        improvements.append("Expand the answer with more detail or an example")

    return AnswerEvaluation(
        question=question, transcript=transcript, scores=scores, overall=overall,
        feedback=feedback, strengths=strengths, improvements=improvements,
        filler_word_count=filler_count, method="heuristic",
    )


def evaluate_full_interview(qa_pairs: List[Dict]) -> Dict:
    """qa_pairs: [{"question":..., "transcript":..., "skill":...}, ...]
    Returns an aggregate report used by the Interview Report / Recruiter views."""
    evaluations = [evaluate_answer(qa["question"], qa.get("transcript", ""), qa.get("skill", "")) for qa in qa_pairs]

    if not evaluations:
        return {"evaluations": [], "aggregate_scores": {d: 0.0 for d in EVAL_DIMENSIONS}, "overall_score": 0.0, "method": "heuristic"}

    aggregate = {
        d: round(sum(e.scores.get(d, 0) for e in evaluations) / len(evaluations), 1)
        for d in EVAL_DIMENSIONS
    }
    overall_score = round(sum(e.overall for e in evaluations) / len(evaluations), 1)
    method = "llm" if any(e.method == "llm" for e in evaluations) else "heuristic"

    return {
        "evaluations": [e.to_dict() for e in evaluations],
        "aggregate_scores": aggregate,
        "overall_score": overall_score,
        "total_filler_words": sum(e.filler_word_count for e in evaluations),
        "method": method,
    }