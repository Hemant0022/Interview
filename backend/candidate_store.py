"""
candidate_store.py
===================
Disk-backed store (same pattern as session_store.py) that ties a
candidate's resume analysis, interview evaluation, and behavior metrics
together under one record, so the Recruiter Dashboard has something to
rank/list without needing a real database.

Layout:
    candidate_data/
        candidates.json   list of candidate records, keyed by id
"""
import json
import os
import time
import uuid
from typing import Dict, List, Optional

DATA_DIR = "candidate_data"
STORE_PATH = os.path.join(DATA_DIR, "candidates.json")


def _ensure_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def _load_all() -> List[Dict]:
    _ensure_dir()
    if not os.path.exists(STORE_PATH):
        return []
    try:
        with open(STORE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []


def _save_all(records: List[Dict]):
    _ensure_dir()
    with open(STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, default=str)


def upsert_candidate(candidate_id: Optional[str], name: str, updates: Dict) -> str:
    """Creates a new record if candidate_id is None, otherwise merges
    `updates` into the existing record. Returns the candidate_id."""
    records = _load_all()

    if candidate_id is None:
        candidate_id = str(uuid.uuid4())[:8]
        record = {"id": candidate_id, "name": name, "created_at": time.time()}
        records.append(record)
    else:
        record = next((r for r in records if r["id"] == candidate_id), None)
        if record is None:
            record = {"id": candidate_id, "name": name, "created_at": time.time()}
            records.append(record)

    record.update(updates)
    record["updated_at"] = time.time()
    _save_all(records)
    return candidate_id


def get_candidate(candidate_id: str) -> Optional[Dict]:
    for r in _load_all():
        if r["id"] == candidate_id:
            return r
    return None


def list_candidates() -> List[Dict]:
    return sorted(_load_all(), key=lambda r: r.get("updated_at", 0), reverse=True)


def overall_score(record: Dict) -> float:
    resume_score = record.get("resume_score", 0.0) or 0.0
    interview_score = record.get("interview_score", 0.0) or 0.0
    if resume_score and interview_score:
        return round(0.5 * resume_score + 0.5 * interview_score, 1)
    return round(resume_score or interview_score, 1)


def recommendation_for(overall: float) -> str:
    if overall >= 80:
        return "Hire"
    if overall >= 65:
        return "Consider"
    if overall >= 45:
        return "Upskill"
    return "Not a fit"
