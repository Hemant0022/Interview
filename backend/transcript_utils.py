import os
from pathlib import Path



def append_transcript(existing_text: str, new_text: str) -> str:
    """Append a new transcription segment to a running transcript without duplication."""
    cleaned_new = (new_text or "").strip()
    if not cleaned_new:
        return existing_text.strip()

    cleaned_existing = (existing_text or "").strip()
    if not cleaned_existing:
        return cleaned_new

    normalized_existing = cleaned_existing.lower()
    normalized_new = cleaned_new.lower()

    if normalized_new in normalized_existing:
        return cleaned_existing

    if normalized_existing.endswith(normalized_new):
        return cleaned_existing

    return f"{cleaned_existing} {cleaned_new}".strip()
