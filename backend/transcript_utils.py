

import os
import re
from pathlib import Path


_WORD_RE = re.compile(r"\S+")

# Must stay in sync with STT_CHUNK_OVERLAP_SECONDS in interview_page.py —
# just an upper bound on how many trailing words a ~0.4s audio overlap
# could plausibly produce, so the search below stays cheap and doesn't
# accidentally match a real, intentional repeated word much further back.
MAX_OVERLAP_WORDS = 8


def append_transcript(existing_text: str, new_text: str) -> str:
    """Append a new transcription segment to a running transcript without duplication.

    interview_page.py's TranscriptionWorker now feeds chunks with a
    small deliberate audio overlap between consecutive chunks (so a word
    cut off at a flush boundary gets a second, whole-word chance in the
    next chunk instead of being split in half). That means each new
    segment's transcribed text will typically start by re-saying the
    last word or two of the previous segment — this function trims that
    overlap at the WORD level (comparing the tail of what we already
    have against the head of the new segment) rather than requiring an
    exact full-string match, since the two decodes of the same audio
    won't always come out byte-identical.

    We deliberately only look at the boundary (last few words vs. first
    few words), never anywhere else in the transcript — an earlier
    version used `normalized_new in normalized_existing`, which silently
    discarded any new word/phrase that happened to be a substring of
    anything said earlier (e.g. a fresh "ready" segment dropped because
    "already" was said earlier). That ate the large majority of real
    chunks and is exactly what this function must avoid.
    """
    cleaned_new = (new_text or "").strip()
    if not cleaned_new:
        return existing_text.strip()

    cleaned_existing = (existing_text or "").strip()
    if not cleaned_existing:
        return cleaned_new

    existing_words = _WORD_RE.findall(cleaned_existing)
    new_words = _WORD_RE.findall(cleaned_new)
    existing_lower = [w.lower() for w in existing_words]
    new_lower = [w.lower() for w in new_words]

    # Find the longest run where the tail of the existing transcript
    # equals the head of the new segment, largest overlap first so we
    # don't under-trim when several words legitimately repeat.
    max_k = min(MAX_OVERLAP_WORDS, len(existing_lower), len(new_lower))
    overlap_k = 0
    for k in range(max_k, 0, -1):
        if existing_lower[-k:] == new_lower[:k]:
            overlap_k = k
            break

    remaining_words = new_words[overlap_k:]
    if not remaining_words:
        # The entire new segment was just the overlap re-heard — nothing
        # new was actually said.
        return cleaned_existing

    return f"{cleaned_existing} {' '.join(remaining_words)}".strip()