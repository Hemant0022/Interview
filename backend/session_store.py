"""
Shared, disk-backed storage for session timeline + violation snapshots.

Why disk-backed instead of st.session_state:
Streamlit's multipage apps re-run the target page's script on every
navigation. st.session_state DOES persist across that within the same
browser session, but the Live Monitoring page's capture loop only
appends new data while that page is the one actively executing — if the
interviewer switches to the Session Report or Dashboard page, the Live
page's script (and its while-loop) stops running. Writing straight to
disk means every other page always sees exactly what's been recorded so
far, regardless of which page happens to be open, and it survives an
app restart too (needed for "snapshots persist until deleted").

Layout:
    session_data/
        timeline.csv       one row per logged timestamp
        violations.json    manifest: list of violation records
        snapshots/         one .jpg per violation, named "<id>.jpg"
"""

import os
import csv
import json
import time
import uuid
import cv2
import pandas as pd

DATA_DIR = "session_data"
SNAPSHOTS_DIR = os.path.join(DATA_DIR, "snapshots")
MANIFEST_PATH = os.path.join(DATA_DIR, "violations.json")
TIMELINE_PATH = os.path.join(DATA_DIR, "timeline.csv")

TIMELINE_COLUMNS = ["timestamp", "epoch", "status", "score_numeric", "fps"]

# How "Focused" / "Distracted" / "Unknown" map to a y-axis for the chart.
STATUS_NUMERIC = {"Focused": 1, "Distracted": 0, "Unknown": -1}


def ensure_dirs():
    os.makedirs(SNAPSHOTS_DIR, exist_ok=True)


# ------------------------------------------------------------------
# Timeline
# ------------------------------------------------------------------

def append_timeline(status, fps):
    """Append one timeline row. Called roughly once per second from the
    Live Monitoring loop."""
    ensure_dirs()
    is_new = not os.path.exists(TIMELINE_PATH)

    with open(TIMELINE_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(TIMELINE_COLUMNS)
        writer.writerow([
            time.strftime("%H:%M:%S"),
            time.time(),
            status,
            STATUS_NUMERIC.get(status, -1),
            round(fps, 2),
        ])


def load_timeline():
    """Returns a pandas DataFrame of the full timeline, or an empty one."""
    if not os.path.exists(TIMELINE_PATH):
        return pd.DataFrame(columns=TIMELINE_COLUMNS)
    try:
        return pd.read_csv(TIMELINE_PATH)
    except Exception:
        return pd.DataFrame(columns=TIMELINE_COLUMNS)


def clear_timeline():
    if os.path.exists(TIMELINE_PATH):
        os.remove(TIMELINE_PATH)


# ------------------------------------------------------------------
# Violations / snapshots
# ------------------------------------------------------------------

def _load_manifest():
    if not os.path.exists(MANIFEST_PATH):
        return []
    try:
        with open(MANIFEST_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return []


def _save_manifest(records):
    ensure_dirs()
    with open(MANIFEST_PATH, "w") as f:
        json.dump(records, f, indent=2)


def save_violation(frame_bgr, violation_type):
    """Encodes the frame as JPEG, saves it to disk, and records it in the
    manifest. Returns the new record."""
    ensure_dirs()

    vid = uuid.uuid4().hex[:12]
    filename = f"{vid}.jpg"
    path = os.path.join(SNAPSHOTS_DIR, filename)

    success, buf = cv2.imencode(".jpg", frame_bgr)
    if success:
        with open(path, "wb") as f:
            f.write(buf.tobytes())

    record = {
        "id": vid,
        "type": violation_type,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "epoch": time.time(),
        "filename": filename,
    }

    records = _load_manifest()
    records.append(record)
    _save_manifest(records)

    return record


def load_violations():
    """Returns violation records, most recent first."""
    records = _load_manifest()
    return sorted(records, key=lambda r: r.get("epoch", 0), reverse=True)


def load_violation_image_path(record):
    return os.path.join(SNAPSHOTS_DIR, record["filename"])


def delete_violation(violation_id):
    records = _load_manifest()
    keep = []
    for r in records:
        if r["id"] == violation_id:
            path = os.path.join(SNAPSHOTS_DIR, r["filename"])
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
        else:
            keep.append(r)
    _save_manifest(keep)


def clear_all_violations():
    for r in _load_manifest():
        path = os.path.join(SNAPSHOTS_DIR, r["filename"])
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
    _save_manifest([])


def violation_counts():
    records = _load_manifest()
    counts = {}
    for r in records:
        counts[r["type"]] = counts.get(r["type"], 0) + 1
    return counts


