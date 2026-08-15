

"""
interview_page.py
==================
Interview module — TWO camera sessions (recruiter + candidate), each
captured from that participant's own browser over WebRTC, with live
per-speaker transcription merged into one shared, timestamped feed.

What changed from the candidate-only version
----------------------------------------------
1. Two roles, two cameras.
   The page now has a "role" concept: "recruiter" or "candidate". Each
   role is a SEPARATE browser tab/session (the recruiter and the
   candidate each open this page on their own device), and each captures
   ONLY its own camera+mic — exactly like the original single-camera
   version did, just once per role instead of once total. A Streamlit
   script run is tied to one browser session, so there is no way for one
   tab to directly read another tab's raw webcam frames; instead each
   side publishes its own (lightly-throttled, JPEG-encoded) frame into a
   small in-process shared store, and reads the other side's frame back
   out of that same store. See `_SHARED_SESSIONS` below.

2. A shared session code is what ties the two tabs together.
   The candidate's "Start Interview" step generates a short code and
   shows it on screen ("share this with your recruiter"); the recruiter
   enters that code to join. Both sides then read/write the same shared
   session dict, keyed by that code.

3. Live transcription is still captured PER MIC (a browser can only ever
   transcribe its own audio), but every finished chunk — from either
   side — is now also appended to the SHARED, time-ordered transcript,
   tagged with who said it. Both tabs render the exact same merged feed,
   styled to match the requested layout: `[Time][Name]: Question: ...`
   for the recruiter and `[Time][Name]: Answer: ...` for the candidate.
   The candidate's own running transcript (used for scoring on the
   report page) is unchanged from before — it is still just the
   candidate's own speech, so `llm_evaluator` needs no changes.

4. The face-detection bounding box (and all attention/behavior/object
   analysis) is now only ever run on the CANDIDATE's frame. The
   recruiter's own video is shown as a plain, unannotated feed — no
   `process_frame()` call, no verification/attention panels — since
   there is nothing to monitor on the recruiter's side.

Important limitation to be aware of
------------------------------------
`_SHARED_SESSIONS` below is a plain in-process dict guarded by a lock.
That's enough for the common case of one Streamlit server process
serving both the recruiter's and the candidate's tab (the same
deployment model the original file assumed for everything else). If
this app is ever run behind multiple worker processes/replicas without
sticky routing to the same process, the two roles could land on
different processes and never see each other's frames/transcript. In
that case, swap `_SHARED_SESSIONS` for a real shared store (e.g. Redis,
or a table in whatever DB `session_store`/`candidate_store` already use)
— the read/write call sites are all funneled through the small
`_get_or_create_shared_session` / `_publish_frame` / `_publish_segment`
helpers below, so that swap only touches this one section.

Everything else — the WebRTC video/audio plumbing, the SenseVoice
transcription worker, the ambient-noise calibration, the violation
cooldown/counting, and the report page's scoring — is carried over from
the candidate-only version unchanged.
"""
import base64
import queue
import os
import secrets
import time
import threading

import av
import cv2
import numpy as np
import streamlit as st
import torch
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
try:
    from funasr import AutoModel as _FunASRAutoModel
    from funasr.utils.postprocess_utils import rich_transcription_postprocess as _sensevoice_postprocess
except Exception:
    _FunASRAutoModel = None
    _sensevoice_postprocess = None

from backend.detect import process_frame, reset_calibration
from backend import session_store, candidate_store, llm_evaluator
from backend.transcript_utils import append_transcript

# Without an ICE server, WebRTC negotiation tends to only succeed on
# localhost/loopback — the moment there's any real network hop between a
# participant's browser and the server, the peer connection can fail
# silently. A public STUN server fixes this for the vast majority of
# networks; a particularly strict corporate firewall/NAT may eventually
# need a TURN server instead.
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})


# ---------------------------------------------------------------------------
# Shared cross-session store — see the module docstring's "Important
# limitation" note above before relying on this beyond a single process.
# ---------------------------------------------------------------------------
_SHARED_LOCK = threading.Lock()
_SHARED_SESSIONS = {}  # session_code -> shared session dict


def _new_session_code() -> str:
    """Short, human-typeable code (e.g. 'A1B2C3') the candidate shares
    with the recruiter so both tabs find the same shared session."""
    return secrets.token_hex(3).upper()


def _get_or_create_shared_session(code: str) -> dict:
    with _SHARED_LOCK:
        if code not in _SHARED_SESSIONS:
            _SHARED_SESSIONS[code] = {
                "created_at": time.time(),
                "session_start_time": None,   # set once, by the candidate, on Start
                "candidate_name": None,
                "recruiter_name": None,
                "connected": {"candidate": False, "recruiter": False},
                "frames": {"candidate": None, "recruiter": None},  # latest JPEG bytes per role
                "transcript_segments": [],    # [{"t","role","name","text"}], time-ordered
                "ended": False,
            }
        return _SHARED_SESSIONS[code]


def _publish_frame(shared: dict, role: str, frame_bgr: np.ndarray, quality: int = 60):
    """Encodes a frame to JPEG and stores it for the OTHER role's tab to
    read back. Cheap enough to call every few frames (not every frame —
    see FRAME_SHARE_EVERY_N_FRAMES in the render loop)."""
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return
    with _SHARED_LOCK:
        shared["frames"][role] = buf.tobytes()
        shared["connected"][role] = True


def _read_other_frame(shared: dict, other_role: str):
    with _SHARED_LOCK:
        return shared["frames"][other_role]


def _publish_segment(shared: dict, role: str, name: str, elapsed: float, text: str):
    with _SHARED_LOCK:
        shared["transcript_segments"].append({
            "t": elapsed,
            "role": role,
            "name": name,
            "text": text,
        })


def _read_shared_segments(shared: dict):
    with _SHARED_LOCK:
        return sorted(list(shared["transcript_segments"]), key=lambda s: s["t"])


def stop_session(end_shared: bool = False):
    """Tears down THIS tab's transcription worker. The WebRTC connection
    itself is owned by the webrtc_streamer component tied to this script
    run; it stops automatically once we stop calling it.

    `end_shared`: only the candidate ending the interview should mark the
    shared session as finished — a recruiter simply leaving must not cut
    the candidate's still-running session off from itself."""
    st.session_state.camera_running = False

    worker = st.session_state.get("transcription_worker")
    if worker is not None:
        worker.stop()
        st.session_state.transcription_worker = None

    if end_shared:
        code = st.session_state.get("session_code")
        if code:
            with _SHARED_LOCK:
                shared = _SHARED_SESSIONS.get(code)
                if shared is not None:
                    shared["ended"] = True


# ---------------------------------------------------------------------------
# Live speech-to-text — SenseVoice only (unchanged from the candidate-only
# version; each tab gets its own TranscriptionWorker transcribing only
# its own mic).
# ---------------------------------------------------------------------------
STT_TARGET_SR = 16000
STT_TRANSCRIBE_INTERVAL_SECONDS = 0.0
STT_MIN_CHUNK_SECONDS = float(os.getenv("STT_MIN_CHUNK_SECONDS", "0.5"))

STT_SILENCE_RMS_THRESHOLD = float(os.getenv("STT_SILENCE_RMS_THRESHOLD", "0.006"))
STT_SILENCE_RMS_THRESHOLD_MAX = float(os.getenv("STT_SILENCE_RMS_THRESHOLD_MAX", "0.02"))
STT_AMBIENT_CALIBRATION_SECONDS = float(os.getenv("STT_AMBIENT_CALIBRATION_SECONDS", "1.5"))
STT_AMBIENT_NOISE_MULTIPLIER = float(os.getenv("STT_AMBIENT_NOISE_MULTIPLIER", "3.0"))

STT_CHUNK_OVERLAP_SECONDS = float(os.getenv("STT_CHUNK_OVERLAP_SECONDS", str(round(STT_MIN_CHUNK_SECONDS * 0.3, 2))))
STT_PAUSE_TRIGGER_SECONDS = float(os.getenv("STT_PAUSE_TRIGGER_SECONDS", "0.35"))
STT_MAX_CHUNK_SECONDS = float(os.getenv("STT_MAX_CHUNK_SECONDS", "4.0"))

SENSEVOICE_MODEL_NAME = os.getenv("SENSEVOICE_MODEL_NAME", "iic/SenseVoiceSmall")
SENSEVOICE_LANGUAGE = os.getenv("SENSEVOICE_LANGUAGE", "en")

_sensevoice_singleton_lock = threading.Lock()
_sensevoice_singleton = {"backend": None}


def _load_stt_backend_once():
    """Loads (once per server process) and caches the SenseVoice backend.
    Not an `st.cache_resource` function on purpose: it's called from a
    background thread (TranscriptionWorker), and st.cache_resource is
    meant to be called from the main Streamlit thread."""
    with _sensevoice_singleton_lock:
        if _sensevoice_singleton["backend"] is not None:
            return _sensevoice_singleton["backend"]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if _FunASRAutoModel is None:
            raise RuntimeError("SenseVoice backend unavailable: funasr is not installed.")

        sv_device = "cuda:0" if device == "cuda" else "cpu"
        model = _FunASRAutoModel(
            model=SENSEVOICE_MODEL_NAME,
            trust_remote_code=True,
            device=sv_device,
            disable_update=True,
            disable_pbar=True,
        )
        _sensevoice_singleton["backend"] = {"kind": "sensevoice", "model": model}
        return _sensevoice_singleton["backend"]


def _make_resampler() -> av.AudioResampler:
    return av.AudioResampler(format="s16", layout="mono", rate=STT_TARGET_SR)


def _audio_frame_to_float32(audio_frame: av.AudioFrame) -> np.ndarray:
    samples = audio_frame.to_ndarray().astype(np.float32).flatten()
    return samples / 32768.0


def _format_elapsed(seconds) -> str:
    """Formats interview-elapsed seconds as MM:SS (or H:MM:SS past an hour)."""
    if seconds is None or seconds < 0:
        seconds = 0
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _transcribe_chunk(audio_array: np.ndarray, backend) -> str:
    if audio_array.size == 0:
        return ""
    result = backend["model"].generate(
        input=audio_array,
        cache={},
        language=SENSEVOICE_LANGUAGE,
        use_itn=True,
        batch_size_s=60,
    )
    if not result:
        return ""
    raw_text = result[0].get("text", "")
    text = _sensevoice_postprocess(raw_text) if _sensevoice_postprocess else raw_text
    return text.strip()


class TranscriptionWorker:
    """Runs entirely on its own background thread — INCLUDING loading the
    SenseVoice model itself — so the main render loop never blocks on it.
    Each browser tab (recruiter or candidate) gets its own instance,
    transcribing only that tab's own mic audio."""

    def __init__(self):
        self.lock = threading.Lock()
        self.backend = None
        self.ready = False
        self.load_failed = False
        self.load_error = None
        self.transcribe_error = None
        self.pending_chunk = None
        self.pending_meta = None
        self.latest_text = None
        self.latest_meta = None
        self.busy = False
        self.stopped = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def is_ready(self) -> bool:
        with self.lock:
            return self.ready

    def get_load_error(self):
        with self.lock:
            return self.load_error

    def get_transcribe_error(self):
        with self.lock:
            return self.transcribe_error

    def submit_chunk(self, chunk: np.ndarray, meta=None):
        """Fire-and-forget. Dropped (not queued) if the worker is still
        busy or the model isn't loaded yet, so lag can never build up."""
        with self.lock:
            if self.ready and not self.busy:
                self.pending_chunk = chunk
                self.pending_meta = meta
                self.busy = True

    def _run(self):
        try:
            self.backend = _load_stt_backend_once()
            with self.lock:
                self.ready = True
        except Exception as e:
            with self.lock:
                self.load_failed = True
                self.load_error = f"{type(e).__name__}: {e}"
            return

        while not self.stopped:
            chunk = None
            meta = None
            with self.lock:
                if self.pending_chunk is not None:
                    chunk = self.pending_chunk
                    meta = self.pending_meta
                    self.pending_chunk = None
                    self.pending_meta = None
            if chunk is None:
                time.sleep(0.05)
                continue
            try:
                text = _transcribe_chunk(chunk, self.backend)
                with self.lock:
                    self.transcribe_error = None
            except Exception as e:
                text = ""
                with self.lock:
                    self.transcribe_error = f"{type(e).__name__}: {e}"
            with self.lock:
                if text:
                    self.latest_text = text
                    self.latest_meta = meta
                self.busy = False

    def get_latest_text(self):
        with self.lock:
            text = self.latest_text
            meta = self.latest_meta
            self.latest_text = None
            self.latest_meta = None
            return text, meta

    def stop(self):
        self.stopped = True


# ---------------------------------------------------------------------------
# Violation snapshots — unchanged. These only ever run against the
# CANDIDATE's frame now (see _render_interview), so the existing Session
# Report tab keeps working unmodified.
# ---------------------------------------------------------------------------
VIOLATION_COOLDOWN_SECONDS = 15

DISTRACTION_PARAMETERS = [
    ("Looking Left",          "behavior", "looking_left"),
    ("Looking Right",         "behavior", "looking_right"),
    ("Multiple Faces",        "behavior", "multiple_faces"),
    ("Face Covered/Missing",  "behavior", "face_missing"),
    ("Phone Detected",        "objects",  "phone_detected"),
    ("Object On Face",        "objects",  "object_on_face"),
    ("Object On Eyes",        "objects",  "object_on_eyes"),
]


def _maybe_log_violation(frame_bgr, violation_type, is_active):
    if not is_active:
        return
    last = st.session_state._last_violation_time.get(violation_type, 0)
    if time.time() - last >= VIOLATION_COOLDOWN_SECONDS:
        try:
            session_store.save_violation(frame_bgr, violation_type)
        except Exception:
            pass
        st.session_state._last_violation_time[violation_type] = time.time()
        st.session_state.distraction_counts[violation_type] = \
            st.session_state.distraction_counts.get(violation_type, 0) + 1


# ---------------------------------------------------------------------------
# Session-state bootstrap
# ---------------------------------------------------------------------------
def _init_state():
    defaults = {
        "camera_running": False,
        "transcription_worker": None,
        "qa_records": [],
        "interview_stage": "setup",  # setup -> in_progress -> report
        "attention_samples": [],
        "eye_contact_samples": [],
        "live_transcript": "",        # THIS tab's own speech only (used for scoring)
        "transcript_segments": [],    # THIS tab's own [{"t","text"}] segments
        "session_start_time": None,
        "_last_violation_time": {},
        "distraction_counts": {},
        "media_session_id": 0,
        # --- new: role / shared-session identity ---
        "role": None,                 # "candidate" or "recruiter"
        "session_code": None,         # shared code both tabs use to find each other
        "participant_name": "",
        "last_rendered_seg_count": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def render():
    _init_state()
    st.title("🎥 Mock Interview")

    if st.session_state.interview_stage == "setup":
        _render_setup()
    elif st.session_state.interview_stage == "in_progress":
        _render_interview()
    else:
        _render_report()


# ---------------------------------------------------------------------------
# Stage 1 — setup: pick a role, then either start (candidate, generates a
# code) or join (recruiter, enters a code).
# ---------------------------------------------------------------------------
def _render_setup():
    st.caption(
        "You'll be asked to allow camera & microphone access in your browser. Recruiter and "
        "candidate each join from their own device. The candidate's face, attention, and "
        "behavior will be tracked throughout, and both sides' speech is transcribed live."
    )

    try:
        query_role = st.query_params.get("role", "")
        query_code = st.query_params.get("session", "")
    except Exception:
        query_role, query_code = "", ""

    role_choice = st.radio(
        "I am joining as the:",
        ["Candidate", "Recruiter"],
        index=(1 if str(query_role).lower() == "recruiter" else 0),
        horizontal=True,
    )
    role_key = "candidate" if role_choice == "Candidate" else "recruiter"
    name = st.text_input(f"Your name ({role_choice})", value=st.session_state.participant_name)

    # Recruiter creates the session (generates/shares the code) and starts
    # it; candidate joins with the code the recruiter shares with them.
    if role_key == "recruiter":
        if "pending_session_code" not in st.session_state:
            st.session_state.pending_session_code = _new_session_code()
        st.info(
            f"Session code: **{st.session_state.pending_session_code}** — share this with your "
            "candidate so they can join from their own device."
        )
        if st.button("▶️ Start Interview", type="primary"):
            code = st.session_state.pending_session_code
            shared = _get_or_create_shared_session(code)
            shared["recruiter_name"] = name or "Recruiter"
            shared["session_start_time"] = time.time()

            # detect.py's calibration state lives in a MODULE-LEVEL dict,
            # not st.session_state — it survives across reruns AND across
            # different candidates in the same server process. Must be
            # reset here, each time a NEW interview starts, so the
            # candidate joining this session gets their own ~3s
            # calibration window instead of inheriting someone else's.
            reset_calibration()

            st.session_state.qa_records = []
            st.session_state.role = "recruiter"
            st.session_state.session_code = code
            st.session_state.participant_name = name or "Recruiter"
            st.session_state.interview_stage = "in_progress"
            st.session_state.camera_running = True
            st.session_state.live_transcript = ""
            st.session_state.transcript_segments = []
            st.session_state.session_start_time = shared["session_start_time"]
            st.session_state.distraction_counts = {}
            st.session_state._last_violation_time = {}
            st.session_state.last_rendered_seg_count = 0
            st.session_state.media_session_id += 1
            st.session_state.pop("pending_session_code", None)
            st.rerun()
    else:
        code_input = st.text_input("Session code (from the recruiter)", value=query_code or "")
        if st.button("🔗 Join Interview", type="primary"):
            code_norm = code_input.strip().upper()
            with _SHARED_LOCK:
                exists = code_norm in _SHARED_SESSIONS
            if not code_norm:
                st.error("Enter the session code the recruiter shared with you.")
            elif not exists:
                st.error("That session code wasn't found. Double-check it with the recruiter.")
            else:
                shared = _get_or_create_shared_session(code_norm)
                shared["candidate_name"] = name or "Candidate"

                st.session_state.role = "candidate"
                st.session_state.session_code = code_norm
                st.session_state.participant_name = name or "Candidate"
                st.session_state.interview_stage = "in_progress"
                st.session_state.camera_running = True
                st.session_state.live_transcript = ""
                st.session_state.transcript_segments = []
                st.session_state.session_start_time = shared["session_start_time"]
                st.session_state.last_rendered_seg_count = 0
                st.session_state.media_session_id += 1
                st.rerun()


# ---------------------------------------------------------------------------
# Stage 2 — in progress: single script run per tab, internal render loop
# (no per-frame st.rerun()). Each tab captures its OWN camera+mic over ONE
# combined WebRTC connection. Video/audio from the OTHER role arrives via
# the shared store published by that role's own tab.
# ---------------------------------------------------------------------------
FRAME_SHARE_EVERY_N_FRAMES = 3   # how often THIS tab publishes its frame for the other side to see
OTHER_FRAME_REFRESH_EVERY_N_FRAMES = 3  # how often THIS tab redraws the other side's frame


def _render_interview():
    role = st.session_state.role
    other_role = "recruiter" if role == "candidate" else "candidate"
    code = st.session_state.session_code
    shared = _get_or_create_shared_session(code)

    top_l, top_r = st.columns([3, 1])
    with top_l:
        st.caption(f"You're connected as **{st.session_state.participant_name}** ({role}). Session code: `{code}`")
    with top_r:
        if role == "candidate":
            end_clicked = st.button("⏹️ Leave Session", type="primary", use_container_width=True)
        else:
            end_clicked = False
            if st.button("🚪 End Interview", use_container_width=True):
                stop_session(end_shared=False)
                st.session_state.interview_stage = "setup"
                st.session_state.role = None
                st.session_state.session_code = None
                st.rerun()

    if end_clicked:
        transcript = st.session_state.live_transcript.strip()
        st.session_state.qa_records = []
        if transcript:
            st.session_state.qa_records.append({
                "question": "Full interview response",
                "skill": "general",
                "transcript": transcript,
                "duration": 0.0,
                "segments": list(st.session_state.transcript_segments),
            })
        # Snapshot the full shared (recruiter+candidate) dialogue onto
        # THIS tab's own session_state so the report page can render it
        # even after the shared session is torn down.
        st.session_state.shared_transcript_snapshot = _read_shared_segments(shared)
        st.session_state.interview_stage = "report"
        stop_session(end_shared=True)
        st.rerun()

    # ---- Two camera panels, side by side, matching the requested layout ----
    cam1_col, cam2_col = st.columns(2)
    with cam1_col:
        st.markdown("**Camera 1**")
        recruiter_frame_box = st.empty()
        recruiter_label_box = st.empty()
    with cam2_col:
        st.markdown("**Camera 2**")
        candidate_frame_box = st.empty()
        candidate_label_box = st.empty()

    own_frame_box = recruiter_frame_box if role == "recruiter" else candidate_frame_box
    own_label_box = recruiter_label_box if role == "recruiter" else candidate_label_box
    other_frame_box = candidate_frame_box if role == "recruiter" else recruiter_frame_box
    other_label_box = candidate_label_box if role == "recruiter" else recruiter_label_box

    def _display_name(r):
        stored = shared.get("recruiter_name" if r == "recruiter" else "candidate_name")
        if stored:
            return stored
        return f"[{'Recruiter' if r == 'recruiter' else 'Candidate'} Name]"

    own_frame_box.info("Waiting for camera connection — click **Start** below.")
    own_label_box.caption(_display_name(role))
    other_frame_box.info("Waiting for the other participant to connect…")
    other_label_box.caption(_display_name(other_role))

    st.divider()
    st.subheader("Camera & Microphone")
    st.caption("Click **Start** once to enable your camera and mic.")
    media_ctx = webrtc_streamer(
        key=f"interview-media-{role}-{st.session_state.media_session_id}",
        mode=WebRtcMode.SENDONLY,
        rtc_configuration=RTC_CONFIGURATION,
        video_receiver_size=128,
        audio_receiver_size=512,
        media_stream_constraints={
            "video": {"width": 640, "height": 480},
            "audio": True,
        },
    )
    mic_level_box = st.empty()
    mic_level_box.info("🎙️ Waiting for microphone connection — click **Start** above.")

    # Verification/attention panels only make sense for the candidate —
    # there is nothing being monitored on the recruiter's side.
    verification_box = None
    monitoring_box = None
    if role == "candidate":
        st.divider()
        v_col, m_col = st.columns(2)
        with v_col:
            st.subheader("✅ System Verification")
            verification_box = st.empty()
        with m_col:
            st.subheader("📡 Attention, Behavior & Objects")
            monitoring_box = st.empty()

    st.divider()
    st.markdown("### Live Transcription")
    transcript_status = st.empty()
    transcribe_error_box = st.empty()
    transcript_box = st.empty()
    _render_shared_transcript(transcript_box, shared)

    if st.session_state.transcription_worker is None:
        st.session_state.transcription_worker = TranscriptionWorker()
    worker = st.session_state.transcription_worker

    PROCESS_EVERY_N_FRAMES = 2
    VIDEO_UPDATE_EVERY_N_FRAMES = 2
    PANEL_UPDATE_EVERY_N_FRAMES = 5

    frame_count = 0
    last_analysis = None
    last_worker_ready_shown = None
    last_transcribe_error_shown = None
    shown_waiting_msg = True
    shown_waiting_mic_msg = True

    resampler = _make_resampler()
    sound_buffer = np.array([], dtype=np.float32)
    last_audio_flush = time.time()
    last_iter_time = time.time()
    silence_run_seconds = 0.0

    effective_silence_threshold = STT_SILENCE_RMS_THRESHOLD
    ambient_calibration_samples = []
    ambient_calibration_done = False

    last_mic_level_update = 0.0
    latest_mic_rms = 0.0
    MIC_LEVEL_UPDATE_SECONDS = 0.2
    MIC_LEVEL_NORMALIZATION = 0.05

    last_seg_count_seen = len(shared["transcript_segments"])
    session_start = shared.get("session_start_time") or time.time()

    try:
        while st.session_state.camera_running:
            # ---- Speech-model status ----
            worker_ready = worker.is_ready()
            worker_load_error = worker.get_load_error()
            if worker_load_error is not None:
                if last_worker_ready_shown != "failed":
                    transcript_status.error(
                        f"❌ Speech model failed to load: {worker_load_error}\n\n"
                        "Camera and detection above are unaffected — this only blocks transcription."
                    )
                    last_worker_ready_shown = "failed"
            elif worker_ready != last_worker_ready_shown:
                if worker_ready:
                    transcript_status.empty()
                else:
                    transcript_status.caption(
                        "🔄 Loading speech model in the background — camera is already live; "
                        "transcription will switch on automatically in a few seconds."
                    )
                last_worker_ready_shown = worker_ready

            worker_transcribe_error = worker.get_transcribe_error()
            if worker_transcribe_error != last_transcribe_error_shown:
                if worker_transcribe_error is not None:
                    transcribe_error_box.warning(f"⚠️ Last transcription attempt failed: {worker_transcribe_error}")
                else:
                    transcribe_error_box.empty()
                last_transcribe_error_shown = worker_transcribe_error

            # ---- Video: drain everything queued and keep only the
            # newest frame (avoids a growing backlog of stale frames). ----
            got_video = False
            if media_ctx.video_receiver is not None:
                latest_vframe = None
                while True:
                    try:
                        vframe = media_ctx.video_receiver.get_frame(timeout=0)
                    except queue.Empty:
                        break
                    latest_vframe = vframe

                if latest_vframe is not None:
                    got_video = True
                    shown_waiting_msg = False
                    frame = latest_vframe.to_ndarray(format="bgr24")
                    frame_count += 1

                    if role == "candidate":
                        # Only the candidate's frame ever gets the
                        # face-detection bounding box / analysis overlay.
                        if frame_count % PROCESS_EVERY_N_FRAMES == 0 or last_analysis is None:
                            frame, analysis = process_frame(frame)
                            last_analysis = analysis

                            _maybe_log_violation(frame, "Phone Detected", analysis["objects"]["phone_detected"])
                            _maybe_log_violation(frame, "Face Covered/Missing", analysis["behavior"]["face_missing"])
                            _maybe_log_violation(frame, "Object On Face", analysis["objects"]["object_on_face"])
                            _maybe_log_violation(frame, "Multiple Faces", analysis["behavior"]["multiple_faces"])
                            _maybe_log_violation(frame, "Looking Left", analysis["behavior"]["looking_left"])
                            _maybe_log_violation(frame, "Looking Right", analysis["behavior"]["looking_right"])
                            _maybe_log_violation(frame, "Object On Eyes", analysis["objects"]["object_on_eyes"])

                            st.session_state.attention_samples.append(analysis["attention"].get("score", 0))
                            st.session_state.eye_contact_samples.append(
                                1 if analysis["attention"].get("looking_at_screen") else 0
                            )
                            try:
                                session_store.append_timeline(analysis["attention"]["status"], analysis["system"].get("fps", 0))
                            except Exception:
                                pass
                        else:
                            analysis = last_analysis
                    else:
                        # Recruiter: plain, unannotated feed — no bbox,
                        # no detection, nothing monitored.
                        analysis = None

                    if frame_count % VIDEO_UPDATE_EVERY_N_FRAMES == 0:
                        own_frame_box.image(frame, channels="BGR", use_container_width=True)

                    if frame_count % FRAME_SHARE_EVERY_N_FRAMES == 0:
                        _publish_frame(shared, role, frame)

                    if role == "candidate" and frame_count % PANEL_UPDATE_EVERY_N_FRAMES == 0 and analysis is not None:
                        face_ok = analysis["face"]["detected"]
                        lighting_ok = analysis["quality"]["lighting"] == "Good"
                        blur_ok = analysis["quality"]["blur"] == "Sharp"
                        visibility_ok = analysis["face"]["visibility"] == "Full"
                        size_ok = analysis["face"]["size"] == "Good"

                        verification_box.markdown(f"""
| Check | Status |
|---|---|
| Face Detected | {'✅' if face_ok else '❌'} {face_ok} |
| Visibility | {'✅' if visibility_ok else '⚠️'} {analysis['face']['visibility']} |
| Size | {'✅' if size_ok else '⚠️'} {analysis['face']['size']} |
| Lighting | {'✅' if lighting_ok else '⚠️'} {analysis['quality']['lighting']} |
| Brightness | {analysis['quality']['brightness']} |
| Blur | {'✅' if blur_ok else '⚠️'} {analysis['quality']['blur']} |
""")

                        _attn_status = analysis['attention']['status']
                        if _attn_status == 'Focused':
                            _attn_icon = '🟢'
                        elif _attn_status == 'Calibrating':
                            _attn_icon = '🟡'
                        else:
                            _attn_icon = '🔴'
                        _attn_label = _attn_status
                        if _attn_status == 'Calibrating':
                            _progress_pct = int(analysis['attention'].get('calibration_progress', 0) * 100)
                            _attn_label = f"Calibrating... please look at the screen ({_progress_pct}%)"

                        monitoring_box.markdown(f"""
#### {_attn_icon} {_attn_label}

**Attention**

| Parameter | Value |
|-----------|-------|
| Looking at Screen | {analysis['attention']['looking_at_screen']} |
| Head Direction | {analysis['attention']['head_direction']} |
| Gaze Direction | {analysis['attention']['gaze_direction']} |

**Behavior**

| Parameter | Value |
|-----------|-------|
| Looking Left | {analysis['behavior']['looking_left']} |
| Looking Right | {analysis['behavior']['looking_right']} |
| Multiple Faces | {analysis['behavior']['multiple_faces']} |
| Face Missing | {analysis['behavior']['face_missing']} |

**Objects**

| Parameter | Value |
|-----------|-------|
| Phone | {analysis['objects']['phone_detected']} |
| Person Count | {analysis['objects']['person_count']} |
| Object On Face | {analysis['objects']['object_on_face']} |
| Object On Eyes | {analysis['objects']['object_on_eyes']} |
""")
            elif not shown_waiting_msg:
                own_frame_box.info("Waiting for camera connection — click **Start** above.")
                shown_waiting_msg = True

            # ---- Mirror the OTHER participant's camera from the shared store ----
            if frame_count % OTHER_FRAME_REFRESH_EVERY_N_FRAMES == 0:
                other_jpeg = _read_other_frame(shared, other_role)
                if other_jpeg is not None:
                    other_frame_box.image(other_jpeg, use_container_width=True)
                other_label_box.caption(_display_name(other_role))
                own_label_box.caption(_display_name(role))

            # ---- Audio: pull pending mic frames and hand a chunk off to
            # the background worker every so often. ----
            got_audio = False
            if media_ctx.audio_receiver is not None:
                try:
                    audio_frames = media_ctx.audio_receiver.get_frames(timeout=0.01)
                except queue.Empty:
                    audio_frames = []

                if audio_frames:
                    got_audio = True
                    shown_waiting_mic_msg = False
                    batch = np.concatenate([_audio_frame_to_float32(af) for af in audio_frames])
                    if batch.size:
                        latest_mic_rms = float(np.sqrt(np.mean(np.square(batch))))

                now_a = time.time()
                dt = max(0.0, now_a - last_iter_time)
                last_iter_time = now_a

                if not ambient_calibration_done:
                    if (now_a - session_start) <= STT_AMBIENT_CALIBRATION_SECONDS:
                        if audio_frames and batch.size:
                            ambient_calibration_samples.append(latest_mic_rms)
                    else:
                        if ambient_calibration_samples:
                            ambient_floor = float(np.median(ambient_calibration_samples))
                            effective_silence_threshold = min(
                                max(STT_SILENCE_RMS_THRESHOLD, ambient_floor * STT_AMBIENT_NOISE_MULTIPLIER),
                                STT_SILENCE_RMS_THRESHOLD_MAX,
                            )
                        ambient_calibration_done = True
                        ambient_calibration_samples = []

                if (not audio_frames) or (batch.size and latest_mic_rms < effective_silence_threshold):
                    silence_run_seconds += dt
                else:
                    silence_run_seconds = 0.0

                for af in audio_frames:
                    af.pts = None
                    for rf in resampler.resample(af):
                        sound_buffer = np.concatenate([sound_buffer, _audio_frame_to_float32(rf)])

                buffer_seconds = sound_buffer.size / STT_TARGET_SR
                should_flush = (
                    buffer_seconds >= STT_MIN_CHUNK_SECONDS
                    and (silence_run_seconds >= STT_PAUSE_TRIGGER_SECONDS or buffer_seconds >= STT_MAX_CHUNK_SECONDS)
                    and now_a - last_audio_flush >= STT_TRANSCRIBE_INTERVAL_SECONDS
                )
                if should_flush:
                    chunk = sound_buffer
                    overlap_samples = int(STT_CHUNK_OVERLAP_SECONDS * STT_TARGET_SR)
                    if overlap_samples > 0 and chunk.size > overlap_samples:
                        sound_buffer = chunk[-overlap_samples:].copy()
                    else:
                        sound_buffer = np.array([], dtype=np.float32)
                    last_audio_flush = now_a
                    silence_run_seconds = 0.0

                    rms = float(np.sqrt(np.mean(np.square(chunk))))
                    chunk_seconds = chunk.size / STT_TARGET_SR
                    if rms >= effective_silence_threshold:
                        # Elapsed time relative to the SHARED session
                        # start (set once by the candidate) so recruiter
                        # and candidate segments merge onto one timeline.
                        chunk_start_elapsed = (now_a - session_start) - chunk_seconds
                        worker.submit_chunk(chunk, meta=max(0.0, chunk_start_elapsed))

                finished_text, finished_meta = worker.get_latest_text()
                if finished_text:
                    st.session_state.live_transcript = append_transcript(
                        st.session_state.live_transcript, finished_text
                    )
                    seg_time = finished_meta if finished_meta is not None else (now_a - session_start)
                    st.session_state.transcript_segments.append({"t": seg_time, "text": finished_text})
                    _publish_segment(shared, role, st.session_state.participant_name, seg_time, finished_text)

                # Redraw the merged transcript whenever ANY new segment
                # appeared — including ones published by the other role.
                current_seg_count = len(shared["transcript_segments"])
                if current_seg_count != last_seg_count_seen:
                    last_seg_count_seen = current_seg_count
                    _render_shared_transcript(transcript_box, shared)

                if not audio_frames:
                    latest_mic_rms *= 0.7

                now_level = time.time()
                if now_level - last_mic_level_update >= MIC_LEVEL_UPDATE_SECONDS:
                    last_mic_level_update = now_level
                    speaking = latest_mic_rms >= effective_silence_threshold
                    level_pct = min(1.0, latest_mic_rms / MIC_LEVEL_NORMALIZATION)
                    with mic_level_box.container():
                        st.progress(
                            level_pct,
                            text=("🎤 Audio detected — speaking" if speaking else "🎧 Mic connected — listening"),
                        )
            elif not shown_waiting_mic_msg:
                mic_level_box.info("🎙️ Waiting for microphone connection — click **Start** above.")
                shown_waiting_mic_msg = True

            if not got_video and not got_audio:
                time.sleep(0.02)

    finally:
        if not st.session_state.camera_running:
            worker_cleanup = st.session_state.transcription_worker
            if worker_cleanup is not None:
                worker_cleanup.stop()
                st.session_state.transcription_worker = None


def _render_shared_transcript(box, shared: dict):
    """Renders the merged, time-ordered recruiter+candidate transcript in
    the requested `[Time][Name]: Question:/Answer: ...` format."""
    segs = _read_shared_segments(shared)
    if not segs:
        box.info("Waiting for speech…")
        return

    lines = []
    for seg in segs:
        seg_role = seg["role"]
        name = seg.get("name") or ("Recruiter" if seg_role == "recruiter" else "Candidate")
        kind = "Question" if seg_role == "recruiter" else "Answer"
        safe_text = (
            seg["text"]
            .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        lines.append(
            f'<div style="color:#3f5fa5;margin-bottom:8px;line-height:1.4;">'
            f'[{_format_elapsed(seg["t"])}][{name}]: {kind}: {safe_text}</div>'
        )

    box.markdown(
        '<div style="background:#ffffff;border-radius:4px;padding:16px;">' + "".join(lines) + "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Stage 3 — report: LLM evaluation of every answer + aggregate (scoring
# logic unchanged — it only ever looked at the candidate's own transcript,
# which is still exactly what `live_transcript`/`qa_records` contain here).
# ---------------------------------------------------------------------------
def _render_report():
    import os

    st.subheader("📊 Interview Report")

    gemini_ready = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))

    if "interview_report" not in st.session_state:
        with st.spinner("Scoring your answers…"):
            report = llm_evaluator.evaluate_full_interview(st.session_state.qa_records)
        st.session_state.interview_report = report

        att_samples = st.session_state.attention_samples
        ec_samples = st.session_state.eye_contact_samples
        avg_attention = round(sum(att_samples) / len(att_samples), 1) if att_samples else 0
        avg_eye_contact = round(100 * sum(ec_samples) / len(ec_samples), 1) if ec_samples else 0
        report["avg_attention"] = avg_attention
        report["avg_eye_contact"] = avg_eye_contact

        candidate_id = st.session_state.get("candidate_id")
        candidate_id = candidate_store.upsert_candidate(
            candidate_id, st.session_state.get("resume_result", {}).get("candidate", {}).get("name", "Candidate"),
            {
                "interview_score": report["overall_score"],
                "interview_report": report,
            },
        )
        st.session_state.candidate_id = candidate_id

    report = st.session_state.interview_report
    if gemini_ready and report.get("method") == "heuristic":
        with st.spinner("Gemini key detected — rebuilding your interview report…"):
            report = llm_evaluator.evaluate_full_interview(st.session_state.qa_records)
            att_samples = st.session_state.attention_samples
            ec_samples = st.session_state.eye_contact_samples
            avg_attention = round(sum(att_samples) / len(att_samples), 1) if att_samples else 0
            avg_eye_contact = round(100 * sum(ec_samples) / len(ec_samples), 1) if ec_samples else 0
            report["avg_attention"] = avg_attention
            report["avg_eye_contact"] = avg_eye_contact

            st.session_state.interview_report = report

            candidate_id = st.session_state.get("candidate_id")
            candidate_id = candidate_store.upsert_candidate(
                candidate_id, st.session_state.get("resume_result", {}).get("candidate", {}).get("name", "Candidate"),
                {
                    "interview_score": report["overall_score"],
                    "interview_report": report,
                },
            )
            st.session_state.candidate_id = candidate_id

    if report.get("method") == "heuristic":
        st.warning("No GEMINI_API_KEY configured — scores below are heuristic estimates, not LLM-graded content evaluation.")

    m1, m2, m3 = st.columns(3)
    m1.metric("Overall Interview Score", f"{report['overall_score']}%")
    m2.metric("Avg. Attention", f"{report.get('avg_attention', 0)}%")
    m3.metric("Avg. Eye Contact", f"{report.get('avg_eye_contact', 0)}%")

    st.markdown("**Score breakdown**")
    for dim, score in report["aggregate_scores"].items():
        st.write(f"{dim.replace('_', ' ').title()} — {score}%")
        st.progress(min(1.0, score / 100))

    st.divider()
    st.markdown("**🚩 Distraction Summary**")
    st.caption(
        "How many separate times each behavior/object was flagged during the "
        "session (counts are spaced at least 5s apart, so one long lapse "
        "counts as a few occurrences, not hundreds of frames)."
    )
    _counts = st.session_state.get("distraction_counts", {})
    _total_flags = sum(_counts.get(label, 0) for label, _, _ in DISTRACTION_PARAMETERS)
    if _total_flags == 0:
        st.success("No distractions flagged during this session.")
    else:
        _rows = "\n".join(
            f"| {label} | {_counts.get(label, 0)} |" for label, _, _ in DISTRACTION_PARAMETERS
        )
        st.markdown(f"""
| Parameter | Count |
|-----------|-------|
{_rows}
""")

    # Full recruiter+candidate dialogue, merged and timestamped, exactly
    # as it appeared live during the session.
    shared_snapshot = st.session_state.get("shared_transcript_snapshot") or []
    if shared_snapshot:
        st.divider()
        st.markdown("**🕒 Timestamped Transcript (Recruiter + Candidate)**")
        st.caption("What was said, by whom, and when, over the course of the session.")
        with st.expander("View full timed transcript", expanded=False):
            for seg in sorted(shared_snapshot, key=lambda s: s["t"]):
                kind = "Question" if seg["role"] == "recruiter" else "Answer"
                name = seg.get("name") or ("Recruiter" if seg["role"] == "recruiter" else "Candidate")
                st.write(f"**[{_format_elapsed(seg['t'])}][{name}]:** {kind}: {seg['text']}")
    elif st.session_state.get("transcript_segments"):
        st.divider()
        st.markdown("**🕒 Timestamped Transcript**")
        st.caption("What the candidate said, and when, over the course of the session.")
        with st.expander("View full timed transcript", expanded=False):
            for seg in st.session_state.transcript_segments:
                st.write(f"**[{_format_elapsed(seg['t'])}]** {seg['text']}")

    # st.divider()
    # st.markdown("**Per-question detail**")
    # for i, ev in enumerate(report["evaluations"], 1):
    #     with st.expander(f"Q{i}: {ev['question']}"):
    #         st.write(f"**Transcript:** {ev['transcript'] or '_No answer captured_'}")
    #         st.write(f"**Overall:** {ev['overall']}%")
    #         cols = st.columns(3)
    #         for j, (dim, score) in enumerate(ev["scores"].items()):
    #             cols[j % 3].write(f"{dim.replace('_',' ').title()}: {score}%")
    #         if ev.get("feedback"):
    #             st.info(ev["feedback"])
    #         if ev.get("strengths"):
    #             st.write("Strengths: " + ", ".join(ev["strengths"]))
    #         if ev.get("improvements"):
    #             st.write("To improve: " + ", ".join(ev["improvements"]))

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔁 Retake Interview"):
            for k in ["qa_records", "interview_report", "attention_samples", "eye_contact_samples",
                      "transcript_segments", "live_transcript", "session_start_time",
                      "distraction_counts", "_last_violation_time", "shared_transcript_snapshot",
                      "role", "session_code", "pending_session_code"]:
                st.session_state.pop(k, None)
            st.session_state.interview_stage = "setup"
            st.rerun()
    with c2:
        if st.button("Continue to Learning Roadmap →", type="primary"):
            st.session_state.nav_target = "Learning Roadmap"
            st.rerun()