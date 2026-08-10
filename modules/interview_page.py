"""
interview_page.py
==================
AI Interview module — candidate-only camera + live face/attention
detection + live mic transcription, built on face1.py's real-time
architecture.

Why "Camera could not be opened" was happening
------------------------------------------------
The previous version opened the camera with `cv2.VideoCapture(0)` —
that grabs a webcam device attached to the SERVER the Streamlit app is
running on, not the candidate's own browser. The mic already avoided
this problem by going through `streamlit_webrtc.webrtc_streamer`, which
streams audio from the candidate's browser to the server over WebRTC.
Video was never given the same treatment, so on any real/hosted
deployment (server has no camera device attached) it failed for every
candidate, every time — not a fluke.

Fix: video is now captured the same way audio already was — through a
single combined `webrtc_streamer` connection (video + audio together,
one "Start" button for both). Frames are pulled server-side via
`ctx.video_receiver.get_frame()`, exactly parallel to how audio frames
are pulled via `ctx.audio_receiver.get_frames()`. There is no
`cv2.VideoCapture` anywhere in this file anymore.

Why attention/behavior/objects were lagging
-----------------------------------------------
face1.py's `LatestFrameCamera` is a background thread that continuously
OVERWRITES a single `self.frame` variable, so there's structurally no
way for a backlog to build up — you always read whatever is most
current. The WebRTC video source is different: `video_receiver` holds a
FIFO QUEUE fed by the browser's camera. Pulling exactly one frame per
loop iteration (as the previous version did) means that whenever a
single iteration takes even slightly longer than the camera's frame
interval — drawing the markdown tables, checking audio, etc. — frames
pile up in that queue, and the next `get_frame()` call just hands you
the NEXT (increasingly stale) frame in the backlog instead of the
CURRENT one. That backlog only ever grows, which is exactly what shows
up as delayed attention/behavior/objects.

Fix: every loop iteration now drains the queue completely and keeps
only the newest frame, discarding any backlog (see the video block in
`_render_interview`) — restoring the same "always the latest frame"
guarantee `LatestFrameCamera` gave you.

Why the earlier freeze was happening
--------------------------------------
The speech model load used to happen directly in the main render thread,
right before the loop started — a slow blocking call the first time,
freezing the whole page until it finished. It is now loaded *inside*
`TranscriptionWorker`'s own background thread, so the render loop never
waits on it: camera/tables render immediately, and live transcription
switches on automatically once SenseVoice finishes loading.

Audio/transcription pipeline (ported from voice.py + transcript_utils.py)
---------------------------------------------------------------------------
This module now uses a single speech backend: <SenseVoice> via funasr.
Audio is resampled to mono 16kHz with `av`, silence-gated by RMS
threshold, and decoded chunk-by-chunk. `append_transcript` is imported
from `backend.transcript_utils` (the dedup-aware version — trims overlap
at chunk boundaries) instead of a local helper.

One bug fixed while porting voice.py's flush logic: it reset
`sound_buffer` to empty on every loop tick (since its flush interval
was 0.0) *before* checking whether there was enough audio — so a
partial chunk below the 0.6s minimum was discarded every single tick
and could never accumulate up to that minimum, meaning transcription
would never actually fire. Here, the buffer is only cleared once it
has genuinely reached the minimum size; otherwise it keeps accumulating
across iterations.

Trade-off worth knowing: if `TranscriptionWorker` is still busy with the
last chunk, it drops the next chunk rather than queueing and adding lag.
On CPU-only servers, this can produce fewer, larger transcribed chunks;
on GPU-backed deployments this effect is usually reduced.

Layout
------
Left column (wide):
  1. Combined camera+mic WebRTC widget (candidate's own browser feed)
  2. Live camera (server-side annotated frame, face box + detections)
  3. Live Transcript, directly below

Right column (narrow):
  1. System Verification table (face detected / visibility / size /
     lighting / brightness / blur)
  2. Attention, Behavior & Objects table

There is no "Save answer & continue" step. The candidate just talks —
camera and mic both run continuously — and clicks "End Interview" once
when they are done; the whole session's transcript becomes the answer
that gets scored on the report page.
"""
import queue
import os
import time
import threading

import av
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
# localhost/loopback — the moment there's any real network hop between
# the candidate's browser and the server (a hosted deployment, a
# different subnet, certain routers/firewalls), the peer connection can
# fail silently: video and/or audio just never arrive, with no error
# shown anywhere. A public STUN server fixes this for the vast majority
# of networks. If you're behind a particularly strict corporate
# firewall/NAT, you may eventually need a TURN server instead — STUN
# alone doesn't relay media, it only helps two peers discover a direct
# path to each other.
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})


def stop_session():
    """Tears down the transcription worker. The WebRTC connection itself
    is owned by the webrtc_streamer component tied to this script run;
    it stops automatically once we stop calling it (i.e. once we leave
    the in_progress stage)."""
    st.session_state.camera_running = False

    worker = st.session_state.get("transcription_worker")
    if worker is not None:
        worker.stop()
        st.session_state.transcription_worker = None


# ---------------------------------------------------------------------------
# Live speech-to-text — SenseVoice only
# ---------------------------------------------------------------------------
STT_TARGET_SR = 16000                        # SenseVoice pipeline uses 16kHz mono audio
STT_TRANSCRIBE_INTERVAL_SECONDS = 0.0        # minimum time between flush attempts
STT_MIN_CHUNK_SECONDS = float(os.getenv("STT_MIN_CHUNK_SECONDS", "0.5"))  # only flush once buffer has at least this much audio
# STT_SILENCE_RMS_THRESHOLD = 0.002            # below this average energy, treat chunk as silence
STT_SILENCE_RMS_THRESHOLD = 0.0005            # below this average energy, treat chunk as silence

# Every chunk is decoded independently (no context carried between them),
# so a word spoken right at a flush boundary gets split across two
# chunks and neither one hears the whole word — this is usually the
# single biggest source of "almost right" transcription. Fix: instead of
# fully draining sound_buffer on flush, keep the last
# STT_CHUNK_OVERLAP_SECONDS of audio and carry it into the start of the
# next chunk, so any word cut at the boundary appears whole at least
# once. append_transcript (transcript_utils.py) does the matching
# word-level overlap trim on the text side so the duplicated audio
# doesn't turn into duplicated words. Kept at roughly ~30% of
# STT_MIN_CHUNK_SECONDS by default — enough to catch a full word/syllable
# without the overlap eating most of a very short buffer (at
# MIN_CHUNK_SECONDS=0.5 in particular, a fixed 0.4s overlap would leave
# almost no genuinely new audio per chunk). Set to 0 to disable.
STT_CHUNK_OVERLAP_SECONDS = float(os.getenv("STT_CHUNK_OVERLAP_SECONDS", str(round(STT_MIN_CHUNK_SECONDS * 0.3, 2))))

# Pause-triggered flush: rather than always waiting for a fixed-size
# window (which forces short windows like MIN_CHUNK_SECONDS=0.5 to cut
# through the middle of continuous speech), the buffer is allowed to
# keep growing past MIN_CHUNK_SECONDS as long as you're actively
# talking, and only flushes once it (a) has at least MIN_CHUNK_SECONDS
# of audio AND (b) detects a short natural lull — giving each decode a
# whole phrase instead of an arbitrary time-slice, while still flushing
# promptly right after you pause. MAX_CHUNK_SECONDS is a safety cap so
# continuous uninterrupted speech (no pauses) still gets flushed
# periodically instead of growing forever.
STT_PAUSE_TRIGGER_SECONDS = float(os.getenv("STT_PAUSE_TRIGGER_SECONDS", "0.35"))
STT_MAX_CHUNK_SECONDS = float(os.getenv("STT_MAX_CHUNK_SECONDS", "4.0"))

# SenseVoice (via funasr) model id — auto-downloaded from ModelScope/HF
# on first run and cached locally after that. "iic/SenseVoiceSmall" is the only
# public SenseVoice checkpoint at time of writing.
SENSEVOICE_MODEL_NAME = os.getenv("SENSEVOICE_MODEL_NAME", "iic/SenseVoiceSmall")
SENSEVOICE_LANGUAGE = os.getenv("SENSEVOICE_LANGUAGE", "en")  # this app is English-only

_sensevoice_singleton_lock = threading.Lock()
_sensevoice_singleton = {"backend": None}


def _load_stt_backend_once():
    """Loads (once per server process) and caches the SenseVoice backend.

    Deliberately NOT an `st.cache_resource` function: those are meant to
    be called from the main Streamlit thread, and calling one from a
    background thread risks touching Streamlit's UI machinery from the
    wrong thread. A plain module-level singleton guarded by a lock is
    simpler and completely safe to call from TranscriptionWorker's
    background thread.
    """
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
    """Downmixes to mono and resamples to 16kHz at the container level
    (pyav/ffmpeg) — matches voice.py's make_resampler exactly."""
    return av.AudioResampler(format="s16", layout="mono", rate=STT_TARGET_SR)


def _audio_frame_to_float32(audio_frame: av.AudioFrame) -> np.ndarray:
    """Convert an already-resampled mono s16 av.AudioFrame to float32 [-1, 1]."""
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
    """Transcribes one chunk with SenseVoice (English)."""
    if audio_array.size == 0:
        return ""

    # Decode each silence-gated chunk independently and append with
    # transcript_utils overlap trimming on the text side.
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
    SenseVoice model itself. The main render loop never blocks on this
    class for anything: it only ever calls the cheap, non-blocking
    `submit_chunk` / `get_latest_text` / `is_ready`."""

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
        """Non-blocking: returns the load error message, or None if the
        model loaded fine (or hasn't finished loading yet)."""
        with self.lock:
            return self.load_error

    def get_transcribe_error(self):
        """Non-blocking: returns the most recent transcribe-time error
        message, or None. Without this, a transcribe() call raising is
        silently swallowed into an empty string — exactly how "mic
        detects audio but no transcript ever appears" can happen with
        zero visible cause."""
        with self.lock:
            return self.transcribe_error

    def submit_chunk(self, chunk: np.ndarray, meta=None):
        """Fire-and-forget. Dropped (not queued) if the worker is still
        busy or the model isn't loaded yet, so lag can never build up.
        `meta` is opaque to the worker — here it's used to carry the
        chunk's start time (interview-elapsed seconds) through to
        get_latest_text(), so timed transcription can attribute text to
        when it was actually spoken, not when it happened to finish
        transcribing."""
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
        """Non-blocking: returns (text, meta) — text is None if nothing
        new finished since the last call. `meta` is whatever was passed
        to submit_chunk() for this result (here: the chunk's start time
        in interview-elapsed seconds)."""
        with self.lock:
            text = self.latest_text
            meta = self.latest_meta
            self.latest_text = None
            self.latest_meta = None
            return text, meta

    def stop(self):
        self.stopped = True


# ---------------------------------------------------------------------------
# Violation snapshots — same cooldown pattern as face1.py, so the
# existing Session Report tab keeps working unmodified.
#
# This cooldown also gates DISTRACTION COUNTING (distraction_counts
# below): each time a violation type is actually logged (i.e. it's been
# active for >= VIOLATION_COOLDOWN_SECONDS since the last time this same
# type was logged), the running total for that type is incremented too.
# That means the count reflects distinct occurrences spaced >= 5s apart,
# not a raw per-frame tally — a candidate who looks left continuously for
# 30s counts as ~6, not several hundred (one per analyzed frame).
# ---------------------------------------------------------------------------
VIOLATION_COOLDOWN_SECONDS = 15

# Every parameter we count "distraction occurred" for on the scoring
# page, and the (behavior|objects) analysis dict + key each maps to.
# Single source of truth — used both to log/count during the interview
# and to render the summary table on the report page, so the two can't
# drift out of sync.
DISTRACTION_PARAMETERS = [
    ("Looking Left",          "behavior", "looking_left"),
    ("Looking Right",         "behavior", "looking_right"),
    ("Multiple Faces",        "behavior", "multiple_faces"),
    # Face Covered/Missing now also covers a hand on the face/eyes —
    # detect.py folds hand_on_face/hand_on_eyes into behavior.face_missing
    # itself, so a hand covering the face is one flagged occurrence here,
    # not a second, separate "Hand On Face"/"Hand On Eyes" count.
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
        "qa_records": [],       # [{question, skill, transcript, duration}]
        "interview_stage": "setup",  # setup -> in_progress -> report
        "attention_samples": [],
        "eye_contact_samples": [],
        "live_transcript": "",        # continuous transcript for the whole session
        "transcript_segments": [],   # [{"t": elapsed_seconds, "text": "..."}] — timed transcription
        "session_start_time": None,  # set when Start Interview is clicked; basis for all timestamps
        "_last_violation_time": {},
        "distraction_counts": {},     # {"Looking Left": 3, "Phone Detected": 1, ...} — see DISTRACTION_PARAMETERS
        # Bumped every time "Start Interview" is clicked and used as part
        # of webrtc_streamer's `key` below. See the note above that call
        # for why this can't just be a fixed string.
        "media_session_id": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def render():
    _init_state()
    st.title("🎥 AI Mock Interview")

    if st.session_state.interview_stage == "setup":
        _render_setup()
    elif st.session_state.interview_stage == "in_progress":
        _render_interview()
    else:
        _render_report()


# ---------------------------------------------------------------------------
# Stage 1 — setup
# ---------------------------------------------------------------------------
def _render_setup():
    # st.info(
    #     # "Start the interview and speak freely about your background, projects, and experience "
    #     "while the interviewer asks questions on their end. "
    #     "No questions are generated or shown by this page."
    # )
    st.caption(
        "You'll be asked to allow camera & microphone access in your browser. Your face, attention, "
        "and behavior will be tracked throughout, and your speech will be transcribed live"
    )
    if st.button("▶️ Start Interview", type="primary"):
        # detect.py's calibration state (_state["calibrated"], neutral_yaw/
        # neutral_pitch) lives in a MODULE-LEVEL dict, not st.session_state
        # — it survives across Streamlit reruns AND across different
        # candidates' sessions in the same server process. Without this
        # call, calibration only ever runs once (on whoever's face is
        # first seen after the server starts) and every candidate after
        # that reuses that first person's neutral pose, which is exactly
        # why it "isn't working" for anyone else. Must be called here,
        # each time a NEW interview starts, so this candidate gets their
        # own ~3s calibration window instead of inheriting someone else's.
        reset_calibration()
        st.session_state.qa_records = []
        st.session_state.interview_stage = "in_progress"
        st.session_state.camera_running = True
        st.session_state.live_transcript = ""
        st.session_state.transcript_segments = []
        st.session_state.session_start_time = time.time()
        # Fresh counts/cooldowns for this attempt — without this, counts
        # from a previous attempt in the same browser session (e.g. after
        # "Retake Interview") would keep accumulating into this one.
        st.session_state.distraction_counts = {}
        st.session_state._last_violation_time = {}
        # New webrtc_streamer key for this attempt (see the note above
        # that call in _render_interview) — required so camera/mic
        # actually (re)start on a second or later "Start Interview" click
        # in the same browser session, not just the first.
        st.session_state.media_session_id += 1
        st.rerun()


# ---------------------------------------------------------------------------
# Stage 2 — in progress: single script run, internal render loop (no
# per-frame st.rerun()). Camera + mic both come from ONE combined WebRTC
# connection to the candidate's own browser. The End Interview button
# sits OUTSIDE the loop so clicking it interrupts the loop via
# Streamlit's normal rerun mechanism.
# ---------------------------------------------------------------------------
def _render_interview():
    top_l, top_r = st.columns([3, 1])
    with top_l:
        st.caption("The interviewer asks questions on their end — just speak naturally; camera and mic are both live.")
    with top_r:
        end_clicked = st.button("⏹️ End Interview", type="primary", use_container_width=True)

    if end_clicked:
        transcript = st.session_state.live_transcript.strip()
        st.session_state.qa_records = []
        if transcript:
            st.session_state.qa_records.append({
                "question": "Full interview response",
                "skill": "general",
                "transcript": transcript,
                "duration": 0.0,
                "segments": list(st.session_state.transcript_segments),  # timed breakdown, for report display only
            })
        st.session_state.interview_stage = "report"
        stop_session()
        st.rerun()

    left, right = st.columns([2, 1])

    with left:
        st.subheader("Camera & Microphone")
        st.caption("Click **Start** once to enable both your camera and mic.")
        # ONE combined connection for video + audio — this is the actual
        # candidate webcam/mic, captured through the browser, not a
        # device attached to the server.
        #
        # IMPORTANT: the key includes media_session_id, which is bumped
        # once each time the "Start Interview" button is clicked (see
        # _render_setup). streamlit-webrtc ties a component's internal
        # RTCPeerConnection / media-stream state to its `key`, on both
        # the frontend and in st.session_state. With a fixed key here,
        # the very first interview attempt in a browser session works
        # fine, but leaving this stage (End Interview without finishing,
        # navigating away, "Retake Interview", etc.) and then clicking
        # "Start Interview" again reuses that exact same component
        # instance instead of creating a new one — so the browser never
        # issues a fresh getUserMedia()/RTCPeerConnection request, and
        # camera + mic silently fail to (re)start on the second attempt
        # onward. Changing the key on every attempt forces a genuinely
        # new component/connection each time.
        media_ctx = webrtc_streamer(
            key=f"interview-media-{st.session_state.media_session_id}",
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

        st.divider()
        st.subheader("Live camera (analyzed)")
        frame_placeholder = st.empty()
        frame_placeholder.info("Waiting for camera connection — click **Start** above.")

        st.divider()
        st.subheader("🎙️ Live Transcript")
        transcript_status = st.empty()
        transcribe_error_box = st.empty()
        transcript_box = st.empty()
        if st.session_state.transcript_segments:
            transcript_box.markdown(
                "\n\n".join(
                    f"**[{_format_elapsed(seg['t'])}]** {seg['text']}"
                    for seg in st.session_state.transcript_segments
                )
            )
        else:
            transcript_box.info("Waiting for speech…")

    with right:
        st.subheader("✅ System Verification")
        verification_box = st.empty()
        st.divider()
        st.subheader("📡 Attention, Behavior & Objects")
        monitoring_box = st.empty()

    # The worker loads SenseVoice itself, on its own thread — created once
    # and reused across reruns via session_state, so the model is only
    # ever loaded a single time per server process.
    if st.session_state.transcription_worker is None:
        st.session_state.transcription_worker = TranscriptionWorker()
    worker = st.session_state.transcription_worker

    PROCESS_EVERY_N_FRAMES = 2    # run face-detection every Nth new frame
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

    # Mic-level meter state — a rough, always-on indicator of whether
    # audio is reaching the server at all vs. loud enough to count as
    # speech (RMS vs. STT_SILENCE_RMS_THRESHOLD). If the meter shows
    # "audio detected" but transcripts still aren't appearing, the
    # problem is downstream in speech transcription — not the mic.
    last_mic_level_update = 0.0
    latest_mic_rms = 0.0
    total_audio_frames_received = 0
    MIC_LEVEL_UPDATE_SECONDS = 0.2
    MIC_LEVEL_NORMALIZATION = 0.05  # rough scale for the progress bar (typical speech RMS)

    # Debug counters — surfaced in the mic caption so a "meter shows audio
    # but nothing transcribes" report can be diagnosed from the UI alone:
    # is the *resampled* chunk actually carrying signal (chunk_rms), is it
    # clearing the silence gate (chunks_submitted), and is the worker
    # actually being asked to transcribe it at all.
    last_chunk_rms = None
    last_chunk_seconds = None
    chunks_flushed = 0
    chunks_submitted = 0
    raw_transcriptions_count = 0
    last_raw_model_output = None

    try:
        while st.session_state.camera_running:
            # ---- Speech-model status (cheap, non-blocking check) ----
            worker_ready = worker.is_ready()
            worker_load_error = worker.get_load_error()
            if worker_load_error is not None:
                if last_worker_ready_shown != "failed":
                    transcript_status.error(
                        f"❌ Speech model failed to load: {worker_load_error}\n\n"
                        "Camera and detection above are unaffected — this only blocks transcription. "
                        "This build uses SenseVoice only. Common causes: `funasr` not installed, "
                        "first-run model download failed (no internet), or an incompatible `torch`/`funasr` install."
                    )
                    last_worker_ready_shown = "failed"
            elif worker_ready != last_worker_ready_shown:
                if worker_ready:
                    transcript_status.empty()
                else:
                    transcript_status.caption(
                        "🔄 Loading speech model in the background — camera and detection are already "
                        "live; transcription will switch on automatically in a few seconds."
                    )
                last_worker_ready_shown = worker_ready

            # ---- Transcribe-time errors (distinct from load errors) —
            # without this, a transcribe() call raising on one chunk is
            # silently swallowed into an empty string, which is exactly
            # how "mic detects audio but no transcript ever appears"
            # can happen with no visible cause anywhere. ----
            worker_transcribe_error = worker.get_transcribe_error()
            if worker_transcribe_error != last_transcribe_error_shown:
                if worker_transcribe_error is not None:
                    transcribe_error_box.warning(f"⚠️ Last transcription attempt failed: {worker_transcribe_error}")
                else:
                    transcribe_error_box.empty()
                last_transcribe_error_shown = worker_transcribe_error

            # ---- Video: drain everything currently queued from the
            # browser's camera and keep only the MOST RECENT frame —
            # mirrors LatestFrameCamera's "always the newest frame"
            # design from face1.py. Without this, any loop iteration
            # that takes even slightly longer than the camera's frame
            # interval (drawing tables, pulling audio, etc.) lets frames
            # pile up in the WebRTC queue, and get_frame() just hands
            # you the NEXT one in that backlog instead of the CURRENT
            # one — that growing backlog is exactly what shows up as
            # delayed attention/behavior/objects.
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

                    if frame_count % VIDEO_UPDATE_EVERY_N_FRAMES == 0:
                        frame_placeholder.image(frame, channels="BGR", use_container_width=True)

                    if frame_count % PANEL_UPDATE_EVERY_N_FRAMES == 0:
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
                            # Distinct from red/Distracted — calibration
                            # treats the candidate as attentive (see
                            # detect.py), so showing red here would read
                            # as a false "you're distracted" for the ~3s
                            # neutral-pose window right at session start.
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
                frame_placeholder.info("Waiting for camera connection — click **Start** above.")
                shown_waiting_msg = True

            # ---- Audio: pull pending mic frames (cheap) and hand a
            # chunk off to the background worker every couple seconds.
            # SenseVoice runs on the worker's own thread — this loop
            # never waits on it, so video and mic stay live together.
            got_audio = False
            if media_ctx.audio_receiver is not None:
                try:
                    audio_frames = media_ctx.audio_receiver.get_frames(timeout=0.01)
                except queue.Empty:
                    audio_frames = []

                if audio_frames:
                    got_audio = True
                    shown_waiting_mic_msg = False
                    total_audio_frames_received += len(audio_frames)
                    # RMS of this batch, purely for the live meter — kept
                    # separate from `sound_buffer` (which accumulates
                    # resampled audio for actual transcription) so the
                    # meter updates every tick instead of only once the
                    # buffer is big enough to flush.
                    batch = np.concatenate([_audio_frame_to_float32(af) for af in audio_frames])
                    if batch.size:
                        latest_mic_rms = float(np.sqrt(np.mean(np.square(batch))))

                # Track how long we've been in a lull, in wall-clock time
                # (not sample count, so it works the same regardless of
                # mic sample rate). Two cases count as "quiet": genuine
                # low-RMS audio, and WebRTC's Opus DTX simply not sending
                # any packets at all during silence (see note below) —
                # both mean "nothing new being said right now".
                now_a = time.time()
                dt = max(0.0, now_a - last_iter_time)
                last_iter_time = now_a
                if (not audio_frames) or (batch.size and latest_mic_rms < STT_SILENCE_RMS_THRESHOLD):
                    silence_run_seconds += dt
                else:
                    silence_run_seconds = 0.0

                for af in audio_frames:
                    af.pts = None
                    for rf in resampler.resample(af):
                        sound_buffer = np.concatenate([sound_buffer, _audio_frame_to_float32(rf)])

                # Flush once the buffer has genuinely reached the minimum
                # size AND either (a) speech just paused, or (b) we've
                # hit the max-chunk safety cap for continuous speech.
                # (voice.py reset the buffer before the size check at all,
                # which meant a too-small chunk was thrown away every
                # single iteration and could never accumulate to the
                # minimum; fixed here so audio actually keeps building up
                # until there's enough of it to transcribe.)
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
                        # Carry the tail forward instead of clearing to
                        # empty — the next chunk will start by re-hearing
                        # this same audio, so a word cut off here gets a
                        # second, whole-word chance in the next decode.
                        sound_buffer = chunk[-overlap_samples:].copy()
                    else:
                        sound_buffer = np.array([], dtype=np.float32)
                    last_audio_flush = now_a
                    silence_run_seconds = 0.0

                    rms = float(np.sqrt(np.mean(np.square(chunk))))
                    last_chunk_rms = rms
                    last_chunk_seconds = chunk.size / STT_TARGET_SR
                    chunks_flushed += 1
                    if rms >= STT_SILENCE_RMS_THRESHOLD:
                        chunks_submitted += 1
                        # Elapsed time (since Start Interview) at which
                        # THIS chunk's audio began, not when it happened
                        # to finish transcribing — that's what makes the
                        # eventual timestamp reflect when it was actually
                        # said.
                        session_start = st.session_state.session_start_time or now_a
                        chunk_start_elapsed = (now_a - session_start) - last_chunk_seconds
                        worker.submit_chunk(chunk, meta=max(0.0, chunk_start_elapsed))

                finished_text, finished_meta = worker.get_latest_text()
                if finished_text is not None:
                    raw_transcriptions_count += 1
                    last_raw_model_output = finished_text
                if finished_text:
                    st.session_state.live_transcript = append_transcript(
                        st.session_state.live_transcript, finished_text
                    )
                    st.session_state.transcript_segments.append({
                        "t": finished_meta if finished_meta is not None else 0.0,
                        "text": finished_text,
                    })
                    segments_markdown = "\n\n".join(
                        f"**[{_format_elapsed(seg['t'])}]** {seg['text']}"
                        for seg in st.session_state.transcript_segments
                    )
                    transcript_box.markdown(segments_markdown)

                # ---- Mic level meter (throttled so it doesn't redraw
                # on every single audio packet) ----
                # NOTE: WebRTC's Opus codec uses DTX (discontinuous
                # transmission) — when you're quiet, the browser simply
                # stops sending packets to save bandwidth. That's normal
                # and expected during any pause (thinking, listening to
                # the interviewer), NOT a dropped connection. So this no
                # longer treats "no packets for a few seconds" as an
                # error — it only ever reports "not connected" when the
                # WebRTC connection itself never came up (handled in the
                # `elif` below). During quiet periods the level just
                # decays back toward zero instead of alarming.
                if not audio_frames:
                    latest_mic_rms *= 0.7

                now_level = time.time()
                if now_level - last_mic_level_update >= MIC_LEVEL_UPDATE_SECONDS:
                    last_mic_level_update = now_level
                    speaking = latest_mic_rms >= STT_SILENCE_RMS_THRESHOLD
                    level_pct = min(1.0, latest_mic_rms / MIC_LEVEL_NORMALIZATION)
                    with mic_level_box.container():
                        st.progress(
                            level_pct,
                            text=("🎤 Audio detected — speaking" if speaking else "🎧 Mic connected — listening"),
                        )
                        # st.caption(
                        #     f"RMS level: {latest_mic_rms:.4f}  •  silence threshold: {STT_SILENCE_RMS_THRESHOLD}  "
                        #     f"•  frames received: {total_audio_frames_received}"
                        # )
                        # st.caption(
                        #     f"Buffer building: {sound_buffer.size / STT_TARGET_SR:.2f}s  "
                        #     f"•  silence run: {silence_run_seconds:.2f}s  "
                        #     f"•  pause trigger: {STT_PAUSE_TRIGGER_SECONDS}s  •  max cap: {STT_MAX_CHUNK_SECONDS}s"
                        # )
                        # if last_chunk_rms is not None:
                        #     st.caption(
                        #         f"Last resampled chunk: {last_chunk_seconds:.2f}s  •  chunk RMS: {last_chunk_rms:.5f}  "
                        #         f"•  flushed: {chunks_flushed}  •  passed silence gate & submitted: {chunks_submitted}  "
                        #         f"•  worker ready: {worker.is_ready()}  •  worker busy: {worker.busy}"
                        #     )
                        #     st.caption(
                        #         f"Raw non-empty model outputs so far: {raw_transcriptions_count}  "
                        #         f"•  last raw output: {last_raw_model_output!r}"
                        #     )
                        # else:
                        #     st.caption("Last resampled chunk: none flushed yet (buffer still filling)")
            elif not shown_waiting_mic_msg:
                mic_level_box.info("🎙️ Waiting for microphone connection — click **Start** above.")
                shown_waiting_mic_msg = True

            if not got_video and not got_audio:
                time.sleep(0.02)

    finally:
        # If the loop exits because the user clicked "End Interview"
        # (which interrupts this thread via Streamlit's rerun mechanism)
        # and that already turned camera_running off, make sure the
        # transcription worker is actually released.
        if not st.session_state.camera_running:
            worker_cleanup = st.session_state.transcription_worker
            if worker_cleanup is not None:
                worker_cleanup.stop()
                st.session_state.transcription_worker = None


# ---------------------------------------------------------------------------
# Stage 3 — report: LLM evaluation of every answer + aggregate (unchanged)
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

    if st.session_state.get("transcript_segments"):
        st.divider()
        st.markdown("**🕒 Timestamped Transcript**")
        st.caption("What the candidate said, and when, over the course of the session.")
        with st.expander("View full timed transcript", expanded=False):
            for seg in st.session_state.transcript_segments:
                st.write(f"**[{_format_elapsed(seg['t'])}]** {seg['text']}")

    st.divider()
    st.markdown("**Per-question detail**")
    for i, ev in enumerate(report["evaluations"], 1):
        with st.expander(f"Q{i}: {ev['question']}"):
            st.write(f"**Transcript:** {ev['transcript'] or '_No answer captured_'}")
            st.write(f"**Overall:** {ev['overall']}%")
            cols = st.columns(3)
            for j, (dim, score) in enumerate(ev["scores"].items()):
                cols[j % 3].write(f"{dim.replace('_',' ').title()}: {score}%")
            if ev.get("feedback"):
                st.info(ev["feedback"])
            if ev.get("strengths"):
                st.write("Strengths: " + ", ".join(ev["strengths"]))
            if ev.get("improvements"):
                st.write("To improve: " + ", ".join(ev["improvements"]))

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔁 Retake Interview"):
            for k in ["qa_records", "interview_report", "attention_samples", "eye_contact_samples",
                      "transcript_segments", "live_transcript", "session_start_time",
                      "distraction_counts", "_last_violation_time"]:
                st.session_state.pop(k, None)
            st.session_state.interview_stage = "setup"
            st.rerun()
    with c2:
        if st.button("Continue to Learning Roadmap →", type="primary"):
            st.session_state.nav_target = "Learning Roadmap"
            st.rerun()