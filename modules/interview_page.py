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
`load_whisper_model()` used to be called directly in the main render
thread, right before the loop started — a slow blocking call the first
time, freezing the whole page until it finished. It's now loaded
*inside* `TranscriptionWorker`'s own background thread, so the render
loop never waits on it: camera/tables render immediately, and live
transcription switches on automatically once the model finishes
loading in the background.

Audio/transcription pipeline (ported from voice.py + transcript_utils.py)
---------------------------------------------------------------------------
Per your voice.py, this now uses Hugging Face `transformers`
(`WhisperProcessor` + `WhisperForConditionalGeneration`, model
"openai/whisper-small") instead of faster-whisper, with the same
resample-to-mono-16kHz-via-av, RMS silence gate, and greedy-decode
`transcribe()` approach. `append_transcript` is imported directly from
`backend.transcript_utils` (the dedup-aware version — skips appending a
segment that's already a substring/suffix of what's there) instead of a
local helper.

One bug fixed while porting voice.py's flush logic: it reset
`sound_buffer` to empty on every loop tick (since its flush interval
was 0.0) *before* checking whether there was enough audio — so a
partial chunk below the 0.6s minimum was discarded every single tick
and could never accumulate up to that minimum, meaning transcription
would never actually fire. Here, the buffer is only cleared once it
has genuinely reached the minimum size; otherwise it keeps accumulating
across iterations.

Trade-off worth knowing: `transformers`' `model.generate()` is
noticeably slower on CPU than faster-whisper's CTranslate2 backend for
the same model size. `TranscriptionWorker` drops a new chunk if it's
still busy transcribing the last one (rather than queuing/lagging), so
on a CPU-only server you may end up with fewer, larger chunks
transcribed rather than a chunk being missed outright. If you have a
GPU available this is much less of a concern.

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
import time
import threading

import av
import numpy as np
import streamlit as st
import torch
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from backend.detect import process_frame
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
# Live speech-to-text — ported from voice.py
# ---------------------------------------------------------------------------
STT_MODEL_NAME = "openai/whisper-medium.en"
STT_TARGET_SR = 16000                        # Whisper expects 16kHz audio
STT_TRANSCRIBE_INTERVAL_SECONDS = 0.0        # minimum time between flush attempts
STT_MIN_CHUNK_SECONDS = 1.2                  # only flush once buffer has at least this much audio
# STT_SILENCE_RMS_THRESHOLD = 0.002            # below this average energy, treat chunk as silence
STT_SILENCE_RMS_THRESHOLD = 0.0005            # below this average energy, treat chunk as silence

_whisper_singleton_lock = threading.Lock()
_whisper_singleton = {"processor": None, "model": None, "device": None}


def _load_whisper_model_once():
    """Loads (once per server process) and caches the processor/model/device.

    Deliberately NOT an `st.cache_resource` function: those are meant to
    be called from the main Streamlit thread, and calling one from a
    background thread risks touching Streamlit's UI machinery from the
    wrong thread. A plain module-level singleton guarded by a lock is
    simpler and completely safe to call from TranscriptionWorker's
    background thread.
    """
    with _whisper_singleton_lock:
        if _whisper_singleton["model"] is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            processor = WhisperProcessor.from_pretrained(STT_MODEL_NAME)
            model = WhisperForConditionalGeneration.from_pretrained(STT_MODEL_NAME).to(device)
            model.eval()
            _whisper_singleton["processor"] = processor
            _whisper_singleton["model"] = model
            _whisper_singleton["device"] = device
        return _whisper_singleton["processor"], _whisper_singleton["model"], _whisper_singleton["device"]


def _make_resampler() -> av.AudioResampler:
    """Downmixes to mono and resamples to 16kHz at the container level
    (pyav/ffmpeg) — matches voice.py's make_resampler exactly."""
    return av.AudioResampler(format="s16", layout="mono", rate=STT_TARGET_SR)


def _audio_frame_to_float32(audio_frame: av.AudioFrame) -> np.ndarray:
    """Convert an already-resampled mono s16 av.AudioFrame to float32 [-1, 1]."""
    samples = audio_frame.to_ndarray().astype(np.float32).flatten()
    return samples / 32768.0


def _transcribe_chunk(audio_array: np.ndarray, processor, model, device) -> str:
    """Greedy-decode transcription via transformers — matches voice.py's
    transcribe() (translation option omitted; this app is English-only)."""
    print("Whisper Started")
    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

    print("Transcribed:", text)

    return text
    if audio_array.size == 0:
        return ""
    inputs = processor(audio_array, sampling_rate=STT_TARGET_SR, return_tensors="pt")
    input_features = inputs["input_features"].to(device)
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            max_new_tokens=200,
            num_beams=1,
        )
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()


class TranscriptionWorker:
    """Runs entirely on its own background thread — INCLUDING loading the
    Whisper model itself. The main render loop never blocks on this
    class for anything: it only ever calls the cheap, non-blocking
    `submit_chunk` / `get_latest_text` / `is_ready`."""

    def __init__(self):
        self.lock = threading.Lock()
        self.processor = None
        self.model = None
        self.device = None
        self.ready = False
        self.load_failed = False
        self.load_error = None
        self.pending_chunk = None
        self.latest_text = None
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

    def submit_chunk(self, chunk: np.ndarray):
        """Fire-and-forget. Dropped (not queued) if the worker is still
        busy or the model isn't loaded yet, so lag can never build up."""
        with self.lock:
            if self.ready and not self.busy:
                self.pending_chunk = chunk
                self.busy = True

    def _run(self):
        try:
            self.processor, self.model, self.device = _load_whisper_model_once()
            with self.lock:
                self.ready = True
        except Exception as e:
            with self.lock:
                self.load_failed = True
                self.load_error = f"{type(e).__name__}: {e}"
            return

        while not self.stopped:
            chunk = None
            with self.lock:
                if self.pending_chunk is not None:
                    chunk = self.pending_chunk
                    self.pending_chunk = None
            if chunk is None:
                time.sleep(0.05)
                continue
            try:
                text = _transcribe_chunk(chunk, self.processor, self.model, self.device)
            except Exception:
                text = ""
            with self.lock:
                if text:
                    self.latest_text = text
                self.busy = False

    def get_latest_text(self):
        """Non-blocking: returns newly finished text, or None."""
        with self.lock:
            text = self.latest_text
            self.latest_text = None
            return text

    def stop(self):
        self.stopped = True


# ---------------------------------------------------------------------------
# Violation snapshots — same cooldown pattern as face1.py, so the
# existing Session Report tab keeps working unmodified.
# ---------------------------------------------------------------------------
VIOLATION_COOLDOWN_SECONDS = 5.0


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
        "_last_violation_time": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def render():
    _init_state()
    st.title("🎥 AI Interview")

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
    st.info(
        "Start the interview and speak freely about your background, projects, and experience "
        "while the interviewer asks questions on their end. "
        "No questions are generated or shown by this page."
    )
    st.caption(
        "You'll be asked to allow camera + microphone access in your browser. Your face, attention, "
        "and behavior are tracked throughout, and your speech is transcribed live — nothing is uploaded anywhere."
    )
    if st.button("▶️ Start Interview", type="primary"):
        st.session_state.qa_records = []
        st.session_state.interview_stage = "in_progress"
        st.session_state.camera_running = True
        st.session_state.live_transcript = ""
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
        media_ctx = webrtc_streamer(
            key="interview-media",
            mode=WebRtcMode.SENDONLY,
            rtc_configuration=RTC_CONFIGURATION,
            video_receiver_size=32,
            audio_receiver_size=128,
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
        transcript_box = st.empty()
        if st.session_state.live_transcript:
            transcript_box.markdown(st.session_state.live_transcript)
        else:
            transcript_box.info("Waiting for speech…")

    with right:
        st.subheader("✅ System Verification")
        verification_box = st.empty()
        st.divider()
        st.subheader("📡 Attention, Behavior & Objects")
        monitoring_box = st.empty()

    # The worker loads Whisper itself, on its own thread — created once
    # and reused across reruns via session_state, so the model is only
    # ever loaded a single time per server process.
    if st.session_state.transcription_worker is None:
        st.session_state.transcription_worker = TranscriptionWorker()
    worker = st.session_state.transcription_worker

    PROCESS_EVERY_N_FRAMES = 5     # run face-detection every Nth new frame
    VIDEO_UPDATE_EVERY_N_FRAMES = 2
    PANEL_UPDATE_EVERY_N_FRAMES = 5

    frame_count = 0
    last_analysis = None
    last_worker_ready_shown = None
    shown_waiting_msg = True
    shown_waiting_mic_msg = True

    resampler = _make_resampler()
    sound_buffer = np.array([], dtype=np.float32)
    last_audio_flush = time.time()

    # Mic-level meter state — a rough, always-on indicator of whether
    # audio is reaching the server at all vs. loud enough to count as
    # speech (RMS vs. STT_SILENCE_RMS_THRESHOLD). If the meter shows
    # "audio detected" but transcripts still aren't appearing, the
    # problem is downstream in Whisper/transcription — not the mic.
    last_mic_level_update = 0.0
    latest_mic_rms = 0.0
    total_audio_frames_received = 0
    MIC_LEVEL_UPDATE_SECONDS = 0.2
    MIC_LEVEL_NORMALIZATION = 0.05  # rough scale for the progress bar (typical speech RMS)

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
                        "Common causes: no internet access to huggingface.co to download "
                        f"\"{STT_MODEL_NAME}\" the first time, or a missing/incompatible `transformers`/`torch` install."
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

                        monitoring_box.markdown(f"""
#### {'🟢' if analysis['attention']['status'] == 'Focused' else '🔴'} {analysis['attention']['status']}

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
| Looking Down | {analysis['behavior']['looking_down']} |
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
            # Whisper itself runs on the worker's own thread — this loop
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

                for af in audio_frames:
                    af.pts = None
                    for rf in resampler.resample(af):
                        sound_buffer = np.concatenate([sound_buffer, _audio_frame_to_float32(rf)])

                # Flush to the transcription worker once the buffer has
                # genuinely reached the minimum size — NOT on every tick.
                # (voice.py reset the buffer before this size check, which
                # meant a too-small chunk was thrown away every single
                # iteration and could never accumulate to the minimum;
                # fixed here so audio actually keeps building up until
                # there's enough of it to transcribe.)
                now_a = time.time()
                if (
                    sound_buffer.size > int(STT_MIN_CHUNK_SECONDS * STT_TARGET_SR)
                    and now_a - last_audio_flush >= STT_TRANSCRIBE_INTERVAL_SECONDS
                ):
                    chunk = sound_buffer
                    sound_buffer = np.array([], dtype=np.float32)
                    last_audio_flush = now_a

                    rms = float(np.sqrt(np.mean(np.square(chunk))))
                    if rms >= STT_SILENCE_RMS_THRESHOLD:
                        worker.submit_chunk(chunk)

                finished_text = worker.get_latest_text()
                if finished_text:
                    st.session_state.live_transcript = append_transcript(
                        st.session_state.live_transcript, finished_text
                    )
                    transcript_box.markdown(st.session_state.live_transcript)

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
                        st.caption(
                            f"RMS level: {latest_mic_rms:.4f}  •  silence threshold: {STT_SILENCE_RMS_THRESHOLD}  "
                            f"•  frames received: {total_audio_frames_received}"
                        )
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
            for k in ["qa_records", "interview_report", "attention_samples", "eye_contact_samples"]:
                st.session_state.pop(k, None)
            st.session_state.interview_stage = "setup"
            st.rerun()
    with c2:
        if st.button("Continue to Learning Roadmap →", type="primary"):
            st.session_state.nav_target = "Learning Roadmap"
            st.rerun()