import cv2
import time
import numpy as np
import mediapipe as mdp
from ultralytics import YOLO
import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
model_path = os.path.join(
    ROOT_DIR, "models", "face_detection_yunet_2023mar.onnx"
)
yolo_model= os.path.join(
    ROOT_DIR, "models", "yolo26n.pt"
)
# ============================================================
# Face Quality Functions
# ============================================================

def check_brightness(face_gray):
    brightness = np.mean(face_gray)

    if brightness < 60:
        return "Too Dark", brightness
    elif brightness > 190:
        return "Too Bright", brightness

    return "Good", brightness


def check_blur(face_gray):
    score = cv2.Laplacian(face_gray, cv2.CV_32F).var()

    if score < 80:
        return "Blurry", score

    return "Sharp", score


def check_face_size(face_width, face_height, frame):
    h, w = frame.shape[:2]
    face_area = face_width * face_height
    frame_area = h * w
    ratio = face_area / frame_area

    if ratio < 0.05:
        return "Small"

    return "Good"


def check_visibility(x, y, x2, y2, frame):
    h, w = frame.shape[:2]

    if x <= 0 or y <= 0:
        return "Partial"
    if x2 >= w or y2 >= h:
        return "Partial"

    return "Full"


def boxes_overlap(box_a, box_b):
    """Simple rectangle-intersection check used for object-on-face heuristics."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


GREEN = (0, 255, 0)
RED = (0, 0, 255)


def draw_face_boxes(frame, last_faces, primary_ok):
    """
    Draws a colored box per detected face — no status text, just color:
      - the primary (first) face: green if primary_ok is True, else red
      - any additional face: always red (a second face means someone
        else is visibly in frame, which is itself the condition being
        flagged, regardless of the candidate's own attention state)
    primary_ok should already fold in both "looking at screen" and
    "nobody else detected" — see the call sites in process_frame().
    """
    for i, detected_face in enumerate(last_faces):
        bx, by, bw, bh = detected_face[:4].astype(int)
        color = GREEN if (i == 0 and primary_ok) else RED
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), color, 2)
        cv2.putText(frame, f"Face {i + 1}", (bx, by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)



# ============================================================
# One-time model loading
# ============================================================
# pip install opencv-contrib-python
#
# Download the YuNet model (one-time, ~2MB) from:
# https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
# and place it next to this script.
# ============================================================

# YUNET_MODEL_PATH = "models/face_detection_yunet_2023mar.onnx
# YUNET_MODEL_PATH = str(ROOT_DIR)
DETECT_EVERY_N_FRAMES = 3   # re-run YuNet face detection every N frames, reuse bbox in between
YOLO_EVERY_N_FRAMES = 5      # re-run YOLO object detection every N frames, reuse results in between

# How much to expand YuNet's detected face box (as a fraction of its
# width/height, on every side) before cropping for FaceMesh/head-pose.
# YuNet's box is frequently tight enough to exclude the chin (see the
# note above the crop code in process_frame for why that breaks pitch
# estimation). 0.3 = 30% padding on each side.
FACE_CROP_PAD_RATIO = 0.3

# MediaPipe FaceMesh (468-point landmark model) was previously running on
# EVERY frame with no throttling at all — unlike YuNet/YOLO above, which
# are both already throttled. FaceMesh is typically the single heaviest
# model of the three per-call, so leaving it unthrottled meant paying its
# full cost on every frame regardless of DETECT_EVERY_N_FRAMES, which
# shows up as general slowness/low FPS rather than periodic stutter.
# Kept lower than the detectors' N=8 since head-pose/gaze is the most
# latency-sensitive signal in this app (still a ~3x reduction in mesh
# inferences vs. before). Head-pose/gaze are recomputed every frame from
# the (possibly reused) landmarks, so attention feedback still updates
# every frame — only the landmark inference itself is throttled.
FACEMESH_EVERY_N_FRAMES = 8
DRAW_LANDMARKS = False       # draw the 468 FaceMesh dots on the frame (adds per-frame cost + visual clutter)

# Head-pose direction thresholds (degrees). This head-pose estimate uses
# only 6 sparse landmark points with no per-person calibration, so pitch
# in particular is noisy — a normal desk glance-down may not produce a
# large angle. If "Down" still doesn't trigger after lowering this, turn
# on the sidebar debug toggle in app.py, watch the raw pitch value while
# looking down, and set PITCH_DOWN_THRESHOLD just below what you see
# (also check the sign isn't flipped — if pitch goes very NEGATIVE when
# you look down instead of positive, swap the comparison signs below).
YAW_THRESHOLD = 18
PITCH_UP_THRESHOLD = -15
PITCH_DOWN_THRESHOLD = 10  # lowered from 18 — was likely too strict to ever trigger

# Also worth knowing: since yaw is checked before pitch, a head tilt that
# combines a downward glance with any sideways turn beyond YAW_THRESHOLD
# will be classified as Left/Right instead of Down.

# How long the neutral-pose calibration window lasts, in seconds, right
# after a face is first successfully tracked (or after reset_calibration()
# is called for a new session). During this window the candidate is
# assumed attentive (status "Calibrating", not "Distracted") while raw
# yaw/pitch samples are collected; the average becomes neutral_yaw /
# neutral_pitch, and every frame afterward is classified by *deviation*
# from that neutral pose instead of the raw angle. This is what makes the
# same YAW_THRESHOLD/PITCH_*_THRESHOLD values above work across different
# webcam heights, laptops, sitting posture, and users, instead of only
# being correct for whatever setup they were tuned on.
CALIBRATION_DURATION_SECONDS = 3.0
CALIBRATION_MIN_SAMPLES = 5  # guard against a too-short/low-FPS window producing a noisy average

# ------------------------------------------------------------------
# Why the box used to flip red/green almost randomly while the
# candidate was genuinely looking at the screen:
#
# 1. `looking_at_screen` required BOTH head-pose AND gaze to read
#    "Center" on the SAME frame (a strict AND of two independently
#    noisy signals). If either one has, say, a 75% chance of reading
#    correctly on a given frame, the combined AND only reads correctly
#    ~56% of the time — i.e. red *more often than not* even while
#    genuinely attentive. This is the main cause of "mostly red, only
#    occasionally green".
# 2. The gaze ratio band (0.35–0.65) was narrow relative to normal
#    frame-to-frame iris-landmark jitter, so a person looking straight
#    ahead would frequently read as "Left"/"Right" by a small margin.
# 3. Raw yaw/pitch/gaze values were used as-is with no smoothing, so a
#    single noisy landmark reading could flip the whole status.
#
# Fix: smooth yaw/pitch/gaze with an exponential moving average (damps
# single-frame noise), widen the gaze "center" band, and require a
# couple of consecutive contrary readings before the *displayed*
# status actually flips (debounce) — a single stray frame no longer
# toggles the box color.
# ------------------------------------------------------------------
POSE_EMA_ALPHA = 0.5   # smoothing for yaw/pitch (higher = more responsive, less smoothing)
GAZE_EMA_ALPHA = 0.4   # smoothing for the iris ratio (gaze is noisier than head pose, smoothed a bit more)

# Widened from (0.35, 0.65) — that band was tight enough that normal
# iris-landmark jitter while looking straight ahead routinely pushed the
# ratio just outside it, misreading "Center" as "Left"/"Right".
GAZE_RATIO_LEFT = 0.28
GAZE_RATIO_RIGHT = 0.72

# Number of consecutive frames of contrary evidence required before the
# displayed Focused/Distracted status (and box color) actually flips.
# Prevents a single noisy frame from toggling the box; status still
# reacts within a couple of evaluations (~1-2s given FACEMESH throttling).
ATTENTION_DEBOUNCE_FRAMES = 2

print("[INFO] Loading YuNet face detector...")
face_detector = cv2.FaceDetectorYN.create(
    # model=YUNET_MODEL_PATH,
    model=str(model_path),
    config="",
    input_size=(320, 320),  # reset per-frame based on actual frame size
    score_threshold=0.6,
    nms_threshold=0.3,
    top_k=5000
)
print("[INFO] YuNet loaded.")

print("[INFO] Loading MediaPipe FaceMesh...")
mp_face_mesh = mdp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    # Iris landmarks (indices 468-477, used by estimate_gaze_direction
    # via LEFT_IRIS/RIGHT_IRIS below) only exist when this is True. With
    # it False, iris_center() indexed past the end of the 468-point list,
    # threw IndexError on every call, and silently fell back to "Center"
    # every time via the try/except in estimate_gaze_direction — meaning
    # gaze estimation was a complete no-op, not actually checking
    # anything. Turning this on costs a bit more per FaceMesh call, but
    # that's already throttled by FACEMESH_EVERY_N_FRAMES.
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.7
)
print("[INFO] FaceMesh loaded.")

print("[INFO] Loading YOLO26n model for object detection...")
model = YOLO(str(yolo_model))  # yolov26n.pt is the smallest, fastest model; yolov26s.pt is a bit more accurate but slower

# Use GPU automatically if the machine has one available (this is by far
# the biggest lever for YOLO speed — CPU inference is the usual
# bottleneck). Falls back to CPU silently if no CUDA device is present.
try:
    import torch
    if torch.cuda.is_available():
        model.to("cuda")
        print("[INFO] YOLO running on GPU (CUDA).")
    else:
        print("[INFO] YOLO running on CPU (no CUDA device found).")
except ImportError:
    print("[INFO] YOLO running on CPU (torch not fully available).")

print("[INFO] YOLO26n model loaded.")

YOLO_INFERENCE_SIZE = 512  # smaller than the default 640 -> much faster, some accuracy trade-off

# 3D face model points used for solvePnP head-pose estimation
FACE_MODEL_3D = {
    1: (0.0, 0.0, 0.0),        # Nose tip
    33: (-30.0, -35.0, -30.0),  # Left eye outer
    263: (30.0, -35.0, -30.0),  # Right eye outer
    61: (-25.0, 30.0, -20.0),   # Left mouth
    291: (25.0, 30.0, -20.0),   # Right mouth
    199: (0.0, 65.0, -5.0)      # Chin
}

# Mediapipe FaceMesh iris landmark indices (only populated when refine_landmarks=True)
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE_CORNERS = (33, 133)
RIGHT_EYE_CORNERS = (362, 263)

# ------------------------------------------------------------
# Module-level state that must persist between process_frame() calls
# (this replaces the local variables that used to live inside the
# script's own `while True` loop).
# ------------------------------------------------------------
_state = {
    "frame_count": 0,
    "last_faces": [],
    "last_yolo_detections": [],  # cached YOLO results, refreshed every YOLO_EVERY_N_FRAMES
    "last_face_box": None,       # [x, y, x2, y2] of the last successfully detected face
    "last_face_seen_time": 0.0,
    "prev_time": time.time(),
    "last_mesh_detected": False,   # cached FaceMesh result, refreshed every FACEMESH_EVERY_N_FRAMES
    "last_face_landmarks": None,   # cached mediapipe landmark object, reused between mesh runs
    # Smoothing / debounce state (see the note above GAZE_RATIO_LEFT/RIGHT).
    "yaw_ema": None,
    "pitch_ema": None,
    "gaze_ratio_ema": None,
    "stable_looking_at_screen": False,  # the debounced value actually shown to the UI
    "contrary_streak": 0,               # consecutive frames disagreeing with stable_looking_at_screen

    # ---- Neutral-pose calibration state (see CALIBRATION_DURATION_SECONDS
    # below and reset_calibration()). yaw/pitch from solvePnP are never
    # trustworthy in an absolute sense — they depend on this specific
    # webcam's height/angle and this specific person's face geometry — so
    # every subsequent frame's yaw/pitch is measured as a DEVIATION from
    # whatever was recorded here, not against a fixed universal "0".
    "calibrated": False,
    "calibration_start_time": None,
    "calibration_yaw_samples": [],
    "calibration_pitch_samples": [],
    "neutral_yaw": 0.0,
    "neutral_pitch": 0.0,
}

# How long to keep trusting the last known face position after detection
# fails, for occlusion checks. A book/phone fully covering the face makes
# YuNet fail to find a face at all — the exact case object-on-face is
# supposed to catch — so we can't rely on a *current* face box in that
# situation. We use the last known position instead, for a short window.
FACE_MEMORY_SECONDS = 3.0


def reset_calibration():
    """
    Call this once when a new interview/session starts (and any time the
    candidate wants to redo calibration, e.g. after visibly repositioning
    the camera). Clears the stored neutral yaw/pitch and smoothing state
    so the next call(s) to process_frame() re-run the ~3s calibration
    window from scratch before resuming normal attention scoring.
    app.py should call this once per session, e.g. right before starting
    the "please look at the screen" countdown.
    """
    _state["calibrated"] = False
    _state["calibration_start_time"] = None
    _state["calibration_yaw_samples"] = []
    _state["calibration_pitch_samples"] = []
    _state["neutral_yaw"] = 0.0
    _state["neutral_pitch"] = 0.0
    _state["yaw_ema"] = None
    _state["pitch_ema"] = None
    _state["gaze_ratio_ema"] = None
    _state["stable_looking_at_screen"] = False
    _state["contrary_streak"] = 0


def _check_object_overlap(detected_objects, face_box):
    """Returns (object_on_face, object_on_eyes) by checking detected YOLO
    boxes (excluding 'person') against the face box and its upper half."""
    fx, fy, fx2, fy2 = face_box
    eyes_box = [fx, fy, fx2, fy + int((fy2 - fy) * 0.5)]

    on_face = False
    on_eyes = False

    for obj in detected_objects:
        if obj["class"] == "person":
            continue
        if boxes_overlap(obj["bbox"], face_box):
            on_face = True
        if boxes_overlap(obj["bbox"], eyes_box):
            on_eyes = True

    return on_face, on_eyes


def _empty_analysis():
    return {
        "timestamp": time.strftime("%H:%M:%S"),

        "face": {
            "detected": False,
            "mesh_detected": False,
            "landmarks": 0,
            "visibility": "Not Detected",
            "size": "Unknown"
        },

        "quality": {
            "lighting": None,
            "brightness": None,
            "blur": None,
        },

        "attention": {
            "status": "Unknown",
            "score": 0,
            "head_direction": "Center",
            "gaze_direction": "Center",
            "yaw": 0.0,
            "pitch": 0.0,
            "roll": 0.0,
            "looking_at_screen": False,
            "calibrating": False,       # True while the neutral-pose calibration window is running
            "calibration_progress": 0.0  # 0.0-1.0, so the UI can show a countdown/progress bar
        },

        "behavior": {
            "looking_left": False,
            "looking_right": False,
            "looking_down": False,
            "multiple_faces": False,
            "face_missing": True
        },

        "objects": {
            "detected": [],
            "phone_detected": False,
            "person_count": 0,
            "object_on_face": False,
            "object_on_eyes": False,
        },

        "system": {
            "fps": 0
        },
    }


def compute_gaze_ratio(face_landmarks):
    """
    Returns the raw (unsmoothed) horizontal iris ratio, averaged across
    both eyes — 0.0 means iris fully toward one side, 1.0 fully toward
    the other, ~0.5 is centered. Returns None if it can't be computed
    (e.g. iris landmarks unavailable) so the caller can fall back
    sensibly instead of silently treating a failure as "Center" forever.
    """
    try:
        def iris_center_x(indices):
            xs = [face_landmarks.landmark[i].x for i in indices]
            return float(np.mean(xs))

        l_iris_x = iris_center_x(LEFT_IRIS)
        r_iris_x = iris_center_x(RIGHT_IRIS)

        l_corner_a = face_landmarks.landmark[LEFT_EYE_CORNERS[0]].x
        l_corner_b = face_landmarks.landmark[LEFT_EYE_CORNERS[1]].x
        r_corner_a = face_landmarks.landmark[RIGHT_EYE_CORNERS[0]].x
        r_corner_b = face_landmarks.landmark[RIGHT_EYE_CORNERS[1]].x

        l_ratio = (l_iris_x - min(l_corner_a, l_corner_b)) / (abs(l_corner_a - l_corner_b) + 1e-6)
        r_ratio = (r_iris_x - min(r_corner_a, r_corner_b)) / (abs(r_corner_a - r_corner_b) + 1e-6)
        return (l_ratio + r_ratio) / 2
    except Exception:
        return None


def classify_gaze_ratio(ratio):
    """Classify an (ideally already-smoothed) iris ratio into Left/Center/Right."""
    if ratio is None:
        return "Center"
    if ratio < GAZE_RATIO_LEFT:
        return "Right"
    elif ratio > GAZE_RATIO_RIGHT:
        return "Left"
    return "Center"


def estimate_gaze_direction(face_landmarks, img_w, img_h):
    """
    Rough gaze estimate from iris position relative to eye-corner landmarks.
    This is independent of head pose, so it can catch a person moving only
    their eyes (not their head) off-screen.

    NOTE: this single-frame, unsmoothed version is kept for any external
    callers, but process_frame() below uses compute_gaze_ratio() +
    classify_gaze_ratio() directly so it can smooth the ratio across
    frames (see GAZE_EMA_ALPHA) before classifying it.
    """
    return classify_gaze_ratio(compute_gaze_ratio(face_landmarks))


def process_frame(frame):
    """
    Runs one frame through face detection, face mesh, head-pose/gaze
    estimation and YOLO object detection, and returns:
        (annotated_frame, analysis_dict)
        (annotated_frame, analysis_dict)
    This is the function app.py calls once per webcam frame.
    """
    analysis = _empty_analysis()
    h, w = frame.shape[:2]

    # ---------------- YOLO object detection (throttled) ----------------
    if _state["frame_count"] % YOLO_EVERY_N_FRAMES == 0:
        yolo_results = model(frame, imgsz=YOLO_INFERENCE_SIZE, verbose=False)

        detections = []
        phone_detected = False
        person_count = 0

        for result in yolo_results:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                name = model.names[cls]
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                detections.append({
                    "class": name,
                    "confidence": round(conf, 2),
                    "bbox": [x1, y1, x2, y2]
                })

                if name == "cell phone":
                    phone_detected = True
                if name == "person":
                    person_count += 1

        _state["last_yolo_detections"] = {
            "detected": detections,
            "phone_detected": phone_detected,
            "person_count": person_count,
        }

    cached = _state["last_yolo_detections"] or {"detected": [], "phone_detected": False, "person_count": 0}
    analysis["objects"]["detected"] = cached["detected"]
    analysis["objects"]["phone_detected"] = cached["phone_detected"]
    analysis["objects"]["person_count"] = cached["person_count"]

    analysis["behavior"]["multiple_faces"] = analysis["objects"]["person_count"] > 1

    # ---------------- YuNet face detection (throttled) ----------------
    if _state["frame_count"] % DETECT_EVERY_N_FRAMES == 0:
        face_detector.setInputSize((w, h))
        _, faces = face_detector.detect(frame)
        _state["last_faces"] = faces if faces is not None else []
    _state["frame_count"] += 1

    last_faces = _state["last_faces"]

    # "Someone else in frame" signal — checked here (as soon as we know
    # it) so every return path below can use it. Two independent signals:
    # YuNet finding more than one face, or YOLO counting more than one
    # person (catches someone visible behind/beside the candidate even
    # if their face isn't frontal enough for YuNet to pick up).
    intrusion_detected = (len(last_faces) > 1) or (analysis["objects"]["person_count"] > 1)

    if len(last_faces) == 0:
        analysis["behavior"]["face_missing"] = True

        # No face detected right now — likely occluded rather than absent
        # if we saw one recently. Check the cached position for occlusion.
        if _state["last_face_box"] is not None and \
                (time.time() - _state["last_face_seen_time"]) <= FACE_MEMORY_SECONDS:
            on_face, on_eyes = _check_object_overlap(
                analysis["objects"]["detected"], _state["last_face_box"]
            )
            analysis["objects"]["object_on_face"] = on_face
            analysis["objects"]["object_on_eyes"] = on_eyes

        current_time = time.time()
        fps = 1 / max(current_time - _state["prev_time"], 1e-6)
        _state["prev_time"] = current_time
        analysis["system"]["fps"] = round(fps, 2)
        return frame, analysis

    # NOTE: FaceMesh is configured for max_num_faces=1, so only the first
    # detected face is analyzed in depth. person_count from YOLO is used
    # above as the multi-person signal.
    face = last_faces[0]
    x, y, fw, fh = face[:4].astype(int)

    x, y = max(0, x), max(0, y)
    x2, y2 = min(w, x + fw), min(h, y + fh)

    # ---- Padded crop for FaceMesh/head-pose only ----
    # YuNet's raw box is often tight — sometimes stopping around the
    # nose/mouth with little to no chin included (exactly what's visible
    # in the reported screenshot: box top at the hairline, bottom well
    # above the chin). Feeding that tight crop straight into FaceMesh
    # means landmarks like the chin (idx 199, used by solvePnP) get
    # placed near the bottom edge of a crop that doesn't actually contain
    # the chin — which solvePnP then reads as the head being tilted
    # forward/down, even when the person is looking straight at the
    # camera. Padding the crop before running FaceMesh gives it the full
    # face to work with, so the landmarks (and the pose computed from
    # them) reflect reality instead of the detector's tight box.
    # The rest of this function (visibility/size checks, the drawn box,
    # occlusion memory) intentionally keeps using the RAW x/y/x2/y2/fw/fh
    # above — only the crop fed to FaceMesh is padded.
    pad_w = int(fw * FACE_CROP_PAD_RATIO)
    pad_h = int(fh * FACE_CROP_PAD_RATIO)
    mesh_x = max(0, x - pad_w)
    mesh_y = max(0, y - pad_h)
    mesh_x2 = min(w, x2 + pad_w)
    mesh_y2 = min(h, y2 + pad_h)

    face_x, face_y = mesh_x, mesh_y  # padded origin, used later to offset landmark coords back to the full frame

    face_crop = frame[mesh_y:mesh_y2, mesh_x:mesh_x2]

    if face_crop.size == 0:
        analysis["behavior"]["face_missing"] = True

        if _state["last_face_box"] is not None and \
                (time.time() - _state["last_face_seen_time"]) <= FACE_MEMORY_SECONDS:
            on_face, on_eyes = _check_object_overlap(
                analysis["objects"]["detected"], _state["last_face_box"]
            )
            analysis["objects"]["object_on_face"] = on_face
            analysis["objects"]["object_on_eyes"] = on_eyes

        draw_face_boxes(frame, last_faces, primary_ok=False)

        current_time = time.time()
        fps = 1 / max(current_time - _state["prev_time"], 1e-6)
        _state["prev_time"] = current_time
        analysis["system"]["fps"] = round(fps, 2)
        return frame, analysis

    analysis["behavior"]["face_missing"] = False

    # Remember this face's position for occlusion checks on future frames
    # where detection fails (e.g. face fully covered by a book/phone).
    _state["last_face_box"] = [x, y, x2, y2]
    _state["last_face_seen_time"] = time.time()

    # ---------------- Face quality ----------------
    face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)  # shared by both checks below, computed once
    lighting, brightness = check_brightness(face_gray)
    blur_status, blur_score = check_blur(face_gray)
    face_size = check_face_size(fw, fh, frame)
    visibility = check_visibility(x, y, x2, y2, frame)

    analysis["quality"].update({
        "lighting": lighting,
        "brightness": round(float(brightness), 1),
        "blur": blur_status,
        "blur_score": round(float(blur_score), 1)
    })

    analysis["face"].update({
        "detected": True,
        "visibility": visibility,
        "size": face_size
    })

    # ---------------- object-on-face / object-on-eyes heuristic ----------------
    on_face, on_eyes = _check_object_overlap(analysis["objects"]["detected"], [x, y, x2, y2])
    analysis["objects"]["object_on_face"] = on_face
    analysis["objects"]["object_on_eyes"] = on_eyes

    # ---------------- MediaPipe FaceMesh (throttled) ----------------
    run_mesh = (_state["frame_count"] % FACEMESH_EVERY_N_FRAMES == 0) or (_state["last_face_landmarks"] is None)

    if run_mesh:
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(face_rgb)
        mesh_detected = bool(results.multi_face_landmarks)
        face_landmarks = results.multi_face_landmarks[0] if mesh_detected else None
        _state["last_mesh_detected"] = mesh_detected
        _state["last_face_landmarks"] = face_landmarks
    else:
        mesh_detected = _state["last_mesh_detected"]
        face_landmarks = _state["last_face_landmarks"]

    if not mesh_detected or face_landmarks is None:
        analysis["face"]["mesh_detected"] = False
        draw_face_boxes(frame, last_faces, primary_ok=False)
        current_time = time.time()
        fps = 1 / max(current_time - _state["prev_time"], 1e-6)
        _state["prev_time"] = current_time
        analysis["system"]["fps"] = round(fps, 2)
        return frame, analysis

    img_h, img_w = face_crop.shape[:2]  # still needed below, just for landmark normalization

    analysis["face"]["mesh_detected"] = True
    analysis["face"]["landmarks"] = len(face_landmarks.landmark)

    # ---------------- Head pose (solvePnP) ----------------
    # IMPORTANT: landmarks come out of FaceMesh normalized to the CROP
    # (0..1 across face_crop's own width/height), but solvePnP needs 2D
    # points and camera intrinsics expressed in the SAME coordinate
    # system. Using the crop's own width as "focal length" (previous
    # behavior) effectively pretends the tightly-cropped face image is
    # the entire camera frame from a very wide lens — the assumed focal
    # length then shrinks/grows with however tight YuNet's bbox happens
    # to be, which wildly amplifies any small head movement or landmark
    # jitter into large yaw/pitch swings. Converting the points to
    # full-frame pixel coordinates (adding back the crop's origin,
    # face_x/face_y — same offset already used for landmark drawing
    # below) and building the camera matrix from the FULL frame's
    # dimensions instead gives solvePnP a stable, physically-plausible
    # camera model that doesn't depend on crop size or distance to camera.
    face_2d, face_3d = [], []
    for idx, model_pt in FACE_MODEL_3D.items():
        lm = face_landmarks.landmark[idx]
        face_2d.append([lm.x * img_w + face_x, lm.y * img_h + face_y])
        face_3d.append(model_pt)

    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)

    focal_length = w  # full FRAME width, not the face crop's width
    cam_matrix = np.array([
        [focal_length, 0, w / 2],
        [0, focal_length, h / 2],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    success, rot_vec, trans_vec = cv2.solvePnP(
        face_3d, face_2d, cam_matrix, dist_matrix,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    direction = "Unknown"

    if success:
        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        raw_pitch, raw_yaw, raw_roll = float(angles[0]), float(angles[1]), float(angles[2])

        # ================================================================
        # STEP 1 — Neutral-pose calibration (runs once per session).
        #
        # raw_yaw/raw_pitch out of solvePnP are NOT comparable to a fixed
        # "0 = straight at the screen" for everyone: they shift with this
        # webcam's height/angle and this person's face geometry (see the
        # note above CALIBRATION_DURATION_SECONDS). So instead of
        # classifying the raw angle, we spend the first few seconds of
        # tracking recording what "looking at the screen" reads as FOR
        # THIS person/camera, then classify every later frame by its
        # DEVIATION from that baseline.
        # ================================================================
        if not _state["calibrated"]:
            if _state["calibration_start_time"] is None:
                _state["calibration_start_time"] = time.time()

            _state["calibration_yaw_samples"].append(raw_yaw)
            _state["calibration_pitch_samples"].append(raw_pitch)

            elapsed = time.time() - _state["calibration_start_time"]
            analysis["attention"]["calibrating"] = True
            analysis["attention"]["calibration_progress"] = round(min(1.0, elapsed / CALIBRATION_DURATION_SECONDS), 2)
            analysis["attention"]["yaw"] = round(raw_yaw, 2)
            analysis["attention"]["pitch"] = round(raw_pitch, 2)
            analysis["attention"]["roll"] = round(raw_roll, 2)

            if elapsed >= CALIBRATION_DURATION_SECONDS and \
                    len(_state["calibration_yaw_samples"]) >= CALIBRATION_MIN_SAMPLES:
                _state["neutral_yaw"] = float(np.mean(_state["calibration_yaw_samples"]))
                _state["neutral_pitch"] = float(np.mean(_state["calibration_pitch_samples"]))
                _state["calibrated"] = True

            # Don't penalize the candidate for a pose we have no baseline
            # for yet — treat the calibration window itself as attentive.
            analysis["attention"]["status"] = "Calibrating"
            analysis["attention"]["head_direction"] = "Center"
            analysis["attention"]["looking_at_screen"] = True
            analysis["attention"]["score"] = 100

            primary_ok = not intrusion_detected
            draw_face_boxes(frame, last_faces, primary_ok=primary_ok)

            current_time = time.time()
            fps = 1 / max(current_time - _state["prev_time"], 1e-6)
            _state["prev_time"] = current_time
            analysis["system"]["fps"] = round(fps, 2)
            return frame, analysis

        # ---- Smooth yaw/pitch with an EMA so one noisy solvePnP result
        # (only 6 sparse points) doesn't flip the direction by itself. ----
        prev_yaw = _state["yaw_ema"]
        prev_pitch = _state["pitch_ema"]
        yaw = raw_yaw if prev_yaw is None else (POSE_EMA_ALPHA * raw_yaw + (1 - POSE_EMA_ALPHA) * prev_yaw)
        pitch = raw_pitch if prev_pitch is None else (POSE_EMA_ALPHA * raw_pitch + (1 - POSE_EMA_ALPHA) * prev_pitch)
        _state["yaw_ema"] = yaw
        _state["pitch_ema"] = pitch

        # ================================================================
        # STEP 2 — Classify by DEVIATION from this session's neutral pose,
        # not the raw angle. This is the actual fix for "always reads as
        # Distracted": previously YAW_THRESHOLD/PITCH_*_THRESHOLD were
        # compared against the raw solvePnP angle, which for a typical
        # laptop webcam (mounted above eye level) can sit well past those
        # thresholds even while genuinely looking at the screen. Comparing
        # the deviation instead cancels out that per-camera/per-person
        # offset, so the same threshold values now generalize.
        # ================================================================
        yaw_dev = yaw - _state["neutral_yaw"]
        pitch_dev = pitch - _state["neutral_pitch"]

        analysis["attention"]["yaw"] = round(yaw_dev, 2)
        analysis["attention"]["pitch"] = round(pitch_dev, 2)
        analysis["attention"]["roll"] = round(raw_roll, 2)

        if yaw_dev < -YAW_THRESHOLD:
            direction = "Left"
        elif yaw_dev > YAW_THRESHOLD:
            direction = "Right"
        elif pitch_dev < PITCH_UP_THRESHOLD:
            direction = "Up"
        elif pitch_dev > PITCH_DOWN_THRESHOLD:
            direction = "Down"
        else:
            direction = "Center"

        analysis["attention"]["head_direction"] = direction
        analysis["behavior"]["looking_left"] = (direction == "Left")
        analysis["behavior"]["looking_right"] = (direction == "Right")
        analysis["behavior"]["looking_down"] = (direction == "Down")

        # ---- Gaze: smooth the raw iris ratio with an EMA (gaze is
        # noisier than head pose, hence the lower alpha) before
        # classifying it, instead of classifying a single raw reading. ----
        raw_ratio = compute_gaze_ratio(face_landmarks)
        prev_ratio = _state["gaze_ratio_ema"]
        if raw_ratio is None:
            smoothed_ratio = prev_ratio  # keep last known value; None -> classify_gaze_ratio treats as Center
        else:
            smoothed_ratio = raw_ratio if prev_ratio is None else (
                GAZE_EMA_ALPHA * raw_ratio + (1 - GAZE_EMA_ALPHA) * prev_ratio
            )
        _state["gaze_ratio_ema"] = smoothed_ratio

        gaze = classify_gaze_ratio(smoothed_ratio)
        analysis["attention"]["gaze_direction"] = gaze

        # ================================================================
        # STEP 3 — Combine gaze (primary) with head pose (secondary,
        # override-only). Gaze alone decides Left/Right eye movement;
        # head pose only pulls the verdict to "not looking at screen" when
        # it has swung past a LARGE calibrated deviation (Up/Down/Left/
        # Right — direction is only ever non-Center once yaw_dev/pitch_dev
        # clear the threshold above, which after calibration corresponds
        # to a genuinely large turn, not sensor noise). Before calibration
        # this same combination amplified noise (see the removed comment
        # this replaced); now that both inputs are correctly centered on
        # this person's neutral pose, requiring both to read "on-screen"
        # is safe rather than punitive.
        # ================================================================
        raw_looking_at_screen = (gaze == "Center") and (direction == "Center")

        stable = _state["stable_looking_at_screen"]
        if raw_looking_at_screen == stable:
            _state["contrary_streak"] = 0
        else:
            _state["contrary_streak"] += 1
            if _state["contrary_streak"] >= ATTENTION_DEBOUNCE_FRAMES:
                stable = raw_looking_at_screen
                _state["contrary_streak"] = 0
        _state["stable_looking_at_screen"] = stable

        looking_at_screen = stable
        analysis["attention"]["looking_at_screen"] = looking_at_screen

        if looking_at_screen:
            analysis["attention"]["score"] = 100
            analysis["attention"]["status"] = "Focused"
        else:
            analysis["attention"]["score"] = 40
            analysis["attention"]["status"] = "Distracted"

        # ---- Hard overrides — take priority over the gaze/head verdict
        # above regardless of what it says. (face_missing and multiple
        # faces are already handled by earlier early-returns in this
        # function; this covers phone/object-on-eyes, the two overrides
        # that can only be known this late once object-overlap has run.) ----
        if analysis["objects"]["phone_detected"] or analysis["objects"]["object_on_eyes"]:
            analysis["attention"]["status"] = "Distracted"
            analysis["attention"]["looking_at_screen"] = False
            analysis["attention"]["score"] = 0
            # Keep the debounce state in sync with this override too, so a
            # phone/object being removed doesn't instantly flip back to
            # green off a single frame either.
            _state["stable_looking_at_screen"] = False

    # ---------------- Draw the candidate's box ----------------
    # Green only when actually looking at the screen AND nobody else is
    # in frame. Red for distraction, phone/object override, OR intrusion
    # (someone else visible) — any of those alone is enough for red.
    primary_ok = analysis["attention"]["looking_at_screen"] and not intrusion_detected
    draw_face_boxes(frame, last_faces, primary_ok=primary_ok)

    # ---------------- Draw landmarks on the full frame (optional) ----------------
    if DRAW_LANDMARKS:
        for landmark in face_landmarks.landmark:
            px = int(landmark.x * img_w) + face_x
            py = int(landmark.y * img_h) + face_y
            cv2.circle(frame, (px, py), 1, (0, 255, 0), -1)

    current_time = time.time()
    fps = 1 / max(current_time - _state["prev_time"], 1e-6)
    _state["prev_time"] = current_time
    analysis["system"]["fps"] = round(fps, 2)

    return frame, analysis