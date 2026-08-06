"""
train_model.py
----------------
Face Detection System backend module supporting:
1. Deep Learning Face Detection via OpenCV YuNet (cv2.FaceDetectorYN) - High accuracy, recommended
2. Classic OpenCV Haar Cascade Classifier (haarcascade_frontalface_default.xml) with tunable parameters

Functions provided:
    - detect_faces(image, ...): unified detection entry point
    - detect_faces_yunet(image, ...): YuNet ONNX deep learning detector
    - detect_faces_haar(image, ...): Haar Cascade detector with optional CLAHE
    - load_face_cascade(): loads XML Haar Cascade
    - load_yunet_model(): loads ONNX YuNet detector
    - save_uploaded_image(image, filename): save original image to uploads/
    - save_output_image(image, filename): save processed image to outputs/
"""

import os
import sys
import urllib.request
from pathlib import Path
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

MODELS_DIR = BASE_DIR / "models"
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Model filenames
DEFAULT_CASCADE_FILENAME = "haarcascade_frontalface_default.xml"
DEFAULT_YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"

CASCADE_ENV_VAR = "HAAR_CASCADE_PATH"
YUNET_ENV_VAR = "YUNET_MODEL_PATH"

LOCAL_CASCADE_PATH = MODELS_DIR / DEFAULT_CASCADE_FILENAME
LOCAL_YUNET_PATH = MODELS_DIR / DEFAULT_YUNET_FILENAME

YUNET_DOWNLOAD_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

CASCADE_DOWNLOAD_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/"
    "data/haarcascades/haarcascade_frontalface_default.xml"
)

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. LOAD HAAR CASCADE MODEL
# ---------------------------------------------------------------------------
def load_face_cascade(cascade_file_path=None):
    """
    Load the pre-trained Haar Cascade face detector.
    """
    if not hasattr(cv2, "CascadeClassifier"):
        print("[ERROR] OpenCV installation does not expose CascadeClassifier.", flush=True)
        return None

    def try_load(path):
        if not os.path.exists(path):
            return None
        cascade = cv2.CascadeClassifier(str(path))
        return cascade if not cascade.empty() else None

    def download_cascade(destination_path):
        try:
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            urllib.request.urlretrieve(CASCADE_DOWNLOAD_URL, destination_path)
            return True
        except Exception as exc:
            print(f"[ERROR] Failed to download Haar Cascade model: {exc}", flush=True)
            return False

    def candidate_paths():
        candidates = []
        if cascade_file_path:
            candidates.append(str(cascade_file_path))

        env_path = os.getenv(CASCADE_ENV_VAR)
        if env_path:
            candidates.append(str(env_path))

        candidates.extend([
            str(LOCAL_CASCADE_PATH),
            str(BASE_DIR / DEFAULT_CASCADE_FILENAME),
            str(BASE_DIR / "models" / DEFAULT_CASCADE_FILENAME),
            str(Path.cwd() / "models" / DEFAULT_CASCADE_FILENAME),
        ])

        if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
            builtin_path = os.path.join(cv2.data.haarcascades, DEFAULT_CASCADE_FILENAME)
            if builtin_path not in candidates:
                candidates.append(builtin_path)

        seen = set()
        for path in candidates:
            if path not in seen:
                seen.add(path)
                yield path

    face_cascade = None
    cascade_path = None

    for path in candidate_paths():
        if os.path.exists(path):
            face_cascade = try_load(path)
            if face_cascade is not None:
                cascade_path = path
                break

    if face_cascade is None:
        print("[INFO] Attempting to download Haar Cascade model...")
        if download_cascade(str(LOCAL_CASCADE_PATH)):
            face_cascade = try_load(str(LOCAL_CASCADE_PATH))
            cascade_path = str(LOCAL_CASCADE_PATH)

    if face_cascade is None:
        print("[ERROR] Could not load Haar Cascade XML file.", flush=True)
        return None

    print(f"[INFO] Haar Cascade loaded from: {cascade_path}", flush=True)
    return face_cascade


FACE_CASCADE = None
_FACE_CASCADE_LOADED = False


def get_face_cascade():
    """Lazily load and cache the Haar Cascade classifier."""
    global FACE_CASCADE, _FACE_CASCADE_LOADED
    if not _FACE_CASCADE_LOADED:
        FACE_CASCADE = load_face_cascade()
        _FACE_CASCADE_LOADED = True
    return FACE_CASCADE


# ---------------------------------------------------------------------------
# 2. LOAD YUNET DEEP LEARNING MODEL
# ---------------------------------------------------------------------------
def get_yunet_model_path(custom_path=None):
    """Resolve absolute path to the YuNet ONNX model file."""
    candidates = []
    if custom_path:
        candidates.append(str(custom_path))
    env_path = os.getenv(YUNET_ENV_VAR)
    if env_path:
        candidates.append(str(env_path))

    candidates.extend([
        str(LOCAL_YUNET_PATH),
        str(BASE_DIR / DEFAULT_YUNET_FILENAME),
        str(BASE_DIR / "models" / DEFAULT_YUNET_FILENAME),
        str(Path.cwd() / "models" / DEFAULT_YUNET_FILENAME),
    ])

    for path in candidates:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return path

    # Attempt download if missing
    print("[INFO] YuNet model not found locally. Downloading from official repository...", flush=True)
    try:
        urllib.request.urlretrieve(YUNET_DOWNLOAD_URL, str(LOCAL_YUNET_PATH))
        if os.path.exists(LOCAL_YUNET_PATH) and os.path.getsize(LOCAL_YUNET_PATH) > 0:
            print(f"[INFO] YuNet model downloaded successfully to {LOCAL_YUNET_PATH}", flush=True)
            return str(LOCAL_YUNET_PATH)
    except Exception as exc:
        print(f"[ERROR] Failed to download YuNet model: {exc}", flush=True)

    return None


def get_yunet_detector(image_width, image_height, score_threshold=0.65, nms_threshold=0.3, custom_path=None):
    """
    Create an instance of cv2.FaceDetectorYN for specified image dimensions.
    """
    if not hasattr(cv2, "FaceDetectorYN"):
        print("[ERROR] OpenCV installation does not support FaceDetectorYN.", flush=True)
        return None

    model_path = get_yunet_model_path(custom_path)
    if not model_path:
        print("[ERROR] YuNet ONNX model path could not be resolved.", flush=True)
        return None

    try:
        detector = cv2.FaceDetectorYN.create(
            model=model_path,
            config="",
            input_size=(int(image_width), int(image_height)),
            score_threshold=float(score_threshold),
            nms_threshold=float(nms_threshold),
            top_k=5000
        )
        return detector
    except Exception as exc:
        print(f"[ERROR] Failed to create YuNet FaceDetectorYN: {exc}", flush=True)
        return None


# ---------------------------------------------------------------------------
# 3. DETECTION ENGINES (YUNET & HAAR CASCADE)
# ---------------------------------------------------------------------------
def detect_faces_yunet(image, confidence_threshold=0.65, nms_threshold=0.3, draw_landmarks=False):
    """
    Detect faces in BGR image using OpenCV YuNet Deep Learning model.

    Returns:
        tuple: (processed_image, face_count, face_details_list)
    """
    if image is None:
        return image, 0, []

    processed_image = image.copy()
    h, w = processed_image.shape[:2]

    detector = get_yunet_detector(w, h, score_threshold=confidence_threshold, nms_threshold=nms_threshold)
    if detector is None:
        print("[WARN] YuNet detector creation failed. Falling back...", flush=True)
        return None, 0, []

    results = detector.detect(processed_image)
    faces = results[1]

    if faces is None or len(faces) == 0:
        return processed_image, 0, []

    face_details = []
    face_count = len(faces)

    for i, face in enumerate(faces):
        # face format: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rc, y_rc, x_lc, y_lc, score]
        box = face[:4].astype(int)
        score = float(face[14])
        x, y, bw, bh = box

        # Ensure box coordinates are within bounds
        x = max(0, x)
        y = max(0, y)
        bw = min(w - x, bw)
        bh = min(h - y, bh)

        face_details.append({
            "box": (x, y, bw, bh),
            "score": score
        })

        # Draw bounding box (Blue)
        cv2.rectangle(processed_image, (x, y), (x + bw, y + bh), (255, 0, 0), 2)

        # Draw confidence badge above bounding box
        label = f"{score:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        label_y = max(y - 5, label_h + 5)
        cv2.rectangle(
            processed_image,
            (x, label_y - label_h - 2),
            (x + label_w + 4, label_y + baseline),
            (255, 0, 0),
            -1
        )
        cv2.putText(
            processed_image,
            label,
            (x + 2, label_y - 1),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

        # Optional: draw facial landmarks (eyes, nose, mouth)
        if draw_landmarks and len(face) >= 14:
            landmarks = face[4:14].reshape((5, 2)).astype(int)
            colors = [(0, 255, 255), (0, 255, 255), (0, 0, 255), (0, 255, 0), (0, 255, 0)]
            for pt, color in zip(landmarks, colors):
                cv2.circle(processed_image, (pt[0], pt[1]), 2, color, -1)

    return processed_image, face_count, face_details


def detect_faces_haar(image, scale_factor=1.1, min_neighbors=5, min_size=(30, 30), use_clahe=False):
    """
    Detect faces using OpenCV Haar Cascade classifier with optional CLAHE contrast enhancement.

    Returns:
        tuple: (processed_image, face_count, face_details_list)
    """
    if image is None:
        return image, 0, []

    cascade = get_face_cascade()
    if cascade is None:
        return image, 0, []

    processed_image = image.copy()
    gray_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_image = clahe.apply(gray_image)

    faces = cascade.detectMultiScale(
        gray_image,
        scaleFactor=float(scale_factor),
        minNeighbors=int(min_neighbors),
        minSize=min_size
    )

    face_details = []
    face_count = len(faces)

    for (x, y, w, h) in faces:
        face_details.append({"box": (x, y, w, h), "score": 1.0})
        cv2.rectangle(processed_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return processed_image, face_count, face_details


# ---------------------------------------------------------------------------
# 4. UNIFIED CORE FACE DETECTION FUNCTION
# ---------------------------------------------------------------------------
def detect_faces(
    image,
    model_type="yunet",
    confidence_threshold=0.65,
    scale_factor=1.1,
    min_neighbors=5,
    use_clahe=False,
    min_size=(30, 30),
    **kwargs
):
    """
    Unified entry point for face detection supporting both YuNet and Haar Cascade.

    Args:
        image (numpy.ndarray): Input BGR image.
        model_type (str): 'yunet' for Deep Learning YuNet, or 'haar' for Haar Cascade.
        confidence_threshold (float): Confidence score threshold for YuNet (0.1 to 1.0).
        scale_factor (float): Haar Cascade parameter scaleFactor (e.g. 1.05 to 1.3).
        min_neighbors (int): Haar Cascade parameter minNeighbors (e.g. 2 to 10).
        use_clahe (bool): Whether to apply CLAHE histogram equalization for Haar Cascade.
        min_size (tuple): Minimum face box size (width, height).
        **kwargs: Absorbs any extra keyword arguments safely.

    Returns:
        tuple: (processed_image, face_count)
    """
    if image is None:
        return image, 0

    if str(model_type).lower() == "yunet":
        processed_img, count, _ = detect_faces_yunet(image, confidence_threshold=confidence_threshold)
        if processed_img is not None:
            return processed_img, count
        print("[WARN] YuNet execution failed. Falling back to Haar Cascade...", flush=True)

    # Default / Fallback: Haar Cascade
    processed_img, count, _ = detect_faces_haar(
        image,
        scale_factor=scale_factor,
        min_neighbors=min_neighbors,
        min_size=min_size,
        use_clahe=use_clahe
    )
    return processed_img, count


# ---------------------------------------------------------------------------
# 5. IMAGE-BASED FACE DETECTION & FILE HELPERS
# ---------------------------------------------------------------------------
def detect_from_image(image_path, model_type="yunet", confidence_threshold=0.65):
    """Load an image file from disk and run face detection on it."""
    if not os.path.exists(image_path):
        print(f"[ERROR] Image file not found: {image_path}")
        return None, 0

    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return None, 0

    return detect_faces(image, model_type=model_type, confidence_threshold=confidence_threshold)


def save_uploaded_image(image, filename):
    """Save original uploaded image to uploads/ folder."""
    if image is None:
        return None
    save_path = os.path.join(UPLOADS_DIR, filename)
    cv2.imwrite(save_path, image)
    return save_path


def save_output_image(image, filename):
    """Save processed image with bounding boxes to outputs/ folder."""
    if image is None:
        return None
    save_path = os.path.join(OUTPUTS_DIR, filename)
    cv2.imwrite(save_path, image)
    return save_path
