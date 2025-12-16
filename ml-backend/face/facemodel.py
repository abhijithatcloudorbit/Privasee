import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

# -----------------------------
# LOGGING
# -----------------------------
log = logging.getLogger("FaceModel")

# -----------------------------
# MODEL LOADING (ONCE)
# -----------------------------
use_yolo = False
yolo_model = None

try:
    from ultralytics import YOLO
    model_path = Path(__file__).parent / "yolov8n-face.pt"

    if model_path.exists():
        yolo_model = YOLO(str(model_path))
        use_yolo = True
        log.info("YOLOv8 face model loaded")
except Exception as e:
    log.warning(f"YOLO not available, falling back to Haar: {e}")

# Haar fallback
haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_path)

# -----------------------------
# FACE DETECTION
# -----------------------------
def detect_faces(img):
    """
    Returns list of bounding boxes as (x1, y1, x2, y2)
    """
    if use_yolo:
        results = yolo_model.predict(img, conf=0.25, verbose=False)
        boxes = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append((x1, y1, x2, y2))
        return boxes

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    return [(x, y, x + w, y + h) for (x, y, w, h) in rects]

# -----------------------------
# STRONG PRIVACY BLUR
# -----------------------------
def apply_strong_blur(image, x1, y1, x2, y2):
    h_img, w_img = image.shape[:2]

    # Expand bounding box (25%)
    w = x2 - x1
    h = y2 - y1
    pad = int(0.25 * max(w, h))

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w_img, x2 + pad)
    y2 = min(h_img, y2 + pad)

    roi = image[y1:y2, x1:x2]

    if roi.size == 0:
        return image

    # Strong blur (privacy-grade)
    roi = cv2.GaussianBlur(roi, (151, 151), 60)

    image[y1:y2, x1:x2] = roi
    return image

# -----------------------------
# PUBLIC API FUNCTION
# -----------------------------
def blur_faces_from_bytes(image_bytes):
    # Decode image ONCE
    img_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Image decode failed")

    # Detect faces
    boxes = detect_faces(image)
    log.info(f"Faces detected: {len(boxes)}")

    # Apply blur
    for (x1, y1, x2, y2) in boxes:
        image = apply_strong_blur(image, x1, y1, x2, y2)

    # Save output
    out_path = f"outputs/face/face_{datetime.now().timestamp()}.jpg"
    cv2.imwrite(out_path, image)

    # Encode and return bytes
    success, encoded = cv2.imencode(".jpg", image)
    if not success:
        raise RuntimeError("Failed to encode image")

    return encoded.tobytes()
