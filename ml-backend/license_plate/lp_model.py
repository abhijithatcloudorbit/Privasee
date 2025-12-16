import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from datetime import datetime
import os

# -----------------------------
# LOAD MODEL ONCE
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "license_plate_detector.pt"

model = YOLO(str(MODEL_PATH))

# Ensure output folder exists
os.makedirs("outputs/lp", exist_ok=True)

# -----------------------------
# PREPROCESSING
# -----------------------------
def preprocess_image(img):
    h, w = img.shape[:2]

    if max(h, w) < 720:
        scale = 720 / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return img

# -----------------------------
# BLUR LOGIC
# -----------------------------
def blur_region(image, x1, y1, x2, y2):
    roi = image[y1:y2, x1:x2]

    if roi.size == 0:
        return image

    k = max(31, (abs(x2 - x1) // 3) | 1)
    image[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)

    return image

# -----------------------------
# PUBLIC API FUNCTION
# -----------------------------
def blur_lp_from_bytes(image_bytes: bytes) -> bytes:
    npimg = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Invalid image")

    img = preprocess_image(img)
    results = model(img)

    for r in results:
        for box in r.boxes:
            if float(box.conf[0]) < 0.40:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)

            img = blur_region(img, x1, y1, x2, y2)

    # ✅ SAVE OUTPUT (THIS WAS MISSING)
    out_path = f"outputs/lp/lp_{datetime.now().timestamp()}.jpg"
    cv2.imwrite(out_path, img)

    # ✅ RETURN BYTES FOR API
    success, buffer = cv2.imencode(".jpg", img)
    if not success:
        raise RuntimeError("Failed to encode LP image")

    return buffer.tobytes()
