# the code which can be used for integration
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from pathlib import Path
import logging
import sys

app = FastAPI(title="Face Blur API")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("FaceBlurAPI")

#LOAD YOLO DETECTOR
use_yolo = False
yolo_model = None

try:
    from ultralytics import YOLO

    if Path("yolov8n-face.pt").exists():
        log.info("Loading YOLOv8n-face...")
        yolo_model = YOLO("yolov8n-face.pt")
        use_yolo = True
    else:
        log.warning("YOLO face model not found! Download yolov8n-face.pt")
except Exception as e:
    log.warning("Ultralytics not available: %s", e)


# LOAD DNN SSD FALLBACK 
use_dnn = False
dnn_net = None

if not use_yolo:
    proto = "deploy.prototxt"
    model = "res10_300x300_ssd_iter_140000.caffemodel"

    if Path(proto).exists() and Path(model).exists():
        dnn_net = cv2.dnn.readNetFromCaffe(proto, model)
        use_dnn = True
        log.info("Using SSD fallback detector")
    else:
        log.warning("DNN SSD not found.")


# LOAD HAAR FALLBACK
use_haar = False
face_cascade = None

if not use_yolo and not use_dnn:
    haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if Path(haar_path).exists():
        face_cascade = cv2.CascadeClassifier(haar_path)
        use_haar = True
        log.info("Using Haar fallback face detector")
    else:
        log.error("No detectors available.")


# FACE DETECTION FUNCTIONS 
def detect_yolo(img):
    try:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = yolo_model.predict(rgb, conf=0.25, verbose=False)

        boxes = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append((x1, y1, x2, y2))
        return boxes
    except:
        return []


def detect_dnn(img):
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                 (300, 300), (104, 177, 123))
    dnn_net.setInput(blob)
    detections = dnn_net.forward()

    boxes = []
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > 0.3:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            boxes.append((x1, y1, x2, y2))
    return boxes


def detect_haar(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = face_cascade.detectMultiScale(gray, 1.1, 4)
    return [(x, y, x+w, y+h) for (x, y, w, h) in rects]


def detect_faces(img):
    if use_yolo:
        boxes = detect_yolo(img)
        if boxes:
            return boxes

    if use_dnn:
        boxes = detect_dnn(img)
        if boxes:
            return boxes

    if use_haar:
        boxes = detect_haar(img)
        return boxes

    return []


def blur_faces(img, boxes):
    out = img.copy()
    for (x1, y1, x2, y2) in boxes:
        roi = out[y1:y2, x1:x2]
        blurred = cv2.GaussianBlur(roi, (51, 51), 30)
        out[y1:y2, x1:x2] = blurred
    return out


#  FASTAPI ENDPOINT
@app.post("/blur-face")
async def blur_face(file: UploadFile = File(...)):
    contents = await file.read()

    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    boxes = detect_faces(img)
    log.info(f"Faces detected: {len(boxes)}")

    blurred = blur_faces(img, boxes)

    _, buffer = cv2.imencode(".jpg", blurred)

    return Response(content=buffer.tobytes(), media_type="image/jpeg")
