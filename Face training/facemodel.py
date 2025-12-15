import os
import sys
import cv2
import math
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

DATASET_DIR = r"C:\Users\jorda\Downloads\Face training\Dataset"   
OUTPUT_DIR = r"C:\Users\jorda\Downloads\Face training\blurred_outputs_yolo"
SAMPLES_DIR = os.path.join(OUTPUT_DIR, "samples")
SAVE_ALL = True            
SAMPLES_TO_SAVE = 5
YOLO_MODEL_FILENAME = "yolov8n-face.pt"   
DNN_PROTO = "deploy.prototxt"             
DNN_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"
YOLO_CONF = 0.25
DNN_CONF = 0.3
HAAR_MIN_NEIGHBORS = 4
HAAR_MIN_SIZE = (20, 20)


os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

# logging
logfile = os.path.join(OUTPUT_DIR, f"run_{datetime.now():%Y%m%d_%H%M%S}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(logfile), logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("FaceBlurYOLO")

# Detector selection 
use_yolo = False
yolo_model = None
try:
    from ultralytics import YOLO
    
    if Path(YOLO_MODEL_FILENAME).exists():
        log.info("Found local YOLO face weights: %s", YOLO_MODEL_FILENAME)
        try:
            yolo_model = YOLO(YOLO_MODEL_FILENAME)
            use_yolo = True
            log.info("Loaded YOLO model from local file.")
        except Exception as e:
            log.warning("Failed to load YOLO from local file: %s", e)
            yolo_model = None
    else:
        try:
            yolo_model = YOLO("yolov8n-face.pt")  
            use_yolo = True
            log.info("Loaded YOLO model 'yolov8n-face.pt' from hub.")
        except Exception:
            try:
                
                yolo_model = YOLO("yolov8n.pt")
                use_yolo = True
                log.info("Loaded generic YOLOv8n model (may detect persons, not faces precisely).")
            except Exception as e:
                log.warning("Unable to load YOLO models: %s", e)
                yolo_model = None
except Exception as e:
    log.info("Ultralytics YOLO not installed or failed to import: %s", e)
    yolo_model = None

# fallback: OpenCV DNN SSD face detector
use_dnn = False
dnn_net = None
if not use_yolo:
    if Path(DNN_PROTO).exists() and Path(DNN_MODEL).exists():
        try:
            dnn_net = cv2.dnn.readNetFromCaffe(DNN_PROTO, DNN_MODEL)
            use_dnn = True
            log.info("Using OpenCV DNN SSD face detector (Caffe).")
        except Exception as e:
            log.warning("Failed to load DNN SSD: %s", e)
            dnn_net = None
    else:
        log.info("DNN SSD files not found; skipping DNN fallback.")

# fallback: Haar cascade
use_haar = False
face_cascade = None
if not use_yolo and not use_dnn:
    haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if Path(haar_path).exists():
        face_cascade = cv2.CascadeClassifier(haar_path)
        use_haar = True
        log.info("Using Haar cascade face detector (fallback).")
    else:
        log.error("No face detector available: enable ultralytics or place DNN files, or ensure OpenCV contains Haar cascades.")
        raise SystemExit("No detector available.")

#helper functions
def detect_with_yolo(img_bgr, conf=YOLO_CONF):
    """
    Use ultralytics YOLO model to detect faces.
    Returns list of boxes (x1,y1,x2,y2) in pixel coords.
    """
    try:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = yolo_model.predict(source=rgb, conf=conf, verbose=False)
        boxes = []
        if len(results) == 0:
            return boxes
        res = results[0]
        if hasattr(res, "boxes") and res.boxes is not None:
            for box in res.boxes:
                coords = box.xyxy.cpu().numpy().astype(int).flatten() if hasattr(box.xyxy, "cpu") else np.array(box.xyxy).astype(int).flatten()
                x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                boxes.append((x1, y1, x2, y2))
        else:
            try:
                xy = res.boxes.xyxy
                for row in xy:
                    x1, y1, x2, y2 = map(int, row[:4])
                    boxes.append((x1, y1, x2, y2))
            except Exception:
                pass
        return boxes
    except Exception as e:
        log.warning("YOLO detection failed for one image: %s", e)
        return []

def detect_with_dnn(img_bgr, conf_thresh=DNN_CONF):
    h, w = img_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img_bgr, (300,300)), 1.0, (300,300), (104.0, 177.0, 123.0))
    dnn_net.setInput(blob)
    detections = dnn_net.forward()
    boxes = []
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf > conf_thresh:
            box = detections[0,0,i,3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w-1, x2), min(h-1, y2)
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
    return boxes

def detect_with_haar(img_bgr, scaleFactor=1.1, minNeighbors=HAAR_MIN_NEIGHBORS, minSize=HAAR_MIN_SIZE):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    rects = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize)
    boxes = []
    for (x,y,w,h) in rects:
        boxes.append((int(x), int(y), int(x+w), int(y+h)))
    return boxes

def detect_faces(img_bgr):
    """
    Try detectors in order: YOLO -> DNN -> Haar.
    Returns list of boxes (x1,y1,x2,y2). Always return list (possibly empty).
    """
    if use_yolo and yolo_model is not None:
        boxes = detect_with_yolo(img_bgr)
        if boxes:
            return boxes
    if use_dnn and dnn_net is not None:
        boxes = detect_with_dnn(img_bgr)
        if boxes:
            return boxes
    if use_haar and face_cascade is not None:
        boxes = detect_with_haar(img_bgr)
        if boxes:
            return boxes
    return []

def adaptive_blur(img_bgr, boxes, max_kernel=101):
    """
    For each box apply a gaussian blur whose kernel scales with box size.
    Returns new image copy.
    """
    out = img_bgr.copy()
    for (x1,y1,x2,y2) in boxes:
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(out.shape[1], int(x2)), min(out.shape[0], int(y2))
        if x2 <= x1 or y2 <= y1:
            continue
        roi = out[y1:y2, x1:x2]
        h, w = roi.shape[:2]
        k = max(3, int(min(h,w)//3))
        if k % 2 == 0:
            k += 1
        k = min(k, max_kernel)
        if k < 3:
            k = 3
        blurred = cv2.GaussianBlur(roi, (k,k), 0)
        out[y1:y2, x1:x2] = blurred
    return out

#Run over dataset
if not os.path.isdir(DATASET_DIR):
    log.error("DATASET_DIR missing: %s", DATASET_DIR)
    sys.exit(1)

files = sorted([f for f in os.listdir(DATASET_DIR) if f.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp"))])
if len(files) == 0:
    log.error("No image files found in DATASET_DIR.")
    sys.exit(1)

summary = {"images":0, "images_with_faces":0, "total_faces":0}
sample_saved = 0

log.info("Starting detection & blur run on %d images", len(files))

for fname in tqdm(files, desc="Processing images"):
    in_path = os.path.join(DATASET_DIR, fname)
    img = cv2.imread(in_path)
    if img is None:
        log.warning("Failed to read image: %s", in_path)
        continue

    boxes = detect_faces(img)
    summary["images"] += 1
    summary["total_faces"] += len(boxes)
    if len(boxes) > 0:
        summary["images_with_faces"] += 1

    out = adaptive_blur(img, boxes)

    out_fname = f"blurred_{fname}"
    out_path = os.path.join(OUTPUT_DIR, out_fname)
    ok = cv2.imwrite(out_path, out)
    if not ok:
        log.error("cv2.imwrite failed for %s", out_path)
    if sample_saved < SAMPLES_TO_SAVE:
        sample_path = os.path.join(SAMPLES_DIR, f"sample_{sample_saved+1}_{fname}")
        ok2 = cv2.imwrite(sample_path, out)
        if ok2:
            sample_saved += 1

log.info("Run complete.")
log.info("Images processed: %d", summary["images"])
log.info("Images with >=1 face detected: %d", summary["images_with_faces"])
log.info("Total faces detected: %d", summary["total_faces"])
if summary["images"]>0:
    log.info("Avg faces per image: %.3f", summary["total_faces"]/summary["images"])

log.info("Outputs written to: %s", OUTPUT_DIR)
log.info("Sample outputs (first %d) in: %s", sample_saved, SAMPLES_DIR)

if summary["images_with_faces"] == 0:
    log.warning("Detector found ZERO faces in all images. Try lowering YOLO/DNN threshold or place a face-specific YOLO weight file (yolov8n-face.pt) in the script folder, or use the DNN/prototxt Caffe files.")
    log.info("YOLO used? %s | DNN used? %s | Haar used? %s", use_yolo, use_dnn, use_haar)
else:
    missed = summary["images"] - summary["images_with_faces"]
    if missed > 0:
        log.warning("%d images had no detected faces. For those, try lowering thresholds or using a stronger detector (YOLO face weights).", missed)

