#this code can be used for integration
import os
import cv2
import pytesseract
import easyocr
import numpy as np
import random
import re
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from pytesseract import Output
# CONFIG
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

USE_GPU = False
easy_reader = easyocr.Reader(["en"], gpu=USE_GPU)

# Temporary folder to store uploaded & output images
TEMP_IN = "temp_inputs"
TEMP_OUT = "temp_outputs"
os.makedirs(TEMP_IN, exist_ok=True)
os.makedirs(TEMP_OUT, exist_ok=True)

# PATTERNS
SENSITIVE_PATTERNS = [
    r"\bname\b", r"\bpatient\b", r"\bmr\b", r"\bmrs\b", r"\bms\b",
    r"\bdr\b", r"\bdoctor\b", r"\bphysician\b", r"\bmd\b", r"\bm\.d\b",
    r"\bsurgeon\b", r"\bclinic\b", r"\bhospital\b", r"\bcenter\b",
    r"\baddress\b", r"\baddr\b",
    r"\bmobile\b", r"\bphone\b", r"\btel\b", r"\bcontact\b",
    r"\bemail\b", r"\bmail\b",
    r"\buhid\b", r"\burid\b",
    r"\bpatient[\s_-]*id\b", 
    r"\breg\b", r"\bregistration\b", r"\bregn\b",
    r"\b\d{10}\b",
    r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b",
    r"\b[a-zA-Z]{1,3}\s*\d{3,6}\b",
]

SAFE_WORDS = [
    "mg", "ml", "tablet", "tab", "cap", "caps", "inj", "syrup",
    "age", "years", "yr", "yrs", "mg/ml", "dose"
]

DOCTOR_TOKENS = [r"\bdr\b", r"\bphysician\b", r"\bmd\b",
                 r"\bm\.d\b", r"\bdnb\b", r"\bmbbs\b"]

# UTILITY FUNCTIONS
def normalize_text(s):
    return re.sub(r'[^A-Za-z0-9@.\-_\s]', ' ', s).strip().lower()

def contains_safe_only(text):
    t = normalize_text(text)
    tokens = [tok for tok in re.split(r'\s+', t) if tok]
    if not tokens:
        return False
    safe_count = sum(1 for tok in tokens for sw in SAFE_WORDS if sw in tok)
    return safe_count == len(tokens)

def looks_sensitive(text):
    if not text.strip():
        return False
    t = normalize_text(text)

    if contains_safe_only(t):
        return False

    for pat in DOCTOR_TOKENS:
        if re.search(pat, t):
            return True

    for pat in SENSITIVE_PATTERNS:
        if re.search(pat, t):
            return True

    if re.search(r'\bname\b', t):
        return True

    if re.search(r'\d{5,}', t):
        return True

    return False


def expand_box(box, image_shape, pad_x=0.05, pad_y=0.25):
    x, y, w, h = box
    H, W = image_shape[:2]
    pad_w = int(w * pad_x)
    pad_h = int(h * pad_y)
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(W, x + w + pad_w)
    y2 = min(H, y + h + pad_h)
    return (x1, y1, x2 - x1, y2 - y1)


def merge_line_boxes(boxes, words, y_thresh=20):
    if not boxes:
        return [], []

    items = sorted(zip(boxes, words), key=lambda b: (b[0][1], b[0][0]))
    merged = []
    merged_texts = []

    cur_box, cur_text = items[0][0], items[0][1]
    for (b, text) in items[1:]:
        x, y, w, h = b
        cx, cy, cw, ch = cur_box
        if abs(y - cy) < y_thresh:
            nx1 = min(cx, x)
            ny1 = min(cy, y)
            nx2 = max(cx + cw, x + w)
            ny2 = max(cy + ch, y + h)
            cur_box = (nx1, ny1, nx2 - nx1, ny2 - ny1)
            cur_text = cur_text + " " + text
        else:
            merged.append(cur_box)
            merged_texts.append(cur_text)
            cur_box, cur_text = b, text

    merged.append(cur_box)
    merged_texts.append(cur_text)

    return merged, merged_texts


def blur_region(img, box, ksize=(65, 65)):
    x, y, w, h = box
    H, W = img.shape[:2]
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(W, int(x + w))
    y2 = min(H, int(y + h))
    if x2 <= x1 or y2 <= y1:
        return img

    roi = img[y1:y2, x1:x2]
    kx, ky = ksize
    if kx % 2 == 0: kx += 1
    if ky % 2 == 0: ky += 1

    blurred = cv2.GaussianBlur(roi, (kx, ky), 0)
    img[y1:y2, x1:x2] = blurred

    return img


# OCR
def ocr_tesseract_with_boxes(img_bgr):
    try:
        data = pytesseract.image_to_data(img_bgr, output_type=Output.DICT)
    except:
        return [], []

    boxes, words = [], []
    n = len(data['text'])
    for i in range(n):
        text = data['text'][i].strip()
        if not text:
            continue
        x = int(data['left'][i])
        y = int(data['top'][i])
        w = int(data['width'][i])
        h = int(data['height'][i])
        boxes.append((x, y, w, h))
        words.append(text)
    return boxes, words


def ocr_easyocr_with_boxes(img_bgr):
    try:
        results = easy_reader.readtext(img_bgr)
    except:
        return [], []

    boxes, words = [], []
    for (bbox, text, conf) in results:
        pts = np.array(bbox).astype(int)
        x1 = int(np.min(pts[:, 0]))
        y1 = int(np.min(pts[:, 1]))
        x2 = int(np.max(pts[:, 0]))
        y2 = int(np.max(pts[:, 1]))
        boxes.append((x1, y1, x2 - x1, y2 - y1))
        words.append(text)
    return boxes, words


# PROCESS SINGLE IMAGE
def anonymize_image(input_path, output_path):

    img = cv2.imread(input_path)
    if img is None:
        raise ValueError("Could not read image")

    h, w = img.shape[:2]

    boxes, words = ocr_tesseract_with_boxes(img)
    if not words:
        boxes, words = ocr_easyocr_with_boxes(img)

    merged_boxes, merged_texts = merge_line_boxes(boxes, words, y_thresh=25)

    sensitive_boxes = []
    for box, text in zip(merged_boxes, merged_texts):
        if looks_sensitive(text):
            expanded = expand_box(box, img.shape, pad_x=0.05, pad_y=0.25)
            sensitive_boxes.append(expanded)

    out_img = img.copy()

    for b in sensitive_boxes:
        out_img = blur_region(out_img, b, ksize=(65, 65))

    cv2.imwrite(output_path, out_img)
    return output_path


# FASTAPI APP
app = FastAPI()


@app.post("/anonymize")
async def anonymize(file: UploadFile = File(...)):

    in_path = os.path.join(TEMP_IN, file.filename)
    out_path = os.path.join(TEMP_OUT, file.filename)

    with open(in_path, "wb") as f:
        f.write(await file.read())

    result = anonymize_image(in_path, out_path)

    return FileResponse(
        path=result,
        media_type="image/jpeg",
        filename="anonymized_" + file.filename
    )
