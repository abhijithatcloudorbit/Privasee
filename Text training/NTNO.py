import cv2
import numpy as np
import easyocr
import re
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from rapidfuzz import fuzz

app = FastAPI()


# OCR + PRIVACY CONFIG
reader = easyocr.Reader(["en"], gpu=False)

LABEL_KEYWORDS = [
    "name", "full name", "address", "email", "e-mail", "phone", "telephone",
    "tel", "contact", "dob", "birth", "date of birth", "id", "number", "no",
    "account", "to", "from", "date", "submitted", "submit", "signature",
    "initials", "ssn", "social security", "passport", "license", "card",
    "user", "username", "password", "pin", "code", "reference", "recipient",
    "company", "fax", "mobile", "cell", "home", "work", "business", "personal"
]

PATTERN_REGEXES = [
    re.compile(r'\b\d{10}\b'),
    re.compile(r'\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'),
    re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'),
    re.compile(r'\b\d{12}\b'),
    re.compile(r'\b\d{2}[/-]\d{2}[/-]\d{4}\b'),
    re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
    re.compile(r'\b[A-Za-z]{2}\d{6}\b'),
    re.compile(r'\b\d{16}\b'),
    re.compile(r'\b\d{4}[-\s]\d{4}[-\s]\d{4}[-\s]\d{4}\b'),
    re.compile(r'\(\d{3}\)\s*\d{3}[-\s]\d{4}')
]

def matches_pattern(text):
    return any(rx.search(text) for rx in PATTERN_REGEXES)

def looks_like_label(text, selected_labels):
    t = text.lower()
    return any(fuzz.partial_ratio(label, t) >= 80 for label in selected_labels)

# MAIN REDACTION LOGIC

def process_image(image_bytes, selected_labels):
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = reader.readtext(img_rgb)
    output = img_rgb.copy()
    blurred = False

    for (bbox, text, conf) in result:
        if conf < 0.3:
            continue

        pts = np.array(bbox).astype(int)

        x1, y1 = np.min(pts[:, 0]), np.min(pts[:, 1])
        x2, y2 = np.max(pts[:, 0]), np.max(pts[:, 1])

        if matches_pattern(text) or looks_like_label(text, selected_labels):
            roi = output[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            k = max(15, (roi.shape[1] // 5) | 1)
            output[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
            blurred = True

    final_img = output if blurred else img_rgb
    return cv2.imencode('.png', cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))[1]


# FASTAPI ROUTE

@app.post("/process")
async def blur_image(
    image: UploadFile = File(...),
    labels: str = ""
):
    selected_labels = labels.split(",") if labels else []

    img_bytes = await image.read()
    processed = process_image(img_bytes, selected_labels)

    return Response(content=processed.tobytes(), media_type="image/png")
