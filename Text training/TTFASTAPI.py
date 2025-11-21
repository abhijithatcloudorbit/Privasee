
#Code which can be used for integration
import os
import cv2
import numpy as np
import pytesseract
import re
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from rapidfuzz import fuzz
import io

# CONFIGURATION

TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

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

# HELPER FUNCTIONS

def get_tesseract_words(img):
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config='--psm 6')
    words = []
    for i in range(len(data['text'])):
        text = str(data['text'][i]).strip()
        if text == "":
            continue

        conf_val = data['conf'][i]
        conf = int(conf_val) if str(conf_val).isdigit() else -1

        words.append({
            'text': text,
            'left': int(data['left'][i]),
            'top': int(data['top'][i]),
            'w': int(data['width'][i]),
            'h': int(data['height'][i]),
            'conf': conf
        })

    return words


def matches_pattern(text):
    return any(rx.search(text) for rx in PATTERN_REGEXES)


def detect_and_blur_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    words = get_tesseract_words(img_bin)

    blurred_any = False

    for w in words:
        if matches_pattern(w["text"]):
            x1, y1 = w['left'], w['top']
            x2, y2 = x1 + w['w'], y1 + w['h']

            roi = orig[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            k = max(15, (roi.shape[1] // 5) | 1)
            orig[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
            blurred_any = True

    return orig if blurred_any else image


# FASTAPI APP

app = FastAPI(title="Image Privacy Redaction API", version="1.0")


@app.post("/process-image/")
async def process_image(file: UploadFile = File(...)):
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    processed = detect_and_blur_image(img)

    _, buffer = cv2.imencode(".png", cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
    io_buf = io.BytesIO(buffer.tobytes())

    return StreamingResponse(io_buf, media_type="image/png")
