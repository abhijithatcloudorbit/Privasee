import os
import cv2
import numpy as np
import pytesseract
import re
import tempfile
from rapidfuzz import fuzz
import streamlit as st

# CONFIGURATION
EAST_MODEL = r"C:\Users\jorda\Downloads\Text training\frozen_east_text_detection.pb"
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# label keywords for label-dependent detection
LABEL_KEYWORDS = [
    "name", "full name", "address", "email", "e-mail", "phone", "telephone",
    "tel", "contact", "dob", "birth", "date of birth", "id", "number", "no", "account",
    "to", "from", "date", "submitted", "submit", "signature", "initials", "ssn", "social security",
    "passport", "license", "card", "user", "username", "password", "pin", "code", "reference",
    "recipient", "company", "fax", "mobile", "cell", "home", "work", "business", "personal"
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

FUZZY_THRESHOLD = 80

def decode_predictions(scores, geometry, conf_threshold=0.5):
    (num_rows, num_cols) = scores.shape[2:4]
    rects, confidences = [], []
    for y in range(num_rows):
        for x in range(num_cols):
            score = scores[0, 0, y, x]
            if score < conf_threshold:
                continue
            offsetX, offsetY = x * 4.0, y * 4.0
            angle = geometry[0, 4, y, x]
            cos, sin = np.cos(angle), np.sin(angle)
            h = geometry[0, 0, y, x] + geometry[0, 2, y, x]
            w = geometry[0, 1, y, x] + geometry[0, 3, y, x]
            endX = int(offsetX + (cos * geometry[0, 1, y, x]) + (sin * geometry[0, 2, y, x]))
            endY = int(offsetY - (sin * geometry[0, 1, y, x]) + (cos * geometry[0, 2, y, x]))
            startX = int(endX - w)
            startY = int(endY - h)
            rects.append((startX, startY, endX, endY))
            confidences.append(float(score))
    return rects, confidences


def non_max_suppression(boxes, probs=None, overlapThresh=0.3):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes).astype("float")
    pick = []
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(probs)
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
    return boxes[pick].astype("int")


def get_tesseract_words(img):
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config='--psm 6')
    words = []
    for i in range(len(data['text'])):
        text = str(data['text'][i]).strip()
        if text == "":
            continue
        conf_val = data['conf'][i]
        if str(conf_val).isdigit():
            conf = int(conf_val)
        else:
            conf = -1

        words.append({
            'text': text,
            'left': int(data['left'][i]),
            'top': int(data['top'][i]),
            'w': int(data['width'][i]),
            'h': int(data['height'][i]),
            'conf': conf,
            'line_num': int(data['line_num'][i])
        })
    return words


def merge_boxes(boxes):
    boxes = np.array(boxes)
    return [int(np.min(boxes[:,0])), int(np.min(boxes[:,1])),
            int(np.max(boxes[:,2])), int(np.max(boxes[:,3]))]


def looks_like_filled_value(text):
    cleaned = re.sub(r'[_\-\s]+', '', text)
    return len(re.findall(r'[A-Za-z0-9]', cleaned)) >= 2


def matches_pattern(text):
    return any(rx.search(text) for rx in PATTERN_REGEXES)


def is_label(word_text, selected_labels):
    t = word_text.lower()
    return any(fuzz.partial_ratio(label, t) >= 80 for label in selected_labels)


def detect_and_blur_image(image, selected_labels):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig = img.copy()
    H, W = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_for_ocr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    words = get_tesseract_words(img_for_ocr)
    blurred_mask = np.zeros((H, W), dtype=np.uint8)
    blurred_any = False

    for w in words:
        if matches_pattern(w["text"]):
            x1, y1 = w['left'], w['top']
            x2, y2 = x1 + w['w'], y1 + w['h']

            roi = orig[y1:y2, x1:x2]
            k = max(15, (roi.shape[1] // 5) | 1)
            orig[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
            blurred_any = True

    return orig if blurred_any else image


# STREAMLIT UI
st.title("Image Privacy Redaction System")
st.write("Upload an image and choose which fields you wish to blur")

uploaded_file = st.file_uploader("Upload Image", type=['png','jpg','jpeg'])

user_labels = st.multiselect(
    "What should be blurred?",
    LABEL_KEYWORDS,
    default=[]
)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(img, caption="Original Image", use_column_width=True)

    if st.button("Process Image"):
        processed = detect_and_blur_image(img, user_labels)

        st.image(processed, caption="Blurred Output", use_column_width=True)

        # prepare file for download
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        cv2.imwrite(temp_file.name, cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))

        with open(temp_file.name, "rb") as f:
            st.download_button(
                label="Download Processed Image",
                data=f,
                file_name="processed.png",
                mime="image/png"
            )
