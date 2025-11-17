# privacy_blur_east_improved.py
import os
import cv2
import numpy as np
import pytesseract
import random
import re
from rapidfuzz import fuzz

# ---------- CONFIG ----------
EAST_MODEL = r"C:\Users\jorda\Downloads\Text training\frozen_east_text_detection.pb"
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

INPUT_DIR = r"C:\Users\jorda\Downloads\Text training\Text dataset\testing_data\images"
OUTPUT_DIR = r"C:\Users\jorda\Downloads\Text training\Text dataset\blurred_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Label keywords (used as labels; must be followed by real content to trigger blur)
LABEL_KEYWORDS = [
    "name", "full name", "address", "email", "e-mail", "phone", "telephone",
    "tel", "contact", "dob", "birth", "date of birth", "id", "number", "no", "account"
]

# Patterns that individually indicate sensitive content
PATTERN_REGEXES = [
    re.compile(r'\b\d{10}\b'),  # 10-digit phone
    re.compile(r'\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'),  # 123-456-7890 or 123 456 7890
    re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'),  # email
    re.compile(r'\b\d{12}\b'),  # 12-digit id (Aadhaar-like)
    re.compile(r'\b\d{2}[/-]\d{2}[/-]\d{4}\b'),  # date
]

# fuzzy threshold for label detection
FUZZY_THRESHOLD = 80

# EAST decode helpers (unchanged)
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

# Utility: get word-level OCR data for full image
def get_tesseract_words(img):
    # returns list of dicts: {text, left, top, width, height, conf, line_num}
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config='--psm 6')
    words = []
    n_boxes = len(data['text'])
    for i in range(n_boxes):
        text = str(data['text'][i]).strip()
        if text == "":
            continue

        # Handle confidence safely across different pytesseract versions
        conf_val = data['conf'][i]
        if isinstance(conf_val, (int, float)):
            conf = int(conf_val)
        else:
            conf = int(conf_val) if str(conf_val).isdigit() else -1

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

# Merge boxes helper
def merge_boxes(boxes):
    if not boxes:
        return []
    boxes = np.array(boxes)
    x1 = np.min(boxes[:,0])
    y1 = np.min(boxes[:,1])
    x2 = np.max(boxes[:,2])
    y2 = np.max(boxes[:,3])
    return [int(x1), int(y1), int(x2), int(y2)]

# Check if candidate value is real content (not underscores, not too short)
def looks_like_filled_value(text):
    # remove underscores and hyphens and spaces
    cleaned = re.sub(r'[_\-\s]+', '', text)
    # require at least two alphanumeric characters
    return len(re.findall(r'[A-Za-z0-9]', cleaned)) >= 2

# Check patterns
def matches_pattern(text):
    for rx in PATTERN_REGEXES:
        if rx.search(text):
            return True
    return False

# Check label fuzzy match
def is_label(word_text):
    t = word_text.lower()
    for label in LABEL_KEYWORDS:
        if fuzz.partial_ratio(label, t) >= FUZZY_THRESHOLD:
            return True
    return False

# Main improved function
def detect_and_blur_image(image_path, output_path, east_threshold=0.5):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] cannot read {image_path}")
        return False

    orig = img.copy()
    H, W = img.shape[:2]

    # 1) get tesseract words for whole page (word-level boxes)
    # do light preprocessing for OCR: grayscale + adaptive threshold
    img_for_ocr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_for_ocr = cv2.threshold(img_for_ocr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    words = get_tesseract_words(img_for_ocr)  # words sorted by appearance

    # Build spatial index: group by line_num for quick neighbor lookups
    lines = {}
    for w in words:
        ln = w['line_num']
        lines.setdefault(ln, []).append(w)

    # 2) run EAST to get coarse text regions (to limit area and reduce false detections)
    newW, newH = 320, 320
    rW, rH = W / float(newW), H / float(newH)
    blob = cv2.dnn.blobFromImage(img, 1.0, (newW, newH),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net = cv2.dnn.readNet(EAST_MODEL)
    net.setInput(blob)
    layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
    (scores, geometry) = net.forward(layerNames)
    rects, confidences = decode_predictions(scores, geometry, conf_threshold=east_threshold)
    boxes = non_max_suppression(rects, confidences)
    # scale boxes
    scaled_boxes = []
    for (sx, sy, ex, ey) in boxes:
        sx = max(0, int(sx * rW)); sy = max(0, int(sy * rH))
        ex = min(W, int(ex * rW)); ey = min(H, int(ey * rH))
        scaled_boxes.append([sx, sy, ex, ey])

    # 3) decide what to blur using tesseract words and heuristics
    blurred_mask = np.zeros((H, W), dtype=np.uint8)
    blurred_any = False

    # build quick lookup: for each line, words sorted by left coordinate
    for ln, items in lines.items():
        items.sort(key=lambda k: k['left'])

    # Helper to find word(s) in given bbox (tesseract words)
    def words_in_bbox(bbox):
        x1,y1,x2,y2 = bbox
        found=[]
        for w in words:
            wx1, wy1 = w['left'], w['top']
            wx2, wy2 = wx1 + w['w'], wy1 + w['h']
            # if center of word is inside bbox
            cx, cy = (wx1+wx2)//2, (wy1+wy2)//2
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                found.append(w)
        return found

    # Iterate over scaled EAST boxes and also use full-page word data
    for sb in scaled_boxes:
        sb_words = words_in_bbox(sb)
        if not sb_words:
            continue

        # Check each word in the EAST box for:
        #  - pattern match (phone/email/ID) -> blur word bbox
        #  - label match -> check neighbor on same line (right side) for filled content -> blur combined
        for w in sb_words:
            wtext = w['text']
            # if word itself is a sensitive pattern (phone/email), blur it
            if matches_pattern(wtext):
                bx = [w['left'], w['top'], w['left']+w['w'], w['top']+w['h']]
                # avoid blurring if already blurred substantially
                mask_roi = blurred_mask[bx[1]:bx[3], bx[0]:bx[2]]
                if mask_roi.size>0 and mask_roi.mean() > 0.2*255:
                    continue
                roi = orig[bx[1]:bx[3], bx[0]:bx[2]]
                k = max(15,(roi.shape[1]//5)|1)
                roi_blur = cv2.GaussianBlur(roi, (k,k), 0)
                orig[bx[1]:bx[3], bx[0]:bx[2]] = roi_blur
                blurred_mask[bx[1]:bx[3], bx[0]:bx[2]] = 255
                blurred_any = True
                continue

            # label detection (fuzzy). If label, find neighbor(s) to the right in same line
            if is_label(wtext):
                ln = w['line_num']
                line_words = lines.get(ln, [])
                # find current index in line
                idx = None
                for i, lw in enumerate(line_words):
                    if lw is w:
                        idx = i
                        break
                if idx is None:
                    continue
                # candidate neighbor words to the right
                neighbor_texts = []
                neighbor_boxes = []
                # take up to next 3 words to the right (covers multi-word filled values)
                for j in range(idx+1, min(len(line_words), idx+4)):
                    nw = line_words[j]
                    neighbor_texts.append(nw['text'])
                    neighbor_boxes.append([nw['left'], nw['top'], nw['left']+nw['w'], nw['top']+nw['h']])
                # join neighbor text to check if filled
                joined = " ".join(neighbor_texts).strip()
                if joined and looks_like_filled_value(joined):
                    # merge label box + neighbor boxes into one bbox to blur
                    label_box = [w['left'], w['top'], w['left']+w['w'], w['top']+w['h']]
                    merge_candidates = [label_box] + neighbor_boxes
                    mb = merge_boxes(merge_candidates)
                    # safety check: ensure not mostly empty / underscores
                    # get tesseract-based cleaned text of merged region
                    cx1, cy1, cx2, cy2 = mb
                    cropped = img_for_ocr[cy1:cy2, cx1:cx2]
                    if cropped.size == 0:
                        continue
                    ttext = pytesseract.image_to_string(cropped, config='--psm 6').strip()
                    if len(re.sub(r'[_\-\s]+','', ttext)) < 2:
                        continue
                    # avoid double-blur
                    mask_roi = blurred_mask[cy1:cy2, cx1:cx2]
                    if mask_roi.size>0 and mask_roi.mean() > 0.2*255:
                        continue
                    roi = orig[cy1:cy2, cx1:cx2]
                    kx = max(15, (roi.shape[1]//5)|1)
                    ky = max(15, (roi.shape[0]//5)|1)
                    roi_blur = cv2.GaussianBlur(roi, (kx,kx), 0)
                    orig[cy1:cy2, cx1:cx2] = roi_blur
                    blurred_mask[cy1:cy2, cx1:cx2] = 255
                    blurred_any = True

    # final fallback: if nothing blurred above but patterns exist anywhere, blur them
    if not blurred_any:
        for w in words:
            if matches_pattern(w['text']):
                bx = [w['left'], w['top'], w['left']+w['w'], w['top']+w['h']]
                mask_roi = blurred_mask[bx[1]:bx[3], bx[0]:bx[2]]
                if mask_roi.size>0 and mask_roi.mean() > 0.2*255:
                    continue
                roi = orig[bx[1]:bx[3], bx[0]:bx[2]]
                k = max(15,(roi.shape[1]//5)|1)
                roi_blur = cv2.GaussianBlur(roi, (k,k), 0)
                orig[bx[1]:bx[3], bx[0]:bx[2]] = roi_blur
                blurred_mask[bx[1]:bx[3], bx[0]:bx[2]] = 255
                blurred_any = True

    # Save only if blurred_any True
    if blurred_any:
        cv2.imwrite(output_path, orig)
        print("[INFO] Blurred saved:", output_path)
    else:
        print("[INFO] No sensitive content found, skipped:", os.path.basename(image_path))

    return blurred_any

# ---------------------- MAIN ----------------------
if __name__ == "__main__":
    all_images = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png','.jpg','.jpeg'))]
    selected = random.sample(all_images, min(5, len(all_images)))
    print(f"[INFO] Running on {len(selected)} random images...")
    cnt=0
    for fn in selected:
        inp = os.path.join(INPUT_DIR, fn)
        out = os.path.join(OUTPUT_DIR, "blurred_"+fn)
        if detect_and_blur_image(inp, out):
            cnt+=1
    print(f"[DONE] {cnt} of {len(selected)} images contained sensitive info and were saved.")
