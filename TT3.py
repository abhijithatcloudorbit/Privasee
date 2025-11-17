import os
import cv2
import numpy as np
import pytesseract
import random
import re
from rapidfuzz import fuzz

# CONFIGURATION
EAST_MODEL = r"C:\Users\jorda\Downloads\Text training\frozen_east_text_detection.pb"
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

INPUT_DIR = r"C:\Users\jorda\Downloads\Text training\Text dataset\testing_data\images"
OUTPUT_DIR = r"C:\Users\jorda\Downloads\Text training\blurred_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# label keywords is used as labels which must be followed by real content to trigger blur
LABEL_KEYWORDS = [
    "name", "full name", "address", "email", "e-mail", "phone", "telephone",
    "tel", "contact", "dob", "birth", "date of birth", "id", "number", "no", "account",
    "to", "from", "date", "submitted", "submit", "signature", "initials", "ssn", "social security",
    "passport", "license", "card", "user", "username", "password", "pin", "code", "reference",
    "recipient", "company", "fax", "mobile", "cell", "home", "work", "business", "personal"
]

# thes are the patterns that individually indicate sensitive content
PATTERN_REGEXES = [
    re.compile(r'\b\d{10}\b'),  # 10-digit phone
    re.compile(r'\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'),  # 123-456-7890 or 123 456 7890
    re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'),  # email
    re.compile(r'\b\d{12}\b'),  # 12-digit id (Aadhaar-like)
    re.compile(r'\b\d{2}[/-]\d{2}[/-]\d{4}\b'),  # date
    re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),  # SSN format
    re.compile(r'\b[A-Za-z]{2}\d{6}\b'),  # passport/ID format
    re.compile(r'\b\d{16}\b'),  # credit card number
    re.compile(r'\b\d{4}[-\s]\d{4}[-\s]\d{4}[-\s]\d{4}\b'),  # formatted credit card
    re.compile(r'\(\d{3}\)\s*\d{3}[-\s]\d{4}'),  # (123) 456-7890 format
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
    n_boxes = len(data['text'])
    for i in range(n_boxes):
        text = str(data['text'][i]).strip()
        if text == "":
            continue
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

def merge_boxes(boxes):
    if not boxes:
        return []
    boxes = np.array(boxes)
    x1 = np.min(boxes[:,0])
    y1 = np.min(boxes[:,1])
    x2 = np.max(boxes[:,2])
    y2 = np.max(boxes[:,3])
    return [int(x1), int(y1), int(x2), int(y2)]

def looks_like_filled_value(text):
    cleaned = re.sub(r'[_\-\s]+', '', text)
    return len(re.findall(r'[A-Za-z0-9]', cleaned)) >= 2

# checks the patterns
def matches_pattern(text):
    for rx in PATTERN_REGEXES:
        if rx.search(text):
            return True
    return False

# check label fuzzy match
def is_label(word_text, selected_labels=None):
    t = word_text.lower()
    if selected_labels is None:
        for label in LABEL_KEYWORDS:
            if fuzz.partial_ratio(label, t) >= FUZZY_THRESHOLD:
                return True
    else:
        for label in selected_labels:
            if fuzz.partial_ratio(label, t) >= FUZZY_THRESHOLD:
                return True
    return False

def find_related_content(words, label_word, img_width, img_height):
    """Find content that appears to be associated with a label"""
    label_x = label_word['left']
    label_y = label_word['top']
    label_w = label_word['w']
    label_h = label_word['h']
    
    right_candidates = []
    below_candidates = []
    column_candidates = []
    
    for word in words:
        if word is label_word:
            continue
            
        word_x = word['left']
        word_y = word['top']
        word_w = word['w']
        word_h = word['h']
        
        if (word_x > label_x + label_w and 
            abs(word_y - label_y) < max(label_h, word_h) * 2 and
            word_x < label_x + img_width * 0.3):  
            right_candidates.append(word)
            
        if (word_y > label_y + label_h and 
            abs(word_x - label_x) < max(label_w, word_w) * 2 and
            word_y < label_y + img_height * 0.2):  
            below_candidates.append(word)
        
        if (abs(word_x - label_x) < max(label_w, word_w) * 2 and
            word_y > label_y + label_h and
            word_y < label_y + img_height * 0.5):  
            column_candidates.append(word)
    

    def distance_to_label(word):
        word_center_x = word['left'] + word['w'] // 2
        word_center_y = word['top'] + word['h'] // 2
        label_center_x = label_x + label_w // 2
        label_center_y = label_y + label_h // 2
        return ((word_center_x - label_center_x) ** 2 + (word_center_y - label_center_y) ** 2) ** 0.5
    
    right_candidates.sort(key=distance_to_label)
    below_candidates.sort(key=distance_to_label)
    column_candidates.sort(key=distance_to_label)
    
    candidates = right_candidates[:3] + below_candidates[:3] + column_candidates[:3]
    
    if candidates:
        joined_text = " ".join([w['text'] for w in candidates])
        return candidates, joined_text.strip()
    
    return [], ""

def find_column_headers(words, selected_labels=None):
    headers = []
    header_columns = {}
    
    for word in words:
        if word['top'] < 100:  
            if is_label(word['text'], selected_labels):
                headers.append(word)
                header_columns[word['text']] = []
    
    for word in words:
        for header in headers:
            if (word['top'] > header['top'] + header['h'] and
                abs(word['left'] - header['left']) < max(header['w'], word['w']) * 3):
                header_columns[header['text']].append(word)
    
    return headers, header_columns

def detect_and_blur_image(image_path, output_path, east_threshold=0.5, selected_labels=None):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] cannot read {image_path}")
        return False

    orig = img.copy()
    H, W = img.shape[:2]

    # light preprocessing for OCR: grayscale + adaptive threshold
    img_for_ocr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_for_ocr = cv2.threshold(img_for_ocr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    words = get_tesseract_words(img_for_ocr)  

    lines = {}
    for w in words:
        ln = w['line_num']
        lines.setdefault(ln, []).append(w)

    # run EAST to get coarse text regions which is to limit area and reduce false detections
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
    scaled_boxes = []
    for (sx, sy, ex, ey) in boxes:
        sx = max(0, int(sx * rW)); sy = max(0, int(sy * rH))
        ex = min(W, int(ex * rW)); ey = min(H, int(ey * rH))
        scaled_boxes.append([sx, sy, ex, ey])

    # to decide what to blur using tesseract words and heuristics
    blurred_mask = np.zeros((H, W), dtype=np.uint8)
    blurred_any = False

    for ln, items in lines.items():
        items.sort(key=lambda k: k['left'])

    # a kind of helper to find words in given box (tesseract words)
    def words_in_bbox(bbox):
        x1,y1,x2,y2 = bbox
        found=[]
        for w in words:
            wx1, wy1 = w['left'], w['top']
            wx2, wy2 = wx1 + w['w'], wy1 + w['h']
            cx, cy = (wx1+wx2)//2, (wy1+wy2)//2
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                found.append(w)
        return found
    
    for w in words:
        wtext = w['text']
        if matches_pattern(wtext):
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

 
    for w in words:
        wtext = w['text']
        if is_label(wtext, selected_labels):
            related_words, related_text = find_related_content(words, w, W, H)
            
            if related_text and looks_like_filled_value(related_text):
                all_boxes = [[w['left'], w['top'], w['left']+w['w'], w['top']+w['h']]]
                for rw in related_words:
                    all_boxes.append([rw['left'], rw['top'], rw['left']+rw['w'], rw['top']+rw['h']])
                
                merged_box = merge_boxes(all_boxes)
                x1, y1, x2, y2 = merged_box
                
                # to check that we are not blurring empty space
                cropped = img_for_ocr[y1:y2, x1:x2]
                if cropped.size == 0:
                    continue
                    
                ttext = pytesseract.image_to_string(cropped, config='--psm 6').strip()
                if len(re.sub(r'[_\-\s]+','', ttext)) < 2:
                    continue
                    
                # to avoid double-blur
                mask_roi = blurred_mask[y1:y2, x1:x2]
                if mask_roi.size>0 and mask_roi.mean() > 0.2*255:
                    continue
                    
                roi = orig[y1:y2, x1:x2]
                kx = max(15, (roi.shape[1]//5)|1)
                ky = max(15, (roi.shape[0]//5)|1)
                roi_blur = cv2.GaussianBlur(roi, (kx,kx), 0)
                orig[y1:y2, x1:x2] = roi_blur
                blurred_mask[y1:y2, x1:x2] = 255
                blurred_any = True

    headers, header_columns = find_column_headers(words, selected_labels)
    
    for header_text, column_words in header_columns.items():
        if is_label(header_text, selected_labels):
            for word in column_words:
                if matches_pattern(word['text']) or looks_like_filled_value(word['text']):
                    bx = [word['left'], word['top'], word['left']+word['w'], word['top']+word['h']]
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
    for sb in scaled_boxes:
        sb_words = words_in_bbox(sb)
        if not sb_words:
            continue

        for w in sb_words:
            wtext = w['text']
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

    # this looks for phone or fax labels and blur entire column
    phone_labels = []
    if selected_labels is None:
        phone_labels = [w for w in words if is_label(w['text']) and ('phone' in w['text'].lower() or 'fax' in w['text'].lower())]
    else:
        phone_labels = [w for w in words if is_label(w['text'], selected_labels) and ('phone' in w['text'].lower() or 'fax' in w['text'].lower())]
    
    for phone_label in phone_labels:
        # to find all words in the same column as phone label
        col_words = []
        for word in words:
            if word is phone_label:
                continue
            if abs(word['left'] - phone_label['left']) < max(phone_label['w'], word['w']) * 3:
                # And below the label
                if word['top'] > phone_label['top'] + phone_label['h']:
                    col_words.append(word)
        
        # to blur all phone or fax numbers in this column
        for word in col_words:
            if matches_pattern(word['text']):
                bx = [word['left'], word['top'], word['left']+word['w'], word['top']+word['h']]
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

    # only save  if the blurred_any is true
    if blurred_any:
        cv2.imwrite(output_path, orig)
        print("[INFO] Blurred saved:", output_path)
    else:
        print("[INFO] No sensitive content found, skipped:", os.path.basename(image_path))

    return blurred_any

# function for user selection 
def get_user_selection():
    print("\n" + "="*60)
    print("WHAT TO BLUR IN THE GIVEN IMAGES?")
    print("="*60)
    print("Select the types of sensitive information you want to blur:")
    print("Enter numbers separated by commas (i.e : 1,3,5) or 'all' for everything")
    print("-"*60)
    
    for i, keyword in enumerate(LABEL_KEYWORDS, 1):
        print(f"{i:2d}. {keyword}")
    
    print("-"*60)
    print("Type 'all' to select everything")
    print("Type 'quit' to exit without processing")
    print("="*60)
    
    while True:
        try:
            user_input = input("Your selection (numbers separated by commas, 'all', or 'quit'): ").strip().lower()
            
            if user_input == 'quit':
                print("Exiting program...")
                return None
            
            if user_input == 'all':
                print("Selected: All categories")
                return LABEL_KEYWORDS
        
            selections = []
            parts = user_input.split(',')
            
            for part in parts:
                part = part.strip()
                if part.isdigit():
                    num = int(part)
                    if 1 <= num <= len(LABEL_KEYWORDS):
                        selections.append(LABEL_KEYWORDS[num-1])
                    else:
                        print(f"Warning: {num} is not a valid option (1-{len(LABEL_KEYWORDS)})")
                else:
                    print(f"Warning: '{part}' is not a valid number")
            
            if not selections:
                print("No valid selections made. Please try again.")
                continue
            
            print(f"Selected: {', '.join(selections)}")
            return selections
            
        except Exception as e:
            print(f"Error: {e}. Please try again.")

# MAIN 
if __name__ == "__main__":
    selected_labels = get_user_selection()
    if selected_labels is None:
        exit(0)
    
    all_images = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png','.jpg','.jpeg'))]
    selected = random.sample(all_images, min(5, len(all_images)))
    print(f"\n[INFO] Running on {len(selected)} random images...")
    cnt=0
    for fn in selected:
        inp = os.path.join(INPUT_DIR, fn)
        out = os.path.join(OUTPUT_DIR, "blurred_"+fn)
        if detect_and_blur_image(inp, out, selected_labels=selected_labels):
            cnt+=1
    print(f"[DONE] {cnt} of {len(selected)} images contained sensitive info and were saved.")