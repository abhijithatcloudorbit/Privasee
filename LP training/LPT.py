import os
import cv2
import numpy as np
from ultralytics import YOLO

# PATHS
MODEL_PATH = r"C:\Users\jorda\Downloads\LP training\license_plate_detector.pt"
INPUT_DIR = r"C:\Users\jorda\Downloads\LP training\Dataset"
OUTPUT_DIR = r"C:\Users\jorda\Downloads\LP training\blurred_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# LOAD YOLO MODEL
print("Loading YOLOv8 license plate model...")
model = YOLO(MODEL_PATH)
print("Model loaded successfully")


# IMAGE PREPROCESSING FOR FAR LICENSE PLATES
def preprocess_image(img):
    # Step 1: Increase resolution of small images
    h, w = img.shape[:2]
    if max(h, w) < 720:
        scale = 720 / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    kernel = np.array([[0, -1,  0],
                       [-1,  5, -1],
                       [0, -1,  0]])
    img = cv2.filter2D(img, -1, kernel)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return img

# BLUR FUNCTION
def blur_region(image, x1, y1, x2, y2):
    roi = image[y1:y2, x1:x2]
    
    # Strong Gaussian Blur
    k = max(21, (abs(x2-x1)//4) | 1)
    blurred = cv2.GaussianBlur(roi, (k, k), 0)
    
    image[y1:y2, x1:x2] = blurred
    return image
# PROCESS EACH IMAGE
def process_image(image_path, output_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error reading image: {image_path}")
        return False

    img_processed = preprocess_image(img.copy())
    results = model(img_processed)
    blurred = False
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            if conf < 0.40:
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_processed.shape[1], x2)
            y2 = min(img_processed.shape[0], y2)

            img_processed = blur_region(img_processed, x1, y1, x2, y2)
            blurred = True

    if blurred:
        cv2.imwrite(output_path, img_processed)
        print("Blurred:", output_path)
    else:
        print("No license plate detected in:", os.path.basename(image_path))

    return blurred
# MAIN EXECUTION
if __name__ == "__main__":
    images = [f for f in os.listdir(INPUT_DIR)
              if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    if not images:
        print("No images found in INPUT_DIR")
        exit()

    print(f"Processing {len(images)} images...")

    count = 0
    for img_name in images:
        inp = os.path.join(INPUT_DIR, img_name)
        out = os.path.join(OUTPUT_DIR, "blurred_" + img_name)

        if process_image(inp, out):
            count += 1

    print(f"Completed. {count} images had license plates and were blurred.")
