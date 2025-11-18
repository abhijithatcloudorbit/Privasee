import os
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from ultralytics import YOLO
from io import BytesIO

# ---------------------------------------------------
# LOAD YOLO MODEL
# ---------------------------------------------------
MODEL_PATH = r"C:\Users\jorda\Downloads\LP training\license_plate_detector.pt"

print("Loading YOLOv8 license plate model...")
model = YOLO(MODEL_PATH)
print("Model loaded successfully")

# ---------------------------------------------------
# FASTAPI INIT
# ---------------------------------------------------
app = FastAPI(title="License Plate Blurring API")


# ---------------------------------------------------
# IMAGE PREPROCESSING
# ---------------------------------------------------
def preprocess_image(img):
    h, w = img.shape[:2]

    if max(h, w) < 720:
        scale = 720 / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    kernel = np.array([[0, -1,  0],
                       [-1,  5, -1],
                       [0, -1,  0]])
    img = cv2.filter2D(img, -1, kernel)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return img


# ---------------------------------------------------
# BLUR METHOD
# ---------------------------------------------------
def blur_region(image, x1, y1, x2, y2):
    roi = image[y1:y2, x1:x2]
    k = max(21, (abs(x2 - x1) // 4) | 1)
    blurred = cv2.GaussianBlur(roi, (k, k), 0)
    image[y1:y2, x1:x2] = blurred
    return image


# ---------------------------------------------------
# MAIN PROCESSING FUNCTION
# ---------------------------------------------------
def process_image_bytes(image_bytes):
    # Convert upload → cv2 image
    file_bytes = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return None, "Invalid Image File"

    img_processed = preprocess_image(img.copy())
    results = model(img_processed)

    blurred = False

    # YOLO detection
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < 0.40:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_processed.shape[1], x2)
            y2 = min(img_processed.shape[0], y2)

            img_processed = blur_region(img_processed, x1, y1, x2, y2)
            blurred = True

    return img_processed, blurred


# ---------------------------------------------------
# API ENDPOINT: UPLOAD + BLUR + RETURN OUTPUT
# ---------------------------------------------------
@app.post("/blur")
async def blur_license_plate(file: UploadFile = File(...)):
    if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
        return JSONResponse({"error": "Unsupported file type"}, status_code=400)

    image_bytes = await file.read()

    processed_img, blurred = process_image_bytes(image_bytes)

    if processed_img is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    # Convert CV image → bytes buffer for response
    _, buffer = cv2.imencode(".jpg", processed_img)
    io_buf = BytesIO(buffer.tobytes())

    return StreamingResponse(
        io_buf,
        media_type="image/jpeg",
        headers={
            "Content-Disposition": f"attachment; filename=blurred_{file.filename}"
        }
    )


# ---------------------------------------------------
# ROOT ENDPOINT
# ---------------------------------------------------
@app.get("/")
def home():
    return {"message": "License Plate Privacy API is running!"}
