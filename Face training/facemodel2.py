# ======================================================
# AI-Based Image Privacy Filter
# Model 1: Face Detection using YOLOv8n
# Dataset: WIDER FACE (Train/Val)
# Python 3.12 Compatible | CPU Safe
# ======================================================

import os
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

# ----------------------------
# PATH CONFIGURATION
# ----------------------------
TRAIN_DIR = r"C:\Users\jorda\Downloads\COT\face_train"
VAL_DIR = r"C:\Users\jorda\Downloads\COT\face_val"
PRETRAINED_FACE_MODEL = r"C:\Users\jorda\Downloads\COT\yolov8n-face.pt"
BASE_MODEL = "yolov8n.pt"  # YOLOv8n base model for training
DATA_YAML_PATH = "face_dataset.yaml"

# ----------------------------
# AUTO DATASET YAML CREATION
# ----------------------------
with open(DATA_YAML_PATH, "w") as f:
    f.write(f"""
path: C:/Users/jorda/Downloads/COT
train: face_train
val: face_val

names:
  0: face
""")

# =====================================================
# UNIQUE PREPROCESSING METHOD
# =====================================================
def enhance_image(image_path):
    """
    Unique preprocessing:
    1. CLAHE for contrast enhancement
    2. Gamma correction for lighting balance
    3. Resize to 640x640
    4. Light Gaussian noise addition
    """
    image = cv2.imread(image_path)
    if image is None:
        return None

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    gamma = 1.2
    look_up = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                        for i in np.arange(0, 256)]).astype("uint8")
    enhanced = cv2.LUT(enhanced, look_up)

    enhanced = cv2.resize(enhanced, (640, 640))

    noise = np.random.normal(0, 3, enhanced.shape).astype(np.uint8)
    final_img = cv2.addWeighted(enhanced, 0.98, noise, 0.02, 0)
    return final_img

# =====================================================
# AUTO LABEL GENERATOR (using pretrained YOLOv8n-face)
# =====================================================
def generate_labels_if_missing(folder_path, model):
    """
    Uses pretrained YOLOv8n-face to auto-label faces if .txt labels are missing.
    """
    print(f"\n🔍 Checking folder for missing labels: {folder_path}")
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(root, file_name)
                label_path = os.path.splitext(image_path)[0] + ".txt"

                if os.path.exists(label_path):
                    continue  # Skip already labeled images

                image = cv2.imread(image_path)
                if image is None:
                    continue

                results = model(image, verbose=False)
                with open(label_path, "w") as f:
                    for box in results[0].boxes.xywhn:
                        x_center, y_center, width, height = box.tolist()
                        f.write(f"0 {x_center} {y_center} {width} {height}\n")
    print(f"✅ Auto-labeling completed for {folder_path}")

# =====================================================
# PREPROCESS SAMPLE BATCH
# =====================================================
def preprocess_sample_images(folder):
    count = 0
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(root, file)
                out = enhance_image(img_path)
                if out is not None and count < 5:
                    cv2.imwrite(f"sample_pre_{count}.jpg", out)
                    count += 1
    print(f"✅ Sample preprocessing done on {count} images.")

# =====================================================
# TRAINING PIPELINE
# =====================================================
def train_model():
    print("🚀 Starting YOLOv8n training for Face Detection...\n")

    # Step 1: Auto-label data if missing
    face_model = YOLO(PRETRAINED_FACE_MODEL)
    generate_labels_if_missing(TRAIN_DIR, face_model)
    generate_labels_if_missing(VAL_DIR, face_model)

    # Step 2: Run preprocessing sample check (optional)
    preprocess_sample_images(TRAIN_DIR)

    # Step 3: Train the YOLOv8n model
    model = YOLO(BASE_MODEL)
    results = model.train(
        data=DATA_YAML_PATH,
        epochs=50,
        imgsz=640,
        batch=8,
        name="face_privacy_yolov8n",
        device='cpu',
        lr0=0.001,
        optimizer="Adam",
        augment=True,
        patience=10
    )

    # Step 4: Validation Metrics
    metrics = model.val()
    print("🎯 Validation Results:")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")

    # Step 5: Visualization
    plt.figure(figsize=(10, 5))
    plt.plot(results.results_dict['train/box_loss'], label='Box Loss')
    plt.plot(results.results_dict['train/cls_loss'], label='Class Loss')
    plt.plot(results.results_dict['train/dfl_loss'], label='DFL Loss')
    plt.title("Training Loss Curves - YOLOv8n Face Detection")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\n✅ Training Completed Successfully!")
    print("🧠 Model Saved Under: runs/detect/face_privacy_yolov8n/")

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    train_model()
