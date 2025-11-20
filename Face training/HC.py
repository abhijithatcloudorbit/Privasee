import os
import shutil
import random
from ultralytics import YOLO
import matplotlib.pyplot as plt

# ============================================
# PATHS
# ============================================
SOURCE_DATASET = r"C:\Users\jorda\Downloads\Face training\Dataset"
YOLO_DATASET = r"C:\Users\jorda\Downloads\Face training\Dataset_YOLO"
DATA_YAML = "face_dataset.yaml"

# ============================================
# STEP 1 — CREATE YOLO FORMAT FOLDERS
# ============================================
images_train = os.path.join(YOLO_DATASET, "images/train")
images_val = os.path.join(YOLO_DATASET, "images/val")
labels_train = os.path.join(YOLO_DATASET, "labels/train")
labels_val = os.path.join(YOLO_DATASET, "labels/val")

for path in [images_train, images_val, labels_train, labels_val]:
    os.makedirs(path, exist_ok=True)

# ============================================
# STEP 2 — SCAN ALL IMAGES + LABEL FILES
# ============================================
images = [f for f in os.listdir(SOURCE_DATASET) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
random.shuffle(images)

# 80% train / 20% val
split_index = int(len(images) * 0.8)
train_files = images[:split_index]
val_files = images[split_index:]

def move_files(file_list, img_dest, label_dest):
    for img_file in file_list:
        src_img = os.path.join(SOURCE_DATASET, img_file)
        src_label = os.path.join(SOURCE_DATASET, os.path.splitext(img_file)[0] + ".txt")

        shutil.copy(src_img, os.path.join(img_dest, img_file))

        if os.path.exists(src_label):
            shutil.copy(src_label, os.path.join(label_dest, os.path.splitext(img_file)[0] + ".txt"))

# Move files
move_files(train_files, images_train, labels_train)
move_files(val_files, images_val, labels_val)

print("Dataset successfully converted to YOLO structure!")

# ============================================
# STEP 3 — WRITE YAML FILE
# ============================================
with open(DATA_YAML, "w") as f:
    f.write(f"""
path: {YOLO_DATASET}

train: images/train
val: images/val

names:
  0: face
""")

print("YAML file created!")

# ============================================
# STEP 4 — TRAIN YOLOv8-M HIGH ACCURACY
# ============================================
print("Starting YOLOv8-M training...")

model = YOLO("yolov8m.pt")

results = model.train(
    data=DATA_YAML,
    epochs=80,
    imgsz=640,
    batch=16,
    device="cpu",      # change to "0" if GPU available
    optimizer="AdamW",
    lr0=0.001,
    patience=20,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    translate=0.2,
    scale=0.5,
    shear=1.0,
    mosaic=1.0,
    name="face_detection_high_accuracy"
)

# ============================================
# STEP 5 — VALIDATION RESULTS
# ============================================
metrics = model.val()
print("Validation Results:")
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")

# ============================================
# STEP 6 — PLOT TRAINING CURVES
# ============================================
plt.figure(figsize=(10, 5))
plt.plot(results.results_dict['train/box_loss'], label='Box Loss')
plt.plot(results.results_dict['train/cls_loss'], label='Class Loss')
plt.plot(results.results_dict['train/dfl_loss'], label='DFL Loss')
plt.title("YOLOv8-M Training Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

print("Training completed successfully!")
