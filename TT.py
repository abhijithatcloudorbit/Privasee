# ======================================================
# AI-Based Image Privacy Filter
# Model 2: Text Detection & Privacy Redaction
# Dataset: FUNSD or Similar (Train/Val)
# Python 3.12 Compatible | CPU Friendly
# ======================================================

import os
import cv2
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


# PATH CONFIGURATION
BASE_PATH = r"C:\Users\jorda\Downloads\Text training\Text dataset"

TRAIN_IMG_DIR = os.path.join(BASE_PATH, "training_data", "images")
TRAIN_ANN_DIR = os.path.join(BASE_PATH, "training_data", "annotations")
VAL_IMG_DIR = os.path.join(BASE_PATH, "testing_data", "images")
VAL_ANN_DIR = os.path.join(BASE_PATH, "testing_data", "annotations")
MODEL_SAVE_PATH = os.path.join(BASE_PATH, "text_privacy_model.keras")

# UNIQUE PREPROCESSING METHOD
def preprocess_text_image(image):
    """
    Unique preprocessing method for text image enhancement:
    1. CLAHE for contrast enhancement
    2. Adaptive thresholding for text edge clarity
    3. Morphological cleaning to remove lines or background noise
    4. Normalization and resizing for CNN input
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 1: Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Step 2: Adaptive thresholding
    thresh = cv2.adaptiveThreshold(enhanced, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 10)

    # Step 3: Morphological cleaning
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Step 4: Resize and normalize
    resized = cv2.resize(cleaned, (128, 128))
    normalized = resized / 255.0

    return np.expand_dims(normalized, axis=-1)

# DATA LOADER FROM JSON LABELS

def load_data(img_dir, ann_dir):
    X, y = [], []
    for file_name in os.listdir(img_dir):
        if file_name.lower().endswith(".png"):
            img_path = os.path.join(img_dir, file_name)
            ann_path = os.path.join(ann_dir, file_name.replace(".png", ".json"))

            image = cv2.imread(img_path)
            if image is None or not os.path.exists(ann_path):
                continue

            processed_img = preprocess_text_image(image)

            # Parse JSON to determine if private data is present
            with open(ann_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            text_fields = [obj.get("text", "").lower() for obj in data.get("form", [])]
            sensitive_keywords = ["name", "address", "phone", "id", "email"]
            label = 1 if any(any(k in t for k in sensitive_keywords) for t in text_fields) else 0

            X.append(processed_img)
            y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# LOAD TRAIN & VALIDATION DATA

print("Loading and preprocessing training data...")
X_train, y_train = load_data(TRAIN_IMG_DIR, TRAIN_ANN_DIR)
print(f"Loaded {len(X_train)} training samples.")

print("Loading and preprocessing validation data...")
X_val, y_val = load_data(VAL_IMG_DIR, VAL_ANN_DIR)
print(f"Loaded {len(X_val)} validation samples.")

# MODEL DEFINITION (Light CNN)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.0005),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# DATA AUGMENTATION

datagen = ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1
)
datagen.fit(X_train)

# MODEL CHECKPOINT

checkpoint = ModelCheckpoint(
    MODEL_SAVE_PATH,
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# TRAINING

print("Starting training for Text Privacy Detection with Augmentation...")

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=16),
    validation_data=(X_val, y_val),
    epochs=25,
    callbacks=[checkpoint],
    verbose=1
)


# SAVE TRAINED MODEL

model.save(MODEL_SAVE_PATH)
print(f"Trained model saved successfully at: {MODEL_SAVE_PATH}")

# EVALUATION

val_loss, val_acc = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {val_acc * 100:.2f}%")

# VISUALIZATION

plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

print("Training completed - Text Detection Model ready for redaction.")
