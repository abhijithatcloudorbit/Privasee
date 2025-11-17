# craft_training_custom.py
import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import sys

# Add path to cloned CRAFT repo
sys.path.append(r"C:\Users\jorda\Downloads\CRAFT-pytorch")  # adjust your path
from craft import CRAFT
from craft_utils import getDetBoxes, adjustResultCoordinates
from dataset import SynthTextDataset  # or your custom dataset
from torch.utils.tensorboard import SummaryWriter

# ================= PATHS =================
TRAIN_IMG_DIR = r"C:\Users\jorda\Downloads\Text training\CRAFT\train_images"
TRAIN_LABEL_DIR = r"C:\Users\jorda\Downloads\Text training\CRAFT\train_labels"
VAL_IMG_DIR = r"C:\Users\jorda\Downloads\Text training\CRAFT\val_images"
VAL_LABEL_DIR = r"C:\Users\jorda\Downloads\Text training\CRAFT\val_labels"
MODEL_SAVE_PATH = r"C:\Users\jorda\Downloads\Text training\CRAFT\craft_custom.pth"

# ================= CUSTOM PREPROCESSING =================
def preprocess_image(image):
    """
    Grayscale + Gaussian blur + Laplacian edge + contrast stretch + normalization
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    lap = cv2.Laplacian(blur, cv2.CV_64F)
    lap = cv2.convertScaleAbs(lap)
    contrast = cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX)
    normalized = contrast / 255.0
    return normalized[np.newaxis, :, :].astype(np.float32)

# ================= DATASET =================
# Use your custom dataset class that returns (image_tensor, gt_score_map, gt_affinity_map)
train_dataset = SynthTextDataset(TRAIN_IMG_DIR, TRAIN_LABEL_DIR, preprocess=preprocess_image)
val_dataset = SynthTextDataset(VAL_IMG_DIR, VAL_LABEL_DIR, preprocess=preprocess_image)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

# ================= MODEL =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CRAFT()
model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 50
writer = SummaryWriter(log_dir="./logs")  # optional for TensorBoard

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for imgs, gt_score_maps, gt_affinity_maps in train_loader:
        imgs = imgs.to(device).float()
        gt_score_maps = gt_score_maps.to(device).float()
        gt_affinity_maps = gt_affinity_maps.to(device).float()
        
        optimizer.zero_grad()
        y_score, y_affinity = model(imgs)
        loss1 = criterion(y_score, gt_score_maps)
        loss2 = criterion(y_affinity, gt_affinity_maps)
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    avg_train_loss = running_loss / len(train_loader)

    # VALIDATION
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, gt_score_maps, gt_affinity_maps in val_loader:
            imgs = imgs.to(device).float()
            gt_score_maps = gt_score_maps.to(device).float()
            gt_affinity_maps = gt_affinity_maps.to(device).float()
            y_score, y_affinity = model(imgs)
            loss1 = criterion(y_score, gt_score_maps)
            loss2 = criterion(y_affinity, gt_affinity_maps)
            val_loss += (loss1+loss2).item()
    
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f} Val Loss: {avg_val_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"CRAFT model saved at {MODEL_SAVE_PATH}")
