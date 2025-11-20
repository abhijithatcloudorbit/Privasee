import os

SOURCE = r"C:\Users\jorda\Downloads\Face training\Dataset"
TARGET = r"C:\Users\jorda\Downloads\Face training\Dataset_YOLO\images\train"

print("\n===== SOURCE DATASET =====")
print(os.listdir(SOURCE))

print("\n===== YOLO TRAIN FOLDER =====")
if os.path.exists(TARGET):
    print(os.listdir(TARGET))
else:
    print("Folder does NOT exist")
