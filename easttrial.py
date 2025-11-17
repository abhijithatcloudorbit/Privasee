import cv2

EAST_MODEL = r"C:\Users\jorda\Downloads\Text training\frozen_east_text_detection.pb"
net = cv2.dnn.readNet(EAST_MODEL)
print("EAST model loaded successfully!")
