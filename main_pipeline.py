import cv2 as cv
import os
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from keras_facenet import FaceNet

# Modify this one so instead of loading from a file, it receives the image from the webcam
def extract_faces(self, filename):
    img = cv.imread(filename)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = self.detector(img)
    # Ensure there is at least one detection
    if len(results[0].boxes.xyxy) > 0:
        # Extract bounding box from the first detection
        x_min, y_min, x_max, y_max = map(int, results[0].boxes.xyxy[0])  # Get bounding box coordinates

        # Crop the face using the bounding box coordinates
        face = img[y_min:y_max, x_min:x_max]

        # Resize the face to the target size
        face_arr = cv.resize(face, self.target_size)
    else:
        print("No face detected.")
    return face_arr

