import cv2 as cv
import os
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
TF_ENABLE_ONEDNN_OPTS=0
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
import pickle


# Modify this one so instead of loading from a file, it receives the image from the webcam
def extract_faces(filename, model):
    img = cv.imread(filename)
    results = model(img)

    # Initialize an empty list to store face arrays
    face_arrays = []

    # Iterate over all detected boxes
    for i, (bbox, conf, cls) in enumerate(zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls)):
        x_min, y_min, x_max, y_max = map(int, bbox)  # Get bounding box coordinates

        # Extract the face from the original image using the bounding box
        face_crop = img[y_min:y_max, x_min:x_max]

        # Convert the face to RGB (FaceNet expects RGB images)
        face_crop = cv.cvtColor(face_crop, cv.COLOR_BGR2RGB)
        # Resize the extracted face to 160x160
        face_resized = cv.resize(face_crop, (160, 160))

        print(f"Face {i} detected with confidence {conf:.2f}")
        # Append the processed face to the list
        face_arrays.append(face_resized)

    if len(face_arrays) == 0:
        print("No faces detected.")
    
    return face_arrays

def embed_faces(face_arrays):

    identities = []
    # Load the FaceNet model
    embedder = FaceNet()

    for face in face_arrays:
        face = face.astype('float32')
        face = np.expand_dims(face, axis=0)

        # Generate face embeddings using FaceNet
        embeddings = embedder.embeddings(face)

        identities.append(embeddings[0])
    
    return identities

def identify_faces(embeddings):
    # Load the Y labels
    data = np.load('faces_embeddings_done_4classes.npz')
    Y = data['labels']

    # Load the SVM model
    with open('model/svm_model_160x160.pkl', 'rb') as f:
        model = pickle.load(f)

    encoder = LabelEncoder()
    encoder.fit(Y)
    
    identified_faces = []
    ypreds = model.predict(embeddings)
    identified_faces = encoder.inverse_transform(ypreds)

    return identified_faces


if __name__ == "__main__":
    # Load the YOLO face detection model
    model = YOLO('model/best.pt')  # Load pre-trained YOLO face detection model

    face_arrays = extract_faces('test_assets/tes_img_1.jpg', model)

    embeddings = embed_faces(face_arrays)

    identities = identify_faces(embeddings)

    print(identities)
