import cv2 as cv
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from ultralytics import YOLO
from keras_facenet import FaceNet


class FACELOADING:
    def __init__(self, directory):
        self.directory = directory
        self.target_size = (160,160)
        self.X = []
        self.Y = []
        self.detector = YOLO('model/best.pt')
    

    def extract_face(self, filename):
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

            # Display the face using OpenCV
            cv.imshow('Detected Face', cv.cvtColor(face_arr, cv.COLOR_RGB2BGR))  # Convert back to BGR for OpenCV
            cv.waitKey(0)  # Wait for a key press
            cv.destroyAllWindows()  # Close the window after key press
        else:
            print("No face detected.")
        return face_arr

    def load_faces(self, dir):
        FACES = []
        for im_name in os.listdir(dir):
            try:
                path = dir + im_name
                print(f'Detecting face in {path}')
                single_face = self.extract_face(path)
                FACES.append(single_face)
            except Exception as e:
                pass
        return FACES

    def load_classes(self):
        for sub_dir in os.listdir(self.directory):
            path = self.directory +'/'+ sub_dir+'/'
            FACES = self.load_faces(path)
            labels = [sub_dir for _ in range(len(FACES))]
            print(f"Loaded successfully: {len(labels)}")
            self.X.extend(FACES)
            self.Y.extend(labels)
        
        return np.asarray(self.X), np.asarray(self.Y)


    def plot_images(self):
        plt.figure(figsize=(18,16))
        for num,image in enumerate(self.X):
            ncols = 3
            nrows = len(self.Y)//ncols + 1
            plt.subplot(nrows,ncols,num+1)
            plt.imshow(image)
            plt.axis('off')

def get_embedding(face_img, embedder):
    face_img = face_img.astype('float32') # 3D(160x160x3)
    face_img = np.expand_dims(face_img, axis=0) 
    # 4D (Nonex160x160x3)
    yhat= embedder.embeddings(face_img)
    return yhat[0] # 512D image (1x1x512)

if __name__ == "__main__":
    # Load the Detected Faces from Database
    loader = FACELOADING('dataset')
    X, Y = loader.load_classes()
    loader.plot_images()
    plt.show()
    print(X.shape, Y.shape)
    print(Y)

    # Initialize the FaceNet model
    embedder = FaceNet()

    # Get the embedding of all faces
    EMBEDDED_x = []

    for img in X:
        EMBEDDED_x.append(get_embedding(img, embedder))

    EMBEDDED_X = np.asarray(EMBEDDED_x)
    