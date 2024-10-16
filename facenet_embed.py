import os
import numpy as np
import cv2
from keras_facenet import FaceNet
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from scipy.spatial import distance


if __name__ == '__main__':
    # Initialize the FaceNet model
    embedder = FaceNet()

    # Directory where extracted faces are saved (from your previous code)
    faces_dir = 'extracted_faces/'

    # List of extracted face images
    face_filenames = [os.path.join(faces_dir, f) for f in os.listdir(faces_dir) if f.endswith('.jpg')]

    identities = []

    # Process each face image
    for face_filename in face_filenames:
        # Read the face image
        face_image = cv2.imread(face_filename)

        # Convert the image to RGB (FaceNet expects RGB images)
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        # Resize the image to 160x160 pixels
        face_image = cv2.resize(face_image, (160, 160))

        # Normalize pixel values (to range [-1, 1] as expected by FaceNet)
        face_image = face_image.astype('float32')
        # face_image = (face_image / 127.5) - 1.0  # Scale pixel values to [-1, 1] (gaperlu normalize jing)

        # Add batch dimension (FaceNet expects input in the shape (batch_size, 160, 160, 3))
        face_image = np.expand_dims(face_image, axis=0)

        # Generate face embeddings using FaceNet
        embeddings = embedder.embeddings(face_image)
        
        print(f"Face Embedding for {face_filename}: {embeddings[0]}")
        identities.append(embeddings[0])

    distance = np.linalg.norm(identities[0] - identities[1])
    print(f"Distance between embeddings 0 and 2: {distance}")
    print("Comparing faces...")
    print(embedder.compute_distance(identities[0], identities[1])) # Compute distance between embeddings
