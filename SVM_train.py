from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import pickle
from keras_facenet import FaceNet
import cv2

def SVM_Train():
    # Load the data
    data = np.load('faces_embeddings_done_4classes.npz')
    EMBEDDED_X = data['embeddings']
    Y = data['labels']

    # Encode the labels
    encoder = LabelEncoder()
    encoder.fit(Y)
    Y = encoder.transform(Y)

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(EMBEDDED_X, Y, shuffle=True, random_state=17)

    # Train the SVM model
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, Y_train)

    # Make predictions
    ypreds_train = model.predict(X_train)
    ypreds_test = model.predict(X_test)

    # Calculate accuracy
    train_accuracy = accuracy_score(Y_train, ypreds_train)
    print(f"Training accuracy: {train_accuracy}")

    #save the model
    with open('model/svm_model_160x160.pkl','wb') as f:
        pickle.dump(model,f)

# Test the model
def SVM_Test(face_filename):
    # Load the Y labels
    data = np.load('faces_embeddings_done_4classes.npz')
    Y = data['labels']

    # Load the FaceNet model
    embedder = FaceNet()

    # Details in facenet_embed.py
    face_image = cv2.imread(face_filename)
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    face_image = cv2.resize(face_image, (160, 160))
    face_image = face_image.astype('float32')
    face_image = np.expand_dims(face_image, axis=0)
    # Generate face embeddings using FaceNet
    embeddings = embedder.embeddings(face_image)

    # Load the SVM model
    with open('model/svm_model_160x160.pkl', 'rb') as f:
        model = pickle.load(f)

    encoder = LabelEncoder()
    encoder.fit(Y)
        
    embeddings = embeddings
    ypreds = model.predict(embeddings)
    print(ypreds)
    print(encoder.inverse_transform(ypreds))

    

if __name__ == '__main__':
    # Train the SVM model
    # SVM_Train()

    # Test the Model
    SVM_Test('extracted_faces/face_0.jpg')