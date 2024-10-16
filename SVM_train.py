from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np

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