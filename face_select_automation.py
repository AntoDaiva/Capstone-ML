import cv2
from ultralytics import YOLO
import os

# Load the YOLO face detection model
model = YOLO('model/best.pt')  # Load pre-trained YOLO face detection model

# Define image path and output directory for the extracted faces
image_path = 'test_assets/tes_img_1.jpg'
output_dir = 'extracted_faces/'
os.makedirs(output_dir, exist_ok=True)

# Load the image
image = cv2.imread(image_path)

# Perform face detection using YOLO
results = model(image)

# Loop over detected faces (each detection)
for i, (bbox, conf, cls) in enumerate(zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls)):
    x_min, y_min, x_max, y_max = map(int, bbox)  # Get bounding box coordinates

    # Extract the face from the original image using the bounding box
    face_crop = image[y_min:y_max, x_min:x_max]

    # Resize the extracted face to 160x160
    # face_resized = cv2.resize(face_crop, (160, 160))

    # Save the resized face to the output directory
    face_filename = os.path.join(output_dir, f'face_{i}.jpg')
    cv2.imwrite(face_filename, face_crop)
    print(f"Resized face {i} saved to {face_filename}")

# Optionally show the detection on the original image (with bounding boxes)
for bbox in results[0].boxes.xyxy:
    x_min, y_min, x_max, y_max = map(int, bbox)
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

# Display the image with detected faces
cv2.imshow("Detected Faces", image)
cv2.waitKey(0)
cv2.destroyAllWindows()