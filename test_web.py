from ultralytics import YOLO

# Load model orang
# model = YOLO("yolov8n-face.pt")

# Load custom model
model = YOLO("model/best.pt")

results = model.predict(source="test_assets/tes_muka.mp4", show=True, save=True)

#live cam
# results = model.predict(source=0, show=True)

print(results)