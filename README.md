# Face Recognition Capstone B05

## Dependency Installation

```python
pip install -r requirements.txt
```

## Scripts
- **test_web.py**: A simple script to test the YOLO functionality
- **face_select_automation.py**: Automate the YOLO functionality to extract the faces detected into images in the extracted_faces folder
- **facenet_embed.py**: A simple script to test the FaceNet functionality
- **yolo_dataset_automate.py**: Automates the process of extracting faces from the dataset directory, embed it with FaceNet and automatically maps it into vectors of embedded_X(the embeddings) and Y(the identity labels)
- **SVM_train.py**: script to make an SVM model out of the vectors made by yolo_dataset_automate
- **main_pipeline.py**: Script of pipeline for the real deal (accepting an image and identify all the face in it)

## Folders
- **dataset**: Bunch of identities, each with photos of the identity (e.g. identity of Dua Lipa with photos of her inside. It is used to make the SVM model to recognize an input into an identity.
- **extracted_faces**: Bunch of cropped faces detected by face_selection_automation.py
- **model**: Bunch of models and weights (best.pt is the weight I trained with WIDERFACE)
- **runs**: A bunch of files made by YOLO (it's default output directory)

## Contributing

Initialized by Anto Gaming

## License

[UGM](https://youtu.be/ZRTNHDd0gL8?si=RG6w2Z71-PId7ac5)