from pathlib import Path
from ultralytics import YOLO
# Load a model

if __name__ == '__main__':
    # Train the model
    

    data_path = str(Path('VisDrone.yaml').resolve())

    
    model = YOLO(r'yolov5su.pt')
    model.train(data=data_path, epochs=100, imgsz=640, degrees=30, batch=4, workers=8)
    
 