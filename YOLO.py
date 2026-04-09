from ultralytics import YOLO
import numpy as np

def find_stop_sign(I):
    model = YOLO("yolov8n.pt")
    results = model(I, verbose=False)
    
    for box in results[0].boxes:
        classifier = int(box.cls[0])
        label = model.names[classifier]
        
        if label == "stop sign":
            x1, y1, x2, y2 = box.xyxy[0]
            return np.array([y1, x1, (y2-y1), (x2-x1)]).astype(int)
    return None