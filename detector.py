from ultralytics import YOLO
import cv2
import numpy as np
import config

class ObjectDetector:
    def __init__(self, model_path='yolov8n.pt'):
        """Initialize YOLOv8 detector"""
        try:
            self.model = YOLO(model_path)
            self.model.to('cpu')
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
    def detect_objects(self, frame):
        """Detect people and tracked objects"""
        results = self.model(frame, conf=config.CONFIDENCE_THRESHOLD, 
                           iou=config.IOU_THRESHOLD, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = result.names[class_id]
                
                if class_id == 0 or class_id in config.TRACKED_OBJECTS:
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': conf,
                        'class_id': class_id,
                        'class_name': class_name,
                        'center': [int((x1+x2)/2), int((y1+y2)/2)]
                    })
        
        return detections
