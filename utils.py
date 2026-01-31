import cv2
import numpy as np

def calculate_iou(box1, box2):
    """Calculate Intersection over Union"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def resize_frame(frame, max_width=1280):
    """Resize frame while maintaining aspect ratio"""
    height, width = frame.shape[:2]
    if width > max_width:
        ratio = max_width / width
        new_size = (max_width, int(height * ratio))
        return cv2.resize(frame, new_size)
    return frame
