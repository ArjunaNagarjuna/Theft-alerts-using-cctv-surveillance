import numpy as np
from collections import defaultdict
import time

class SimpleTracker:
    def __init__(self, max_disappeared=30):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = defaultdict(int)
        self.max_disappeared = max_disappeared
        
    def register(self, detection):
        self.objects[self.next_object_id] = {
            'bbox': detection['bbox'],
            'center': detection['center'],
            'class_id': detection['class_id'],
            'class_name': detection['class_name'],
            'last_seen': time.time(),
            'first_seen': time.time()
        }
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
        return self.next_object_id - 1
    
    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
    
    def update(self, detections):
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects
        
        if len(self.objects) == 0:
            for detection in detections:
                self.register(detection)
        else:
            object_ids = list(self.objects.keys())
            object_centers = [self.objects[oid]['center'] for oid in object_ids]
            detection_centers = [d['center'] for d in detections]
            
            dist_matrix = self._calculate_distances(object_centers, detection_centers)
            matched_indices = self._greedy_match(dist_matrix, threshold=100)
            
            unmatched_detections = set(range(len(detections)))
            unmatched_objects = set(object_ids)
            
            for obj_idx, det_idx in matched_indices:
                object_id = object_ids[obj_idx]
                self.objects[object_id].update({
                    'bbox': detections[det_idx]['bbox'],
                    'center': detections[det_idx]['center'],
                    'last_seen': time.time()
                })
                self.disappeared[object_id] = 0
                unmatched_detections.discard(det_idx)
                unmatched_objects.discard(object_id)
            
            for object_id in unmatched_objects:
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            for det_idx in unmatched_detections:
                self.register(detections[det_idx])
        
        return self.objects
    
    def _calculate_distances(self, centers1, centers2):
        centers1 = np.array(centers1)
        centers2 = np.array(centers2)
        dist = np.linalg.norm(centers1[:, np.newaxis] - centers2, axis=2)
        return dist
    
    def _greedy_match(self, dist_matrix, threshold=100):
        matches = []
        used_rows = set()
        used_cols = set()
        flat_indices = np.argsort(dist_matrix.ravel())
        
        for idx in flat_indices:
            row = idx // dist_matrix.shape[1]
            col = idx % dist_matrix.shape[1]
            
            if row in used_rows or col in used_cols:
                continue
            
            if dist_matrix[row, col] < threshold:
                matches.append((row, col))
                used_rows.add(row)
                used_cols.add(col)
        
        return matches
