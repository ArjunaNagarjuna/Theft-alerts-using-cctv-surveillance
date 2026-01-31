from group_manager import GroupManager

import time
import numpy as np

class OwnershipManager:
    def __init__(self):
        self.ownership_map = {}
        self.person_entry_objects = {}
        self.person_exit_objects = {}
        self.unattended_objects = {}
        self.group_manager = GroupManager()  # ADD THIS LINE
 

    def calculate_proximity(self, person_bbox, object_bbox):
        x1_p, y1_p, x2_p, y2_p = person_bbox
        x1_o, y1_o, x2_o, y2_o = object_bbox
        
        x1_i = max(x1_p, x1_o)
        y1_i = max(y1_p, y1_o)
        x2_i = min(x2_p, x2_o)
        y2_i = min(y2_p, y2_o)
        
        if x2_i > x1_i and y2_i > y1_i:
            return 1.0
        
        center_p = [(x1_p + x2_p) / 2, (y1_p + y2_p) / 2]
        center_o = [(x1_o + x2_o) / 2, (y1_o + y2_o) / 2]
        distance = np.sqrt((center_p[0] - center_o[0])**2 + (center_p[1] - center_o[1])**2)
        
        proximity = 1 / (1 + distance / 100)
        return proximity
    
    def associate_objects_to_people(self, tracked_objects):
        people = {id: obj for id, obj in tracked_objects.items() if obj['class_id'] == 0}
        objects = {id: obj for id, obj in tracked_objects.items() if obj['class_id'] != 0}
        
        current_time = time.time()
        
        for obj_id, obj_data in objects.items():
            best_person = None
            best_proximity = 0
            
            for person_id, person_data in people.items():
                proximity = self.calculate_proximity(person_data['bbox'], obj_data['bbox'])
                
                if proximity > best_proximity and proximity > 0.3:
                    best_proximity = proximity
                    best_person = person_id
            
            if best_person:
                if obj_id not in self.ownership_map:
                    self.ownership_map[obj_id] = {
                        'owner': best_person,
                        'timestamp': current_time,
                        'status': 'with_owner'
                    }
                else:
                    previous_owner = self.ownership_map[obj_id]['owner']
                    if best_person != previous_owner:
                        self.ownership_map[obj_id] = {
                            'owner': best_person,
                            'previous_owner': previous_owner,
                            'timestamp': current_time,
                            'status': 'transferred'
                        }
                
                if obj_id in self.unattended_objects:
                    del self.unattended_objects[obj_id]
            else:
                if obj_id in self.ownership_map and obj_id not in self.unattended_objects:
                    self.unattended_objects[obj_id] = {
                        'owner': self.ownership_map[obj_id]['owner'],
                        'unattended_since': current_time
                    }
    
    def get_unattended_duration(self, object_id):
        if object_id in self.unattended_objects:
            return time.time() - self.unattended_objects[object_id]['unattended_since']
        return 0
    def check_group_authorization(self, obj_id, previous_owner, new_owner):
        """Check if object transfer is authorized based on group membership.
    Returns (is_authorized, alert_type, description)
    """
        if not self.group_manager.is_group_object(obj_id):
        # Object not registered in any group - use normal theft detection
            return False, 'theft', f"Unregistered object transfer"
    
        is_auth, reason, alert_type = self.group_manager.is_authorized_transfer(obj_id, previous_owner, new_owner
    )
    
        if is_auth:
        # Authorized transfer within group
            return True, 'authorized_transfer', reason
        else:
        # Unauthorized - potential theft
            return False, 'group_theft', reason

