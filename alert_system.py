import cv2
import time
import os
from datetime import datetime
import config
from audio_alert import AudioAlert

class AlertSystem:
    def __init__(self, ownership_manager):
        self.ownership_manager = ownership_manager
        self.alerts = []
        self.alert_count = {'theft': 0, 'suspicious': 0, 'abandonment': 0, 'lost': 0}
        self.audio_alert = AudioAlert()
        
    def check_alerts(self, tracked_objects, frame, frame_number):
        current_alerts = []
        
        for obj_id, unattended_data in self.ownership_manager.unattended_objects.items():
            duration = self.ownership_manager.get_unattended_duration(obj_id)
            
            if obj_id in tracked_objects:
                obj_info = tracked_objects[obj_id]
                owner_id = unattended_data['owner']
                
                if duration < config.THEFT_THRESHOLD:
                    alert_type = 'potential_theft'
                    confidence = 60
                elif duration < config.SUSPICIOUS_THRESHOLD:
                    alert_type = 'suspicious_retrieval'
                    confidence = 75
                elif duration < config.LOST_THRESHOLD:
                    alert_type = 'abandonment'
                    confidence = 85
                else:
                    alert_type = 'lost_object'
                    confidence = 95
                
                if not self._is_duplicate_alert(obj_id, alert_type, frame_number):
                    alert = self._generate_alert(
                        alert_type=alert_type,
                        object_id=obj_id,
                        person_id=owner_id,
                        object_data=obj_info,
                        frame=frame,
                        frame_number=frame_number,
                        duration=duration,
                        confidence=confidence
                    )
                    current_alerts.append(alert)
                    self.alerts.append(alert)
                    self.alert_count[alert_type.replace('potential_', '')] += 1
        
        for obj_id, ownership_data in self.ownership_manager.ownership_map.items():
            if ownership_data['status'] == 'transferred':
                if not self._is_duplicate_alert(obj_id, 'theft', frame_number):
                    obj_info = tracked_objects.get(obj_id)
                    if obj_info:
                        alert = self._generate_alert(
                            alert_type='theft',
                            object_id=obj_id,
                            person_id=ownership_data['owner'],
                            object_data=obj_info,
                            frame=frame,
                            frame_number=frame_number,
                            previous_owner=ownership_data.get('previous_owner'),
                            confidence=90
                        )
                        current_alerts.append(alert)
                        self.alerts.append(alert)
                        self.alert_count['theft'] += 1
        
        return current_alerts
    
    def _generate_alert(self, alert_type, object_id, person_id, object_data, frame, 
                       frame_number, duration=0, previous_owner=None, confidence=80):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        description = self._generate_description(
            alert_type, object_id, person_id, object_data, duration, previous_owner
        )
        
        snapshot_path = self._save_snapshot(frame, object_data['bbox'], alert_type, frame_number)
        
        # Play alarm sound
        if alert_type in ['theft', 'potential_theft']:
            self.audio_alert.play_theft_alarm(alert_type)
            print(f"🚨 ALARM: {alert_type.upper()} detected!")
        
        alert = {
            'id': len(self.alerts) + 1,
            'type': alert_type,
            'timestamp': timestamp,
            'frame_number': frame_number,
            'object_id': object_id,
            'person_id': person_id,
            'object_class': object_data['class_name'],
            'description': description,
            'snapshot_path': snapshot_path,
            'duration': duration,
            'confidence': confidence,
            'previous_owner': previous_owner
        }
        
        return alert
    
    def _generate_description(self, alert_type, obj_id, person_id, obj_data, duration, previous_owner):
        obj_name = obj_data['class_name']
        
        descriptions = {
            'theft': f"🚨 THEFT ALERT: Person #{person_id} took {obj_name} (ID: {obj_id}) belonging to Person #{previous_owner}!",
            'potential_theft': f"⚠️ SUSPICIOUS: {obj_name.capitalize()} (ID: {obj_id}) picked up {duration:.1f}s after Person #{person_id} left it.",
            'suspicious_retrieval': f"⚠️ WARNING: {obj_name.capitalize()} (ID: {obj_id}) unattended for {duration:.1f}s, owned by Person #{person_id}.",
            'abandonment': f"📦 ABANDONMENT: Person #{person_id} left {obj_name} (ID: {obj_id}) for {duration:.1f}s.",
            'lost_object': f"🔍 LOST: {obj_name.capitalize()} (ID: {obj_id}) abandoned {duration:.1f}s. Owner: Person #{person_id}."
        }
        
        return descriptions.get(alert_type, f"Alert for {obj_name}")
    
    def _save_snapshot(self, frame, bbox, alert_type, frame_number):
        snapshot = frame.copy()
        x1, y1, x2, y2 = bbox
        
        colors = {
            'theft': (0, 0, 255),
            'potential_theft': (0, 165, 255),
            'suspicious_retrieval': (0, 255, 255),
            'abandonment': (0, 255, 0),
            'lost_object': (255, 0, 0)
        }
        color = colors.get(alert_type, (255, 255, 255))
        
        cv2.rectangle(snapshot, (x1, y1), (x2, y2), color, 3)
        cv2.putText(snapshot, alert_type.upper(), (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        filename = f"{alert_type}_{frame_number}_{int(time.time())}.jpg"
        filepath = os.path.join(config.SNAPSHOT_DIR, filename)
        cv2.imwrite(filepath, snapshot)
        
        return filepath
    
    def _is_duplicate_alert(self, obj_id, alert_type, current_frame, window=30):
        for alert in reversed(self.alerts[-10:]):
            if (alert['object_id'] == obj_id and 
                alert['type'] == alert_type and 
                current_frame - alert['frame_number'] < window):
                return True
        return False
    
    def get_summary_report(self):
        if not self.alerts:
            return "No incidents detected."
        
        summary = f"📊 **Incident Summary Report**\n\n"
        summary += f"Total Alerts: {len(self.alerts)}\n"
        summary += f"- Theft Detected: {self.alert_count['theft']}\n"
        summary += f"- Suspicious Activity: {self.alert_count['suspicious']}\n"
        summary += f"- Abandonments: {self.alert_count['abandonment']}\n"
        summary += f"- Lost Objects: {self.alert_count['lost']}\n\n"
        
        summary += "**Recent Incidents:**\n"
        for alert in self.alerts[-5:]:
            summary += f"- [{alert['timestamp']}] {alert['description']}\n"
        
        return summary
