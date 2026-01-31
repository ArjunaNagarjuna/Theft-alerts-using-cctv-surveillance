import streamlit as st
import cv2
import tempfile
import pandas as pd
import os
import time

from detector import ObjectDetector
from tracker import SimpleTracker
from ownership_manager import OwnershipManager
from alert_system import AlertSystem
import config

st.set_page_config(
    page_title="Theft & Abandonment Detection",
    page_icon="🔒",
    layout="wide"
)

if 'processing' not in st.session_state:
    st.session_state.processing = False

def visualize_tracking(frame, tracked_objects, alerts):
    annotated = frame.copy()
    
    for obj_id, obj_data in tracked_objects.items():
        x1, y1, x2, y2 = obj_data['bbox']
        class_name = obj_data['class_name']
        
        color = (0, 255, 0) if obj_data['class_id'] == 0 else (255, 0, 0)
        
        for alert in alerts:
            if alert['object_id'] == obj_id:
                color = (0, 0, 255)
        
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} ID:{obj_id}"
        cv2.putText(annotated, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return annotated

def process_video(video_path, progress_bar, status_text):
    detector = ObjectDetector('yolov8n.pt')
    tracker = SimpleTracker()
    ownership_mgr = OwnershipManager()
    alert_system = AlertSystem(ownership_mgr)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_skip = max(1, fps // config.TARGET_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = os.path.join(config.OUTPUT_DIR, f"processed_{int(time.time())}.mp4")
    out = cv2.VideoWriter(output_path, fourcc, config.TARGET_FPS, 
                         (int(cap.get(3)), int(cap.get(4))))
    
    frame_number = 0
    processed_frames = 0
    all_alerts = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_number % frame_skip == 0:
            detections = detector.detect_objects(frame)
            tracked_objects = tracker.update(detections)
            ownership_mgr.associate_objects_to_people(tracked_objects)
            current_alerts = alert_system.check_alerts(tracked_objects, frame, frame_number)
            all_alerts.extend(current_alerts)
            
            annotated_frame = visualize_tracking(frame, tracked_objects, current_alerts)
            out.write(annotated_frame)
            
            processed_frames += 1
            progress = min(frame_number / total_frames, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Frame {frame_number}/{total_frames} | Alerts: {len(all_alerts)}")
        
        frame_number += 1
    
    cap.release()
    out.release()
    
    return output_path, all_alerts, alert_system

# UI
st.title("🔒 Theft & Abandonment Detection System")
st.markdown("**AI-powered surveillance with audio alerts and behavioral analysis**")

with st.sidebar:
    st.header("⚙️ Configuration")
    st.subheader("🔊 Audio Alerts")
    enable_audio = st.checkbox("Enable Sound Alerts", value=True)
    
    st.subheader("Detection Settings")
    conf_threshold = st.slider("Confidence", 0.1, 1.0, 0.5)
    
    st.subheader("Alert Thresholds (sec)")
    st.write(f"Theft: < {config.THEFT_THRESHOLD}s")
    st.write(f"Suspicious: {config.SUSPICIOUS_THRESHOLD}s")
    st.write(f"Lost: > {config.LOST_THRESHOLD}s")

tab1, tab2, tab3 = st.tabs(["📹 Upload & Process", "📊 Alerts", "📈 Analytics"])

with tab1:
    st.header("Video Upload")
    
    uploaded_file = st.file_uploader("Upload surveillance video", 
                                     type=config.VIDEO_FORMATS)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if uploaded_file and st.button("🚀 Start Processing", type="primary"):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = tmp_file.name
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("Processing..."):
                try:
                    output_path, alerts, alert_system = process_video(
                        video_path, progress_bar, status_text
                    )
                    
                    st.success(f"✅ Complete! {len(alerts)} alerts detected.")
                    st.session_state.alerts = alerts
                    st.session_state.alert_system = alert_system
                    st.session_state.output_video = output_path
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.exception(e)
    
    with col2:
        if 'output_video' in st.session_state:
            st.video(st.session_state.output_video)

with tab2:
    st.header("Alert Dashboard")
    
    if 'alerts' in st.session_state and st.session_state.alerts:
        alerts = st.session_state.alerts
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🚨 Total", len(alerts))
        col2.metric("🔴 Theft", sum(1 for a in alerts if 'theft' in a['type']))
        col3.metric("🟡 Suspicious", sum(1 for a in alerts if 'suspicious' in a['type']))
        col4.metric("🟢 Abandonment", sum(1 for a in alerts if 'abandonment' in a['type']))
        
        st.subheader("📝 Incident Report")
        if 'alert_system' in st.session_state:
            st.info(st.session_state.alert_system.get_summary_report())
        
        st.subheader("🔍 Alert Details")
        alert_df = pd.DataFrame(alerts)
        st.dataframe(alert_df[['timestamp', 'type', 'description', 'confidence']], 
                    use_container_width=True)
        
        st.subheader("📸 Snapshots")
        cols = st.columns(3)
        for idx, alert in enumerate(alerts[:6]):
            with cols[idx % 3]:
                if os.path.exists(alert['snapshot_path']):
                    st.image(alert['snapshot_path'], caption=alert['description'][:40])
    else:
        st.info("Upload and process a video to see alerts")

with tab3:
    st.header("Analytics")
    
    if 'alerts' in st.session_state and st.session_state.alerts:
        alert_df = pd.DataFrame(st.session_state.alerts)
        
        st.subheader("Alert Distribution")
        type_counts = alert_df['type'].value_counts()
        st.bar_chart(type_counts)
        
        st.subheader("Confidence Scores")
        st.line_chart(alert_df['confidence'])
    else:
        st.info("No data available")

st.markdown("---")
st.markdown("**Powered by YOLOv8n Pretrained Model | Audio Alerts Enabled**")
