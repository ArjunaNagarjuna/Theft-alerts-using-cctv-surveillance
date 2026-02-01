🛡️ Theft Alerts Using CCTV Surveillance

An AI-based CCTV surveillance system that detects theft by tracking people and their associated objects in video footage. The system identifies unauthorized object transfers and generates alerts using real-time computer vision techniques.

📌 Project Overview

This project analyzes CCTV video streams to detect theft scenarios by monitoring interactions between people and objects. It uses object detection, multi-object tracking, and ownership analysis to identify suspicious activities. When an object is taken by a person other than its registered owner, the system raises a theft alert and logs the event for review.

🧠 Methodology (How the System Works)

The complete workflow of the system is as follows:

1️⃣ Video Input

The system accepts CCTV footage as input (uploaded video or live camera feed).

Video frames are extracted sequentially for processing.

2️⃣ Person and Object Detection

Each frame is processed using a deep learning-based object detection model (YOLOv8).

The model detects persons and objects such as bags, laptops, phones, etc.

Bounding boxes are generated for all detected entities.

3️⃣ Multi-Object Tracking

A tracking algorithm (such as Deep SORT / ByteTrack) assigns unique IDs to each detected person and object.

These IDs are maintained across frames to track movement and interactions over time.

4️⃣ Group Registration (Ownership Assignment)

During the initial setup, objects are linked to their respective owners.

This ownership mapping helps the system understand which person owns which object.

Ownership information is stored internally for monitoring.

5️⃣ Interaction and Movement Analysis

The system continuously measures the distance between objects and their registered owners.

It observes:

Object movement

Owner proximity

Interaction with other persons

6️⃣ Theft Detection Logic

If an object is moved away from its owner or handled by another person, the system evaluates the situation.

Theft is detected when:

An object is taken by a non-owner

Ownership rules are violated for a defined duration

Upon detection, the system flags the event as a theft.

7️⃣ Alert Generation

When a theft is detected:

A visual alert is shown in the processed video

An optional audio alert is triggered

Alerts are generated only for critical events to avoid false positives.

8️⃣ Event Logging

All theft-related events are logged with:

Timestamp

Person ID

Object ID

Event description

Logs are displayed in the dashboard and can be stored for future analysis.

9️⃣ Visualization Dashboard

A Streamlit-based dashboard provides:

Video upload interface

Real-time processing view

Theft alert summary

Event and transfer logs

This makes the system interactive and user-friendly.

⚙️ Technology Stack

Programming Language: Python

Object Detection: YOLOv8

Object Tracking: Deep SORT / ByteTrack

Computer Vision: OpenCV

Dashboard: Streamlit

Alert System: Audio & visual alerts

Concepts: CCTV Surveillance, Computer Vision, Object Ownership Analysis

🚀 How to Run the Project
# Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py


Upload a CCTV video through the dashboard and start the detection process.
