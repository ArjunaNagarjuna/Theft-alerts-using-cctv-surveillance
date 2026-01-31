import cv2
import os

# Create output folder
os.makedirs("extracted_frames", exist_ok=True)

# Your video path - UPDATE THIS
video_path = "training_video.mp4"

# Check if file exists
if not os.path.exists(video_path):
    print(f"❌ Error: Video file not found at '{video_path}'")
    print(f"📁 Current directory: {os.getcwd()}")
    print(f"📋 Files in current directory:")
    for f in os.listdir():
        print(f"  - {f}")
    exit()

# Open video
print(f"📹 Opening video: {video_path}")
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print(f"❌ Error: Could not open video file")
    print(f"💡 Try these solutions:")
    print(f"   1. Make sure the video file is in the same folder as this script")
    print(f"   2. Copy the video to this folder: {os.getcwd()}")
    print(f"   3. Or update the video_path in the script to the full path")
    exit()

# Get video info
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Handle case where fps is 0
if fps == 0:
    print("⚠️ Warning: Could not detect FPS, using default 30")
    fps = 30

duration = total_frames / fps if fps > 0 else 0

print(f"\n📊 Video Information:")
print(f"  Total Frames: {total_frames}")
print(f"  FPS: {fps}")
print(f"  Duration: {duration:.2f} seconds")
print(f"\n🔄 Extracting frames...\n")

frame_count = 0
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Extract every 10th frame
    if frame_count % 10 == 0:
        frame_filename = f"extracted_frames/frame_{saved_count:04d}.jpg"
        success = cv2.imwrite(frame_filename, frame)
        
        if success:
            saved_count += 1
            print(f"✓ Saved: {frame_filename}")
        else:
            print(f"❌ Failed to save: {frame_filename}")
    
    frame_count += 1

cap.release()

print(f"\n✅ Successfully extracted {saved_count} frames!")
print(f"📁 Frames saved in: extracted_frames/")
print(f"\n📋 Next Steps:")
print(f"1. Go to https://roboflow.com")
print(f"2. Create account and new project")
print(f"3. Upload all {saved_count} images from 'extracted_frames' folder")
print(f"4. Annotate people and objects")
print(f"5. Generate dataset with augmentation")
print(f"6. Download in YOLOv8 format")
