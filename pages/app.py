import streamlit as st
import cv2
import tempfile
import os
import numpy as np
import pandas as pd
import time
from collections import Counter
from datetime import datetime
import torch
import warnings
from fpdf import FPDF
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

# ✅ Load OpenAI key
env_path = find_dotenv()
load_dotenv(env_path)
openai_api_key = os.getenv("MY_AI_KEY")

if not openai_api_key:
    with st.sidebar.expander("⚠️ System Notice", expanded=False):
        st.warning("OpenAI API key is missing. Some AI features may not work.")
# ✅ Else — don’t show any success message (keep it silent)

# Initialize OpenAI client

# --- AUTHENTICATION CHECK ---
if "authenticated" not in st.session_state or not st.session_state.authenticated:
    st.warning("⚠️ Please login first!")
    st.switch_page("login.py")
    st.stop()

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Theft Detection", layout="wide")
# --- REMEMBER LAST THEME ---
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "🌙 Dark Mode"


# --- THEME TOGGLE (Dark/Light Mode) ---
with st.sidebar:
    st.markdown("### 🎨 Theme Settings")
    theme_mode = st.radio(
        "Choose Theme",
        ["🌙 Dark Mode", "☀️ Light Mode"],
        horizontal=False,
        index=0 if st.session_state.theme_mode == "🌙 Dark Mode" else 1
    )
    st.session_state.theme_mode = theme_mode


# --- APPLY THEME STYLES ---
if theme_mode == "🌙 Dark Mode":
    st.markdown("""
        <style>
        body, .stApp { background-color: #0f2027; color: white; }
        .stMetricValue, .stMarkdown, .stDataFrame { color: white !important; }
        .stButton>button { background-color: #00bfa6 !important; color: white !important; }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        body, .stApp { background-color: #f8fafc; color: #0f2027; }
        .stMetricValue, .stMarkdown, .stDataFrame { color: #0f2027 !important; }
        .stButton>button { background-color: #00796b !important; color: white !important; }
        </style>
    """, unsafe_allow_html=True)

# --- LOAD CUSTOM STYLE ---
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- LOGOUT BUTTON ---
# --- LOGOUT BUTTON ---
#st.sidebar.button("🚪 Logout", on_click=lambda: st.session_state.update({"authenticated": False}))
#if not st.session_state.get("authenticated"):
   # st.switch_page("login.py")
   # st.stop()
# --- SIDEBAR USER INFO + LOGOUT ---
#with st.sidebar:
    # Show who is logged in
   # if "current_user" in st.session_state and st.session_state.current_user:
        #st.markdown(f"👤 **Logged in as:** `{st.session_state.current_user}`")

    # Logout button
    #if st.button("🚪 Logout"):
        #st.session_state.authenticated = False
      #  st.session_state.current_user = None
       # st.switch_page("login.py")

    #st.markdown("---")
with st.sidebar:
    # Show who is logged in
    if "current_user" in st.session_state and st.session_state.current_user:
        st.markdown(f"👤 **Logged in as:** `{st.session_state.current_user}`")

    # Logout button
    if st.button("🚪 Logout"):
        # Log logout activity
        import sqlite3
        conn3 = sqlite3.connect("users.db")
        cur3 = conn3.cursor()
        cur3.execute("""
            CREATE TABLE IF NOT EXISTS activity_logs (
                username TEXT,
                action TEXT,
                timestamp TEXT
            )
        """)
        conn3.commit()
        cur3.execute("INSERT INTO activity_logs VALUES (?, ?, ?)",
                     (st.session_state.current_user, "Logout", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn3.commit()
        conn3.close()

        st.session_state.authenticated = False
        st.session_state.current_user = None
        st.switch_page("login.py")

    # Divider
    st.markdown("---")

    # --- Admin Dashboard Button ---
    if st.button("📊 Go to Admin Dashboard"):
        st.switch_page("pages/admin_dashboard.py")

    st.markdown("---")


# --- WARNINGS ---
warnings.filterwarnings("ignore")

# --- IMPORTS ---
if hasattr(torch.serialization, 'add_safe_globals'):
    try:
        from ultralytics.nn.tasks import DetectionModel
        torch.serialization.add_safe_globals([DetectionModel])
    except:
        pass
from ultralytics import YOLO

# --- AUDIO ---
try:
    import pygame
    AUDIO_ENABLED = True
except:
    AUDIO_ENABLED = False

# --- DIRECTORIES ---
for d in ["output", "audio"]:
    os.makedirs(d, exist_ok=True)

# --- OBJECT CLASSES ---
OBJECTS = {
    24: 'backpack', 25: 'umbrella', 26: 'handbag', 28: 'suitcase',
    39: 'bottle', 63: 'laptop', 64: 'mouse', 65: 'remote',
    66: 'keyboard', 67: 'cell phone', 73: 'book', 76: 'scissors'
}

# --- AUDIO SYSTEM CLASS ---
class AudioSystem:
    def __init__(self):
        self.ready = False
        if AUDIO_ENABLED:
            try:
                pygame.mixer.init()
                import wave, struct, math
                with wave.open("audio/alarm.wav", 'w') as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(44100)
                    for i in range(44100):
                        w.writeframesraw(struct.pack('<h', int(32767 * math.sin(2 * math.pi * 1200 * i / 44100))))
                self.sound = pygame.mixer.Sound("audio/alarm.wav")
                self.ready = True
            except:
                pass

    def play(self):
        if self.ready:
            for _ in range(3):
                self.sound.play()
                pygame.time.wait(1000)

# --- TRACKER CLASS ---
class Tracker:
    def __init__(self):
        self.next_id = 0
        self.tracks = {}
        self.lost = {}
        self.all_people = set()
        self.all_objects = set()

    def update(self, dets):
        if not dets:
            for tid in list(self.lost.keys()):
                self.lost[tid] += 1
                if self.lost[tid] > 50:
                    if tid in self.tracks:
                        del self.tracks[tid]
                    del self.lost[tid]
            return self.tracks

        if not self.tracks:
            for d in dets:
                tid = self.next_id
                self.tracks[tid] = {
                    'bbox': d['bbox'], 'class_id': d['class_id'],
                    'class_name': d['class_name'], 'center': d['center'],
                    'votes': [d['class_name']], 'frames': 1, 'ok': False
                }
                self.lost[tid] = 0
                self.next_id += 1
            return self.tracks

        tids = list(self.tracks.keys())
        dists = np.zeros((len(tids), len(dets)))

        for i, tid in enumerate(tids):
            tc = self.tracks[tid]['center']
            for j, d in enumerate(dets):
                dc = d['center']
                dist = np.sqrt((tc[0] - dc[0])**2 + (tc[1] - dc[1])**2)
                penalty = 0 if self.tracks[tid]['class_id'] == d['class_id'] else 2000
                dists[i, j] = dist + penalty

        matched_t = set()
        matched_d = set()

        for _ in range(min(len(tids), len(dets))):
            if len(matched_t) >= len(tids) or len(matched_d) >= len(dets):
                break
            min_i, min_j = -1, -1
            min_val = float('inf')
            for i in range(len(tids)):
                if i in matched_t:
                    continue
                for j in range(len(dets)):
                    if j in matched_d:
                        continue
                    if dists[i, j] < min_val:
                        min_val = dists[i, j]
                        min_i, min_j = i, j

            if min_val < 150:
                matched_t.add(min_i)
                matched_d.add(min_j)
                tid = tids[min_i]
                d = dets[min_j]

                self.tracks[tid]['bbox'] = d['bbox']
                self.tracks[tid]['center'] = d['center']
                self.tracks[tid]['votes'].append(d['class_name'])
                self.tracks[tid]['frames'] += 1
                self.lost[tid] = 0

                if len(self.tracks[tid]['votes']) >= 20:
                    most_common = Counter(self.tracks[tid]['votes']).most_common(1)[0][0]
                    self.tracks[tid]['class_name'] = most_common

                if self.tracks[tid]['frames'] >= 15:
                    self.tracks[tid]['ok'] = True
                    if d['class_id'] == 0:
                        self.all_people.add(tid)
                    elif d['class_id'] in OBJECTS:
                        self.all_objects.add(tid)

        for i, tid in enumerate(tids):
            if i not in matched_t:
                self.lost[tid] += 1
                if self.lost[tid] > 50:
                    del self.tracks[tid]
                    del self.lost[tid]

        for j in range(len(dets)):
            if j not in matched_d:
                tid = self.next_id
                d = dets[j]
                self.tracks[tid] = {
                    'bbox': d['bbox'], 'class_id': d['class_id'],
                    'class_name': d['class_name'], 'center': d['center'],
                    'votes': [d['class_name']], 'frames': 1, 'ok': False
                }
                self.lost[tid] = 0
                self.next_id += 1

        return self.tracks

# --- OWNERSHIP, ALERTS (same as your original code) ---
# keep your full ownership, alerts, and UI section as-is from your working code
# (not rewriting here to save space)

# You can paste all your previous "Ownership", "Alerts", and UI logic below this point


# Ownership
class Ownership:
    def __init__(self):
        self.map = {}
        self.transfers = []
        
    def update(self, tracks):
        people = {k: v for k, v in tracks.items() if v['class_id'] == 0 and v['ok']}
        objects = {k: v for k, v in tracks.items() if v['class_id'] in OBJECTS and v['ok']}
        
        for oid, obj in objects.items():
            best_p = None
            best_s = 0
            
            for pid, person in people.items():
                # IoU + distance
                pb, ob = person['bbox'], obj['bbox']
                xi = max(0, min(pb[2], ob[2]) - max(pb[0], ob[0]))
                yi = max(0, min(pb[3], ob[3]) - max(pb[1], ob[1]))
                
                if xi * yi > 0:
                    score = 1.0
                else:
                    pc, oc = person['center'], obj['center']
                    dist = np.sqrt((pc[0] - oc[0])**2 + (pc[1] - oc[1])**2)
                    score = 1 / (1 + dist / 60)
                
                if score > best_s and score > 0.5:
                    best_s = score
                    best_p = pid
            
            if best_p:
                if oid not in self.map:
                    self.map[oid] = {'orig': best_p, 'curr': best_p, 'frames': 0, 'theft': False}
                else:
                    if best_p != self.map[oid]['curr']:
                        self.map[oid]['frames'] += 1
                        if self.map[oid]['frames'] >= 10:
                            orig = self.map[oid]['orig']
                            if best_p != orig and not self.map[oid]['theft']:
                                self.map[oid]['curr'] = best_p
                                self.map[oid]['theft'] = True
                                self.transfers.append({'oid': oid, 'orig': orig, 'thief': best_p})
                    else:
                        self.map[oid]['frames'] = 0

# Alerts
class Alerts:
    def __init__(self, audio):
        self.audio = audio
        self.alerts = []
        self.done = set()
        
    def check(self, own, tracks, frame_n, fps):
        new = []
        t = frame_n / fps
        
        for oid, data in own.map.items():
            if data['theft'] and oid not in self.done and oid in tracks:
                orig = data['orig']
                thief = data['curr']
                name = tracks[oid]['class_name']
                
                alert = {
                    'time': f"{t:.1f}s",
                    'type': 'theft',
                    'desc': f"Person #{thief} took {name} (ID:{oid}) from Person #{orig}"
                }
                
                new.append(alert)
                self.alerts.append(alert)
                self.done.add(oid)
                self.audio.play()
                print(f"\n🚨 THEFT at {t:.1f}s: {alert['desc']}\n")
        
        return new


# --- THEFT REPORT GENERATOR ---
def generate_theft_report(alerts):
    """Generate a theft report as a PDF file."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "AI Theft Detection Report", ln=True, align="C")
    pdf.ln(10)

    # Header Info
    pdf.set_font("Arial", "", 12)
    pdf.cell(200, 8, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(200, 8, f"Total Thefts Detected: {len(alerts)}", ln=True)
    pdf.ln(8)

    if not alerts:
        pdf.cell(200, 10, "✅ No thefts detected in this session.", ln=True)
    else:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(40, 8, "Time", 1)
        pdf.cell(150, 8, "Description", 1)
        pdf.ln()

        pdf.set_font("Arial", "", 12)
        for alert in alerts:
            pdf.cell(40, 8, alert["time"], 1)
            pdf.cell(150, 8, alert["desc"], 1)
            pdf.ln()

    filename = f"output/Theft_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(filename)
    return filename
##
##
## new function to generate AI summary
# =========================================
# 🤖 AI SUMMARY + CHAT ASSISTANT SECTION


# ==========================================

import streamlit as st
import os
from openai import OpenAI

# ========================
# 🌟 GLOBAL PAGE STYLING
# ========================
st.markdown("""
    <style>
        /* Page & section padding */
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 1.5rem;
        }

        /* Main section headings */
        h2, h3 {
            color: #00bfa6;
            font-weight: 700;
        }

        /* Buttons */
        .stButton>button {
            background-color: #00bfa6 !important;
            color: white !important;
            font-weight: 600 !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 8px 20px !important;
            width: auto !important;
            height: 42px !important;
            text-align: left !important;
        }
        .stButton {
            display: flex;
            justify-content: flex-start !important; /* left-align */
        }

        /* Text input box */
        .stTextInput>div>div>input {
            border-radius: 8px !important;
            padding: 10px !important;
        }

        /* Chat bubbles */
        .user-bubble {
            background-color: #111827;
            color: white;
            padding: 10px 15px;
            border-radius: 10px;
            margin-bottom: 5px;
        }
        .ai-bubble {
            background-color: #00bfa620;
            color: white;
            padding: 10px 15px;
            border-radius: 10px;
            margin-bottom: 10px;
        }

        /* Card styling */
        .card {
            background-color: #1b1f2a;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# ========================
import streamlit as st
import os
from openai import OpenAI


# ========================
# 🧠 1️⃣ GENERATE AI SUMMARY
# ========================
def generate_ai_summary(alerts):
    """Generate a professional AI-written theft summary."""
    api_key = os.getenv("MY_AI_KEY")
    if not api_key:
        return "⚠️ No API key found in .env file."

    try:
        client = OpenAI(api_key=api_key)
        if not alerts:
            prompt = "Generate a short, professional report stating that no theft incidents were detected."
        else:
            theft_details = "\n".join([f"- {a['desc']} (at {a['time']})" for a in alerts])
            prompt = (
                "You are an AI Security Analyst. Write a detailed and formal summary for the following theft events:\n"
                f"{theft_details}\n\n"
                "Structure your response with:\n"
                "1️⃣ Incident Overview\n2️⃣ Analysis\n3️⃣ Recommendations\n"
                "Keep it factual, concise, and professional."
            )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional AI security analyst."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.6,
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"⚠️ AI Summary could not be generated: {str(e)}"


# ========================
# 💬 2️⃣ AI CHAT ASSISTANT
# ========================
def ai_chat_section(alerts):
    """Interactive AI chat like a mini ChatGPT."""
    st.markdown("## 💬 AI Theft Intelligence Assistant")

    # Chat memory
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "clear_flag" not in st.session_state:
        st.session_state.clear_flag = False

    # ✅ Safely clear previous input on rerun
    if st.session_state.clear_flag:
        st.session_state.user_input = ""
        st.session_state.clear_flag = False

    # Input field
    st.markdown("#### 💭 Ask AI about theft events:")
    query = st.text_input(
        "",
        placeholder="e.g. Who committed the most thefts? or What happened in the video?",
        key="user_input",
    )

    # Left-aligned Ask button
    ask_btn = st.button("🧠 Ask AI")

    if ask_btn:
        api_key = os.getenv("MY_AI_KEY")
        if not api_key:
            st.error("⚠️ Missing API key. Please check your .env file.")
            return

        if not alerts:
            st.warning("⚠️ No theft data available to analyze yet.")
            return

        if not query.strip():
            st.warning("✏️ Please enter a valid question.")
            return

        theft_info = "\n".join([f"- {a['desc']} (at {a['time']})" for a in alerts])
        full_prompt = (
            f"You are a theft analysis AI assistant.\n"
            f"Here are recent theft events:\n{theft_info}\n\n"
            f"User's Question: {query}\n"
            "Answer clearly, professionally, and with analytical insight."
        )

        try:
            client = OpenAI(api_key=api_key)
            with st.spinner("🤖 Thinking..."):
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an intelligent assistant specializing in theft detection."},
                        {"role": "user", "content": full_prompt},
                    ],
                    temperature=0.7,
                )

            ai_answer = response.choices[0].message.content
            st.session_state.chat_history.append({"user": query, "ai": ai_answer})

            # ✅ Set flag to clear the input safely after next rerun
            st.session_state.clear_flag = True
            st.rerun()

        except Exception as e:
            st.error(f"⚠️ AI Chat could not process your query: {str(e)}")

    # --- Collapsible Chat History ---
    if st.session_state.chat_history:
        with st.expander("🕘 Chat History", expanded=False):
            st.markdown(
                """
                <style>
                    .user-bubble {
                        background-color: #1e293b;
                        padding: 10px 15px;
                        border-radius: 10px;
                        color: #f1f5f9;
                        margin-bottom: 6px;
                    }
                    .ai-bubble {
                        background-color: #0f766e;
                        padding: 10px 15px;
                        border-radius: 10px;
                        color: #f1f5f9;
                        margin-bottom: 10px;
                    }
                    .stButton>button {
                        background-color: #00bfa6 !important;
                        color: white !important;
                        font-weight: 600 !important;
                        border-radius: 8px !important;
                        border: none !important;
                        padding: 6px 18px !important;
                    }
                </style>
                """,
                unsafe_allow_html=True,
            )

            for chat in reversed(st.session_state.chat_history):
                st.markdown(f"<div class='user-bubble'>🧑 <b>You:</b> {chat['user']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='ai-bubble'>🤖 <b>AI:</b> {chat['ai']}</div>", unsafe_allow_html=True)

            if st.button("🧹 Clear Chat History"):
                st.session_state.chat_history = []
                st.success("✅ Chat history cleared.")



# ========================
# 🤖 3️⃣ COMBINED SECTION
# ========================
def show_ai_summary_section(alerts):
    """Unified professional UI for Summary + Chat with collapsible expanders."""
    st.markdown("## 🤖 AI Summary & Chat Assistant")

    # --- Collapsible Summary Section ---
    with st.expander("🧾 Generate AI Theft Summary", expanded=False):
        st.markdown("Easily generate a professional summary of detected theft incidents using GPT intelligence.")
        gen_btn = st.button("🚀 Generate AI Summary")
        if gen_btn:
            with st.spinner("🧠 Generating AI Summary..."):
                ai_summary = generate_ai_summary(alerts)
            st.success("✅ AI Summary Generated Successfully!")
            st.text_area("📋 AI Summary Report", ai_summary, height=250)
            st.download_button(
                label="⬇️ Download AI Summary",
                data=ai_summary,
                file_name="AI_Summary_Report.txt",
                mime="text/plain",
            )

    st.markdown("---")
    ai_chat_section(alerts)




# Init
if 'loaded' not in st.session_state:
    st.session_state.loaded = False
if 'audio' not in st.session_state:
    st.session_state.audio = AudioSystem()

st.title("🔒 Theft Detection System")
st.markdown("**Accurate Detection • Live Processing • Real Alerts**")

with st.sidebar:
    st.header("Settings")
    conf = st.slider("Confidence", 0.3, 0.9, 0.55)
    st.markdown("---")
    st.write("🔊 Audio:", "✅" if st.session_state.audio.ready else "❌")

# Load model
if not st.session_state.loaded:
    with st.spinner("Loading..."):
        try:
            st.session_state.model = YOLO('yolov8n.pt')
            st.session_state.loaded = True
            st.success("✅ Ready!")
        except Exception as e:
            st.error(str(e))
            st.stop()

model = st.session_state.model

# Upload
st.header("Upload Video")
file = st.file_uploader("Choose video", type=['mp4', 'avi', 'mov'])

if file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(file.read())
        path = tmp.name
    
    st.success(f"✅ {file.name}")
    
    if st.button("🚀 Start", type="primary"):
        st.markdown("### Processing...")
        
        tracker = Tracker()
        own = Ownership()
        alerts = Alerts(st.session_state.audio)
        
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        out_path = f"output/out_{int(time.time())}.mp4"
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        vid_ph = st.empty()
        met_ph = st.empty()
        prog = st.progress(0)
        
        fc = 0
        dets_list = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect
            res = model(frame, conf=conf, iou=0.4, verbose=False)
            dets = []
            for r in res:
                for b in r.boxes:
                    cid = int(b.cls[0])
                    x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy())
                    
                    if cid == 0 or cid in OBJECTS:
                        dets.append({
                            'bbox': [x1, y1, x2, y2],
                            'class_id': cid,
                            'class_name': r.names[cid],
                            'center': [(x1 + x2) // 2, (y1 + y2) // 2]
                        })
            
            tracks = tracker.update(dets)
            own.update(tracks)
            curr_alerts = alerts.check(own, tracks, fc, fps)
            
            up, uo = len(tracker.all_people), len(tracker.all_objects)
            
            # Draw
            disp = frame.copy()
            
            for tid, t in tracks.items():
                if not t['ok']:
                    continue
                
                x1, y1, x2, y2 = t['bbox']
                cx, cy = t['center']
                name = t['class_name']
                
                if t['class_id'] == 0:
                    color = (0, 255, 0)
                    label = f"P#{tid}"
                else:
                    color = (255, 150, 0)
                    
                    if tid in own.map:
                        if own.map[tid]['theft']:
                            orig = own.map[tid]['orig']
                            thief = own.map[tid]['curr']
                            label = f"STOLEN {name} #{tid}"
                            color = (0, 0, 255)
                            # RED CIRCLE
                            rad = max(x2 - x1, y2 - y1) // 2 + 25
                            cv2.circle(disp, (cx, cy), rad, (0, 0, 255), 6)
                        else:
                            label = f"{name} #{tid} [P{own.map[tid]['orig']}]"
                    else:
                        label = f"{name} #{tid}"
                
                thick = 5 if color == (0, 0, 255) else 2
                cv2.rectangle(disp, (x1, y1), (x2, y2), color, thick)
                
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(disp, (x1, y1 - lh - 10), (x1 + lw + 10, y1), color, -1)
                cv2.putText(disp, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Alerts overlay
            if curr_alerts:
                for i, a in enumerate(curr_alerts[-2:]):
                    txt = f"🚨 {a['desc']}"
                    y = 40 + i * 50
                    cv2.rectangle(disp, (10, y - 30), (w - 10, y + 10), (0, 0, 0), -1)
                    cv2.putText(disp, txt, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            out.write(disp)
            
            rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
            vid_ph.image(rgb)
            
            with met_ph.container():
                c1, c2, c3 = st.columns(3)
                c1.metric("👥 People", up)
                c2.metric("📦 Objects", uo)
                c3.metric("🚨 Theft", len(alerts.alerts))
            
            for tid, t in tracks.items():
                if t['ok']:
                    dets_list.append({'time': f"{fc/fps:.1f}s", 'id': tid, 'type': t['class_name']})
            
            prog.progress(min(fc / total, 1.0))
            fc += 1
        
        cap.release()
        out.release()
        
        st.success("✅ Done!")
        st.balloons()
        
        st.session_state.vid = out_path
        st.session_state.dets = dets_list
        st.session_state.alerts = alerts.alerts
        st.session_state.up = up
        st.session_state.uo = uo
        st.session_state.trans = own.transfers

# Results
# --- RESULTS SECTION ---
if 'vid' in st.session_state:
    st.markdown("---")
    st.header("Results")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("👥 People", st.session_state.up)
    c2.metric("📦 Objects", st.session_state.uo)
    c3.metric("🚨 Theft", len(st.session_state.alerts))
    
    st.markdown("---")
    st.subheader("🎥 Processed Video")
    st.video(st.session_state.vid)
    
    st.markdown("---")
    
    # --- THEFT ALERTS TABLE ---
    if st.session_state.alerts:
        st.subheader("🚨 Theft Alerts")
        df = pd.DataFrame(st.session_state.alerts)
        st.dataframe(df, use_container_width=True)

        # --- PDF REPORT DOWNLOAD FEATURE ---
        st.markdown("---")
        st.info("📑 Generate an official theft summary report for this session.")
        if st.button("📄 Download Theft Report as PDF"):
            report_path = generate_theft_report(st.session_state.alerts)
            with open(report_path, "rb") as f:
                st.download_button(
                    label="⬇️ Click here to download your report",
                    data=f,
                    file_name=os.path.basename(report_path),
                    mime="application/pdf"
                )
    else:
        st.success("✅ No Theft Detected")

    st.markdown("---")
    #
    #
        # --- AI SUMMARY REPORT (GPT) ---
    show_ai_summary_section(st.session_state.alerts)


    # --- TRANSFER LOGS ---
    if st.session_state.trans:
        st.subheader("🔄 Transfers")
        td = [{'Object': t['oid'], 'From': f"P#{t['orig']}", 'To': f"P#{t['thief']}"} for t in st.session_state.trans]
        st.dataframe(pd.DataFrame(td), use_container_width=True)
    
    st.markdown("---")
    st.subheader("📊 Detections Summary")
    dd = pd.DataFrame(st.session_state.dets).drop_duplicates(subset=['id', 'type'])
    st.dataframe(dd, use_container_width=True)

st.markdown("---")
st.markdown("**🤖 Production-Ready Theft Detection System**")


# In sidebar, after settings
st.markdown("---")
st.subheader("👥 Group Registration")

# Initialize ownership manager if not exists
if 'ownership' not in st.session_state:
    from ownership_manager import OwnershipManager
    st.session_state.ownership = OwnershipManager()

ownership_mgr = st.session_state.ownership

# Registration controls
if st.button("🔵 Start Group Registration"):
    ownership_mgr.group_manager.start_registration()
    st.success("Registration started - Show belongings now")

if ownership_mgr.group_manager.registration_mode:
    st.info("📸 REGISTRATION MODE ACTIVE")
    
    # Manual registration inputs
    with st.form("register_form"):
        st.write("Register Family Members & Objects")
        person_ids = st.text_input("Person IDs (comma-separated)", "1,2,3,4")
        
        st.write("Objects (format: person_id,object_id,object_name)")
        obj_input = st.text_area("Objects", "1,101,backpack\n2,102,phone\n3,103,suitcase")
        
        if st.form_submit_button("Register All"):
            # Register people
            for pid in person_ids.split(','):
                ownership_mgr.group_manager.register_person(int(pid.strip()))
            
            # Register objects
            for line in obj_input.strip().split('\n'):
                if line:
                    parts = line.split(',')
                    ownership_mgr.group_manager.register_object(
                        int(parts[0]), int(parts[1]), parts[2].strip()
                    )
            
            st.success("Items registered!")
    
    if st.button("✅ Complete Registration"):
        group_id = ownership_mgr.group_manager.complete_registration()
        st.success(f"Group #{group_id} created!")
        st.balloons()

# Show registered groups
if len(ownership_mgr.group_manager.groups) > 0:
    with st.expander("📊 Registered Groups"):
        st.text(ownership_mgr.group_manager.get_group_summary())
