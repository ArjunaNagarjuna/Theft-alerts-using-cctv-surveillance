import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(page_title="Admin Dashboard | AI Theft Detection", layout="wide", page_icon="📊")

# --- HEADER STYLING ---
st.markdown("""
    <style>
        h1 {
            text-align: center;
            color: #00bfa6;
        }
        .metric-card {
            text-align: center;
            padding: 20px;
            border-radius: 15px;
            background-color: #111827;
            color: white;
            box-shadow: 0px 2px 10px rgba(0,0,0,0.4);
        }
        .stButton>button {
            border-radius: 10px;
            height: 42px;
            background-color: #00bfa6 !important;
            color: white !important;
            font-weight: bold !important;
            border: none;
        }
        .stDataFrame {
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>📊 AI Theft Detection - Admin Dashboard</h1>", unsafe_allow_html=True)
st.markdown("#### 🔐 Monitor users, view activities, and analyze system events in real-time.")

# --- CONNECT DATABASE ---
conn = sqlite3.connect("users.db")
cur = conn.cursor()

# --- ENSURE TABLES EXIST ---
cur.execute("""
CREATE TABLE IF NOT EXISTS activity_logs (
    username TEXT,
    action TEXT,
    timestamp TEXT
)
""")
conn.commit()

# --- FETCH DATA ---
users = pd.read_sql_query("SELECT username FROM users", conn)
logs = pd.read_sql_query("SELECT * FROM activity_logs ORDER BY timestamp DESC", conn)

# --- 1️⃣ METRIC OVERVIEW ---
st.markdown("### 📈 System Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("👥 Total Users", len(users))
col2.metric("🕒 Total Activities", len(logs))
col3.metric("📅 Logins Today", len(logs[logs['action'].str.contains('Login', na=False)]) if len(logs) else 0)
col4.metric("🚪 Logouts Today", len(logs[logs['action'].str.contains('Logout', na=False)]) if len(logs) else 0)

st.markdown("---")

# --- 2️⃣ REGISTERED USERS SECTION ---
st.subheader("👤 Registered Users")

if not users.empty:
    st.dataframe(users, use_container_width=True)

    # 📥 Export button for users
    user_csv = users.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Export User List (CSV)",
        data=user_csv,
        file_name="registered_users.csv",
        mime="text/csv",
        use_container_width=True,
    )
else:
    st.info("No registered users found yet.")

st.markdown("---")

# --- 3️⃣ ACTIVITY LOGS SECTION ---
st.subheader("🕘 Activity Logs")

if not logs.empty:
    st.dataframe(logs, use_container_width=True)

    # 📊 Activity Trend (Last 7 Days)
    st.markdown("#### 📊 Activity Trend (Last 7 Days)")
    logs["timestamp"] = pd.to_datetime(logs["timestamp"], errors="coerce")
    recent_logs = logs[logs["timestamp"] >= datetime.now() - pd.Timedelta(days=7)]

    if not recent_logs.empty:
        trend = recent_logs["timestamp"].dt.date.value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(trend.index, trend.values, marker="o", color="#00bfa6")
        ax.set_title("User Activity Over the Last 7 Days", fontsize=12)
        ax.set_xlabel("Date")
        ax.set_ylabel("Number of Actions")
        ax.grid(True, linestyle="--", alpha=0.4)
        st.pyplot(fig)
    else:
        st.info("No activity recorded in the last 7 days.")

    # 📥 Export logs button
    logs_csv = logs.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Export Activity Logs (CSV)",
        data=logs_csv,
        file_name="activity_logs.csv",
        mime="text/csv",
        use_container_width=True,
    )
else:
    st.info("No user activity logs recorded yet.")

st.markdown("---")

# --- 4️⃣ ADMIN CONTROLS ---
st.subheader("⚙️ Administrative Controls")
st.markdown("Use the options below to maintain or reset system data safely.")

col1, col2 = st.columns(2)
with col1:
    if st.button("🧹 Clear Logs"):
        with st.spinner("Deleting logs..."):
            cur.execute("DELETE FROM activity_logs")
            conn.commit()
        st.success("✅ All activity logs cleared successfully!")

with col2:
    if st.button("🔄 Refresh Dashboard"):
        st.rerun()

st.markdown("---")

# --- 5️⃣ FOOTER ---
st.markdown("""
    <div style='text-align:center; color:gray; font-size:14px;'>
        © 2025 <b>AI Theft Detection System</b> | Admin Control Panel <br>
        Developed with ❤️ using <b>Streamlit</b>
    </div>
""", unsafe_allow_html=True)
