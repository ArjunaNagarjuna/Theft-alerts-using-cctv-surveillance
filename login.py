import streamlit as st
import sqlite3
import bcrypt
import random
import string
import os
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Theft Detection Login", layout="centered")

# --- DATABASE CONNECTION ---
conn = sqlite3.connect("users.db")
cur = conn.cursor()

# Create user and reset code tables
cur.execute("""
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT NOT NULL
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS reset_codes (
    username TEXT PRIMARY KEY,
    code TEXT NOT NULL
)
""")

# Create table for user activity logs
cur.execute("""
CREATE TABLE IF NOT EXISTS activity_logs (
    username TEXT,
    action TEXT,
    timestamp TEXT
)
""")

conn.commit()

# --- PAGE STYLE ---
st.markdown("""
    <style>
    /* Remove unwanted dark container spacing below the image */
    div[data-testid="stVerticalBlock"] > div:has(img) + div:empty {
        display: none !important;
        background: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    /* Tighten spacing between image and card */
    div[data-testid="stVerticalBlock"] > div:has(img) {
        margin-bottom: -20px !important;
        padding-bottom: 0 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "reset_step" not in st.session_state:
    st.session_state.reset_step = 1
if "reset_user" not in st.session_state:
    st.session_state.reset_user = None

# --- HEADER ---
st.markdown("<h2 style='text-align:center;'>🔐 AI Theft Detection Login</h2>", unsafe_allow_html=True)
st.markdown("<h4 class='glow-text' style='text-align:center;'>Smart Surveillance • Secure Access</h4>", unsafe_allow_html=True)

mode = st.radio("Select Mode", ["Login", "Register", "Forgot Password"], horizontal=True)

# --- DISPLAY MODE-SPECIFIC IMAGE ---
assets_path = "assets"
if mode == "Login" and os.path.exists(f"{assets_path}/undraw_secure-login_m11a.png"):
    st.image(f"{assets_path}/undraw_secure-login_m11a.png", width=300)
elif mode == "Register" and os.path.exists(f"{assets_path}/undraw_profile-data_xkr9.png"):
    st.image(f"{assets_path}/undraw_profile-data_xkr9.png", width=300)
elif mode == "Forgot Password" and os.path.exists(f"{assets_path}/undraw_forgot-password_nttj.png"):
    st.image(f"{assets_path}/undraw_forgot-password_nttj.png", width=300)

# 🧩 Small spacer
st.markdown("<br>", unsafe_allow_html=True)

# --- LOGIN CARD START ---
st.markdown("<div class='login-card'>", unsafe_allow_html=True)

# ---------------- LOGIN ----------------
if mode == "Login":
    username = st.text_input("👤 Username")
    password = st.text_input("🔑 Password", type="password")

    if st.button("Login"):
        cur.execute("SELECT password FROM users WHERE username=?", (username,))
        result = cur.fetchone()

        # --- SUCCESSFUL LOGIN ---
        if result and bcrypt.checkpw(password.encode(), result[0]):
            st.session_state.authenticated = True
            st.session_state.current_user = username
            st.success(f"✅ Welcome back, {username}!")

            # --- LOG LOGIN ACTIVITY ---
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
                         (username, "Login", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            conn3.commit()
            conn3.close()

            st.switch_page("pages/app.py")

        # --- FAILED LOGIN (BONUS TIP) ---
        else:
            st.error("❌ Invalid username or password.")
            conn3 = sqlite3.connect("users.db")
            cur3 = conn3.cursor()
            cur3.execute("INSERT INTO activity_logs VALUES (?, ?, ?)",
                         (username, "Failed Login", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            conn3.commit()
            conn3.close()

# ---------------- REGISTER ----------------
elif mode == "Register":
    new_user = st.text_input("🆕 Choose Username")
    new_pass = st.text_input("🔑 Choose Password", type="password")
    confirm_pass = st.text_input("🔁 Confirm Password", type="password")

    if st.button("Create Account"):
        if new_user == "" or new_pass == "":
            st.warning("⚠️ Please fill all fields.")
        elif new_pass != confirm_pass:
            st.error("❌ Passwords do not match.")
        else:
            cur.execute("SELECT * FROM users WHERE username=?", (new_user,))
            if cur.fetchone():
                st.error("⚠️ Username already exists.")
            else:
                hashed = bcrypt.hashpw(new_pass.encode(), bcrypt.gensalt())
                cur.execute("INSERT INTO users (username, password) VALUES (?, ?)", (new_user, hashed))
                conn.commit()
                st.success("✅ Account created successfully! Please log in.")

# ---------------- FORGOT PASSWORD ----------------
elif mode == "Forgot Password":
    step = st.session_state.reset_step

    # Step 1: Enter username
    if step == 1:
        uname = st.text_input("👤 Enter your username")
        if st.button("Send Reset Code"):
            cur.execute("SELECT * FROM users WHERE username=?", (uname,))
            if cur.fetchone():
                code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
                cur.execute("REPLACE INTO reset_codes (username, code) VALUES (?, ?)", (uname, code))
                conn.commit()
                st.session_state.reset_user = uname
                st.session_state.reset_step = 2
                st.info(f"📩 Your reset code is: **{code}** (for demo only)")
            else:
                st.error("⚠️ Username not found.")

    # Step 2: Verify reset code + set new password
    elif step == 2:
        st.info(f"🔐 Reset code sent for **{st.session_state.reset_user}**")
        code_input = st.text_input("📨 Enter Reset Code")
        new_pass = st.text_input("🔑 New Password", type="password")
        confirm_pass = st.text_input("🔁 Confirm Password", type="password")

        if st.button("Reset Password"):
            cur.execute("SELECT code FROM reset_codes WHERE username=?", (st.session_state.reset_user,))
            row = cur.fetchone()

            if not row or code_input.strip() != row[0]:
                st.error("❌ Invalid reset code.")
            elif new_pass != confirm_pass:
                st.error("⚠️ Passwords do not match.")
            else:
                hashed = bcrypt.hashpw(new_pass.encode(), bcrypt.gensalt())
                cur.execute("UPDATE users SET password=? WHERE username=?",
                            (hashed, st.session_state.reset_user))
                conn.commit()
                cur.execute("DELETE FROM reset_codes WHERE username=?", (st.session_state.reset_user,))
                conn.commit()
                st.success("✅ Password reset successful! Please login.")
                st.session_state.reset_step = 1
                st.session_state.reset_user = None

st.markdown("</div>", unsafe_allow_html=True)
