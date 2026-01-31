import os
from dotenv import load_dotenv

# Load .env manually, overriding any system variables
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")

print(f"🔍 Loading .env from: {dotenv_path}")

if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path, override=True)  # ✅ important
else:
    print("⚠️ .env file not found!")

api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    print(f"OPENAI_API_KEY = {api_key[:20]}...")  # show partial key for security
else:
    print("❌ No API key found in .env")
