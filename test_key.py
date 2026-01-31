from dotenv import load_dotenv, find_dotenv
import os

# Locate and load the .env file
env_path = find_dotenv()
print("📂 .env Path:", env_path)

# Load environment variables
load_dotenv(env_path)

# Print the first few characters of your key
key = os.getenv("OPENAI_API_KEY")
if key:
    print("🔑 Loaded Key Starts With:", key[:10])
else:
    print("❌ No API key found in .env file.")
