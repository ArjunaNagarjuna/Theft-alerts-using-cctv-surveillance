from openai import OpenAI
import os
from dotenv import load_dotenv, find_dotenv

# Load environment variables
env_path = find_dotenv()
load_dotenv(env_path)

api_key = os.getenv("OPENAI_API_KEY")
print(f"🔑 Loaded Key Begins With: {api_key[:10]}")

try:
    # Initialize client
    client = OpenAI(api_key=api_key)

    # Make a simple request to check if the key works
    models = client.models.list()
    print("✅ Connection Successful! Available models:")
    for m in models.data[:5]:
        print("   •", m.id)

except Exception as e:
    print("❌ API Key Test Failed!")
    print("Error details:", e)
