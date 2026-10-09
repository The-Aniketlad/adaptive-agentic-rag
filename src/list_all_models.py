import os
from dotenv import load_dotenv

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")
gemini_key = os.getenv("GOOGLE_API_KEY")

print("--- Checking Google Gemini Models ---")
if gemini_key:
    try:
        from google import genai
        client = genai.Client(api_key=gemini_key)
        print("Fetching Gemini models...")
        for m in client.models.list():
            if "gemini" in m.name.lower():
                print(f"Gemini Model: {m.name}")
    except Exception as e:
        print("Gemini List Error:", e)

print("\n--- Checking Groq Models ---")
if groq_key:
    try:
        from groq import Groq
        client = Groq(api_key=groq_key)
        for m in client.models.list().data:
            print(f"Groq Model: {m.id}")
    except Exception as e:
        print("Groq List Error:", e)
