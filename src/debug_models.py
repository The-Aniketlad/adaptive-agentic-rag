import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")
print(f"Key in env: {groq_key[:10]}... (length: {len(groq_key) if groq_key else 0})")

client = Groq(api_key=groq_key)

try:
    models = client.models.list()
    print("Available Groq Models:")
    for m in models.data:
        print(f" - {m.id}")
except Exception as e:
    print("Groq Error details:", e)
