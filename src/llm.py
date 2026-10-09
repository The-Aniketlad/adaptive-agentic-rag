import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def get_primary_llm(temperature: float = 0.0):
    """
    Returns the primary fast LLM (Groq Qwen-3.8 / GPT-OSS or Gemini 3.8 Flash).
    Prioritizes Groq for ultra-low latency (~300ms).
    """
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        try:
            return ChatGroq(
                model="qwen/qwen3.8-27b",
                temperature=temperature,
                groq_api_key=groq_key,
                timeout=15
            )
        except Exception:
            pass
            
    # Fallback to Google Gemini
    gemini_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if gemini_key:
        return ChatGoogleGenerativeAI(
            model="gemini-3.8-flash",
            temperature=temperature,
            google_api_key=gemini_key
        )
        
    raise ValueError("No valid LLM API key configured.")

def get_fallback_llm(temperature: float = 0.0):
    """
    Returns a backup LLM (Groq GPT-OSS or Gemini).
    """
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        try:
            return ChatGroq(
                model="openai/gpt-oss-120b",
                temperature=temperature,
                groq_api_key=groq_key,
                timeout=15
            )
        except Exception:
            pass
            
    gemini_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    return ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=temperature,
        google_api_key=gemini_key
    )
