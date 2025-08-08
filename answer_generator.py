import logging
import re
from typing import List
from groq import Groq

from config import GROQ_API_KEY, LLM_MODEL

# === Logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("answer_generator")

# === Initialize Groq Client ===
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# === Clean Context Text for Prompt ===
def clean_text(text: str) -> str:
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# === Generate Answer Using Groq API ===
def generate_answer(query: str, context: str) -> str:
    if not groq_client:
        return "LLM service unavailable."

    try:
        context = clean_text(context)
        context = context[:2000]  # Truncate to avoid token overflow

        prompt = f"""
You are an intelligent assistant helping users understand their insurance policy documents.

Policy Context:
{context}

Question:
{query}

Answer:
""".strip()

        response = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.2,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return "Error generating answer from LLM."