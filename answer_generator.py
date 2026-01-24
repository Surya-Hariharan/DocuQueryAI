import logging
import re
from typing import List
from groq import Groq

from config import GROQ_API_KEY, LLM_MODEL
from utils import monitor_performance, retry_on_failure

# === Logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("answer_generator")

# === Initialize Groq Client ===
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# === Clean Context Text for Prompt ===
def clean_text(text: str) -> str:
    """Clean and normalize text for LLM prompt."""
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# === Generate Answer Using Groq API with Retry ===
@monitor_performance("generate_answer")
@retry_on_failure(max_retries=3, delay=1.0)
def generate_answer(query: str, context: str, max_context_length: int = 4000) -> str:
    """
    Generate answer using Groq LLM with retry logic and better prompting.
    
    Args:
        query: User question
        context: Retrieved context from vector store
        max_context_length: Maximum context length to avoid token overflow
    
    Returns:
        Generated answer
    """
    if not groq_client:
        logger.error("Groq client not initialized")
        return "LLM service unavailable."

    try:
        # Clean and truncate context
        context = clean_text(context)
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."
            logger.warning(f"Context truncated to {max_context_length} chars")

        # Improved prompt template
        prompt = f"""You are an intelligent assistant helping users understand their documents.

Context from documents:
{context}

User question:
{query}

Instructions:
- Answer based on the provided context
- Be concise and accurate
- If the context doesn't contain enough information, say so
- Cite specific details from the context when possible

Answer:""".strip()

        # Call Groq API
        response = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.2,
        )

        answer = response.choices[0].message.content.strip()
        logger.info(f"✅ Answer generated ({len(answer)} chars)")
        return answer

    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        raise  # Let retry decorator handle it