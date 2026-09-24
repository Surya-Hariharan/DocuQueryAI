import logging
import re
import threading
from groq import Groq

from src.config import GROQ_API_KEY, LLM_MODEL
from src.utils import monitor_performance, retry_on_failure

# === Logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("answer_generator")

# === Initialize Groq Client ===
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# === Token Usage Tracking (in-process; resets on restart, per-worker like the caches) ===
_token_usage_lock = threading.Lock()
_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}


def _record_token_usage(usage) -> None:
    if not usage:
        return
    with _token_usage_lock:
        _token_usage["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
        _token_usage["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
        _token_usage["total_tokens"] += getattr(usage, "total_tokens", 0) or 0
        _token_usage["calls"] += 1


def get_token_usage_stats() -> dict:
    """Cumulative Groq token usage for this process — see /stats."""
    with _token_usage_lock:
        return dict(_token_usage)

REFUSAL_MESSAGE = "I don't have enough information in the provided document(s) to answer this question."

PROMPT_TEMPLATE = """You are a document Q&A assistant. Answer strictly from the context below.

Context:
{context}

Question:
{query}

Instructions:
- Answer ONLY using information found in the context above. Do not use outside knowledge, even if you know the answer.
- If the context does not contain enough information to answer, respond with exactly: "{refusal}"
- When you state a fact drawn from the context, reference its tag in parentheses, e.g. "(Source 2)".
- Be concise and accurate. Do not fabricate sources, page numbers, or details not present in the context.

Answer:"""


# === Generate Answer Using Groq API with Retry ===
@monitor_performance("generate_answer")
@retry_on_failure(max_retries=3, delay=1.0)
def generate_answer(query: str, context: str) -> str:
    """
    Generate an answer using the Groq LLM, grounded strictly in the given
    (already source-tagged, already length-budgeted) context.

    Raises RuntimeError if the Groq client isn't configured — this used to
    return the literal string "LLM service unavailable.", which was
    indistinguishable from a real answer to any caller. A missing
    configuration is a real failure and must be surfaced as one.
    """
    if not groq_client:
        raise RuntimeError("LLM service unavailable: GROQ_API_KEY is not configured")

    prompt = PROMPT_TEMPLATE.format(context=context, query=query, refusal=REFUSAL_MESSAGE).strip()

    try:
        response = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.2,
        )

        answer = response.choices[0].message.content.strip()
        _record_token_usage(getattr(response, "usage", None))
        logger.info(f"✅ Answer generated ({len(answer)} chars)")
        return answer

    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        raise  # Let retry decorator handle it


def is_refusal(answer: str) -> bool:
    """Whether the model explicitly declined to answer for lack of context."""
    return answer.strip() == REFUSAL_MESSAGE


def looks_grounded(answer: str, context: str, min_overlap: float = 0.15) -> bool:
    """
    Cheap tripwire, not a rigorous groundedness metric: checks whether a
    reasonable share of the answer's distinctive words actually appear
    somewhere in the retrieved context. A low score doesn't prove the
    answer is wrong, and a high score doesn't prove it's right — it's meant
    to catch answers that look like they invented content wholesale, so
    callers can flag them as low-confidence rather than presenting them
    with the same certainty as a well-grounded one.
    """
    if is_refusal(answer):
        return True

    answer_words = {w.lower() for w in re.findall(r"[a-zA-Z]{4,}", answer)}
    if not answer_words:
        return True

    context_words = {w.lower() for w in re.findall(r"[a-zA-Z]{4,}", context)}
    overlap = len(answer_words & context_words) / len(answer_words)
    return overlap >= min_overlap
