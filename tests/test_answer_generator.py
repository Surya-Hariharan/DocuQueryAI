import pytest

pytest.importorskip("groq", reason="groq package not installed in this environment")

from docuqueryai.generation.answer_generator import is_refusal, looks_grounded, REFUSAL_MESSAGE


def test_is_refusal_matches_exact_sentinel():
    assert is_refusal(REFUSAL_MESSAGE) is True
    assert is_refusal(REFUSAL_MESSAGE + " ") is True  # trailing whitespace tolerated
    assert is_refusal("I don't know") is False


def test_refusal_is_always_considered_grounded():
    # A refusal isn't a fabrication risk — it should never be flagged
    # low-confidence just because it doesn't overlap with the context.
    assert looks_grounded(REFUSAL_MESSAGE, context="completely unrelated context text") is True


def test_grounded_answer_passes():
    context = "The warranty period is 24 months from the date of purchase (Source 1)."
    answer = "The warranty period is 24 months from the date of purchase (Source 1)."
    assert looks_grounded(answer, context) is True


def test_fabricated_answer_is_flagged():
    context = "The warranty period is 24 months from the date of purchase."
    answer = "According to quantum thermodynamics regulations, spacecraft require annual recalibration."
    assert looks_grounded(answer, context) is False


def test_empty_answer_is_not_flagged():
    # Nothing to check overlap against — shouldn't be treated as a fabrication.
    assert looks_grounded("", context="some context") is True
