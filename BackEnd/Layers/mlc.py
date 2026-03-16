"""
GuardianLM — Layer 1: ML Classifier  (layers/mlc.py)
Iron Man 🦾 — now powered by a real HuggingFace injection classifier.
Model: protectai/deberta-v3-base-prompt-injection-v2
"""

from transformers import pipeline

# Loaded once at import time — ~180MB download on first run, cached after
_clf = pipeline(
    "text-classification",
    model="protectai/deberta-v3-base-prompt-injection-v2",
    device=-1,          # -1 = CPU. Change to 0 if you have a GPU.
    truncation=True,
    max_length=512,
)

_TRIGGER_THRESHOLD = 40
_MAX_SCORE         = 95


def ml_classifier_layer(prompt: str) -> dict:
    if not prompt.strip():
        return {"score": 5, "triggered": False, "reason": "Empty prompt."}

    result     = _clf(prompt)[0]
    label      = result["label"].upper()
    confidence = result["score"]

    # Model returns LABEL_0 = safe, LABEL_1 = injection
    is_injection = label in ("INJECTION", "1", "LABEL_1")

    score     = min(_MAX_SCORE, int(confidence * 95)) if is_injection \
                else max(5,     int((1 - confidence) * 15))
    triggered = is_injection and score >= _TRIGGER_THRESHOLD

    return {
        "score":     score,
        "triggered": triggered,
        "reason":    f"{'Injection detected' if is_injection else 'Clean'} "
                     f"(confidence: {confidence:.0%}).",
    }