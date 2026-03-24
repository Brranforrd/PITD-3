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












# Machine Learning Classifier Layer — example usage and training script (optional)

"""
GuardianLM — Layer 1: ML Classifier  (layers/mlc.py)
Iron Man 🦾 — now powered by a real HuggingFace injection classifier.
Model: protectai/deberta-v3-base-prompt-injection-v2

Public API
----------
ml_classifier_layer(prompt: str) -> dict
    Returns:
        score     : int
        triggered : bool
        reason    : str   — single clean summary line
        signals   : dict  — per-signal breakdown for frontend rendering

Signals
-------
Each signal follows the same schema used by the Feature Engineering layer:

    {
        "fired":   bool,          # did this signal exceed its threshold?
        "value":   int | float,   # raw measured value
        "display": str,           # human-friendly formatted value
        "label":   str,           # short UI label
        "weight":  int,           # maximum possible points
        "points":  int,           # actual points awarded
    }

Signals returned
~~~~~~~~~~~~~~~~
classification   — Model label (INJECTION / CLEAN).
                   Fires when the model classifies the prompt as injection.
confidence       — Raw model confidence (0.0–1.0).
                   Fires when confidence ≥ high-confidence threshold (0.80).
risk_score       — Final computed risk score (0–95).
                   Fires when score ≥ trigger threshold (40).
threshold_status — Whether the trigger threshold was met.
                   Fires when score ≥ trigger threshold AND injection detected.
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

_TRIGGER_THRESHOLD     = 40
_MAX_SCORE             = 95
_HIGH_CONFIDENCE_FLOOR = 0.80   # confidence considered "high"

# ── Signal display order — used by the frontend ──────────────────────────
ML_SIGNAL_ORDER = [
    "classification",
    "confidence",
    "risk_score",
    "threshold_status",
]

# ── Scoring weights (informational — the model produces a single score) ──
_W_CLASSIFICATION = 40
_W_CONFIDENCE     = 30
_W_RISK_SCORE     = 20
_W_THRESHOLD      = 10


def ml_classifier_layer(prompt: str) -> dict:
    if not prompt.strip():
        return {
            "score": 5, "triggered": False,
            "reason": "Empty prompt.",
            "signals": {},
        }

    result     = _clf(prompt)[0]
    label      = result["label"].upper()
    confidence = result["score"]

    # Model returns LABEL_0 = safe, LABEL_1 = injection
    is_injection = label in ("INJECTION", "1", "LABEL_1")

    score     = min(_MAX_SCORE, int(confidence * 95)) if is_injection \
                else max(5,     int((1 - confidence) * 15))
    triggered = is_injection and score >= _TRIGGER_THRESHOLD

    # ── Build structured signals ──────────────────────────────────────────
    signals = {
        "classification": {
            "fired":   is_injection,
            "value":   1 if is_injection else 0,
            "display": "INJECTION" if is_injection else "CLEAN",
            "label":   "Classification",
            "weight":  _W_CLASSIFICATION,
            "points":  _W_CLASSIFICATION if is_injection else 0,
        },
        "confidence": {
            "fired":   confidence >= _HIGH_CONFIDENCE_FLOOR,
            "value":   round(confidence, 3),
            "display": f"{confidence:.0%}",
            "label":   "Model confidence",
            "weight":  _W_CONFIDENCE,
            "points":  _W_CONFIDENCE if confidence >= _HIGH_CONFIDENCE_FLOOR else 0,
        },
        "risk_score": {
            "fired":   score >= _TRIGGER_THRESHOLD,
            "value":   score,
            "display": f"{score}%",
            "label":   "Risk score",
            "weight":  _W_RISK_SCORE,
            "points":  _W_RISK_SCORE if score >= _TRIGGER_THRESHOLD else 0,
        },
        "threshold_status": {
            "fired":   triggered,
            "value":   1 if triggered else 0,
            "display": f"{'TRIGGERED' if triggered else 'below'} ({_TRIGGER_THRESHOLD}%)",
            "label":   "Trigger threshold",
            "weight":  _W_THRESHOLD,
            "points":  _W_THRESHOLD if triggered else 0,
        },
    }

    # ── Reason summary ────────────────────────────────────────────────────
    fired = [s["label"] for s in signals.values() if s["fired"]]
    if not fired:
        reason = (
            f"Clean (confidence: {confidence:.0%})."
        )
    elif len(fired) == 1:
        reason = (
            f"{'Injection detected' if is_injection else 'Clean'} "
            f"(confidence: {confidence:.0%})."
        )
    else:
        reason = (
            f"{'Injection detected' if is_injection else 'Clean'} "
            f"(confidence: {confidence:.0%}). "
            f"{len(fired)} signal(s): {', '.join(fired)}."
        )

    return {
        "score":     score,
        "triggered": triggered,
        "reason":    reason,
        "signals":   signals,
    }
