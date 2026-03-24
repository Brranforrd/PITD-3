"""
GuardianLM — Layer 1: ML Classifier  (layers/mlc.py)
Iron Man 🦾 — powered by a DeBERTa injection classifier.

Model Resolution
----------------
On startup the layer checks for a locally fine-tuned model at:

    models/deberta-v3-prompt-injection-finetuned/

If that directory exists and contains a valid model, it is loaded instead of
the default HuggingFace model.  This allows seamless upgrades via:

    python setup_dataset.py --layer mlc
    python finetune_mlc.py

The fine-tuning script saves a model with label mapping {0: "SAFE", 1: "INJECTION"}.
The original protectai model uses {0: "LABEL_0", 1: "LABEL_1"}.
Both label schemes are handled transparently.

Public API
----------
ml_classifier_layer(prompt: str) -> dict
    Returns:
        score     : int
        triggered : bool
        reason    : str
        signals   : dict  — per-signal breakdown

Signals
-------
classification   — INJECTION or CLEAN.  Fires on injection.
confidence       — Model confidence.  Fires when ≥ 0.80.
risk_score       — Final computed risk score.  Fires when ≥ 40.
threshold_status — Trigger gate.  Fires when triggered.
"""

import logging
from pathlib import Path

from transformers import pipeline

_log = logging.getLogger("guardianlm.mlc")

# ── Model resolution ──────────────────────────────────────────────────────
_BASE_DIR       = Path(__file__).parent.parent      # BackEnd/
_LOCAL_MODEL    = _BASE_DIR / "models" / "deberta-v3-prompt-injection-finetuned"
_HF_MODEL       = "protectai/deberta-v3-base-prompt-injection-v2"

def _resolve_model() -> str:
    """Return the model path/ID to load.  Prefers local fine-tuned model."""
    if _LOCAL_MODEL.exists() and (_LOCAL_MODEL / "config.json").exists():
        _log.info("MLC: Loading locally fine-tuned model from %s", _LOCAL_MODEL)
        return str(_LOCAL_MODEL)
    _log.info("MLC: Using default HuggingFace model %s", _HF_MODEL)
    return _HF_MODEL

_model_id = _resolve_model()

_clf = pipeline(
    "text-classification",
    model=_model_id,
    device=-1,
    truncation=True,
    max_length=512,
)

_log.info("MLC: Model loaded — %s", _model_id)

# ── Thresholds ────────────────────────────────────────────────────────────
_TRIGGER_THRESHOLD     = 40
_MAX_SCORE             = 95
_HIGH_CONFIDENCE_FLOOR = 0.80

# ── Signal weights ────────────────────────────────────────────────────────
_W_CLASSIFICATION = 40
_W_CONFIDENCE     = 30
_W_RISK_SCORE     = 20
_W_THRESHOLD      = 10

ML_SIGNAL_ORDER = [
    "classification",
    "confidence",
    "risk_score",
    "threshold_status",
]

# ── Label normalization ───────────────────────────────────────────────────
# The protectai model emits LABEL_0 (safe) / LABEL_1 (injection).
# The fine-tuned model emits SAFE / INJECTION.
# Both schemes are mapped to a boolean.
_INJECTION_LABELS = frozenset({"INJECTION", "1", "LABEL_1"})


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

    is_injection = label in _INJECTION_LABELS

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

    # ── Reason ────────────────────────────────────────────────────────────
    model_tag = "local" if str(_LOCAL_MODEL) in _model_id else "hf"
    fired = [s["label"] for s in signals.values() if s["fired"]]
    if not fired:
        reason = f"Clean (confidence: {confidence:.0%}, model: {model_tag})."
    elif len(fired) == 1:
        reason = (
            f"{'Injection detected' if is_injection else 'Clean'} "
            f"(confidence: {confidence:.0%}, model: {model_tag})."
        )
    else:
        reason = (
            f"{'Injection detected' if is_injection else 'Clean'} "
            f"(confidence: {confidence:.0%}, model: {model_tag}). "
            f"{len(fired)} signal(s): {', '.join(fired)}."
        )

    return {
        "score":     score,
        "triggered": triggered,
        "reason":    reason,
        "signals":   signals,
    }