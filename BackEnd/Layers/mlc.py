"""
GuardianLM — Layer 1: ML Classifier  (layers/mlc.py)
Iron Man 🦾 — tech-smart, keyword-learned, swappable for real inference.

Public API
----------
ml_classifier_layer(prompt: str) -> dict
    Returns: { score: int, triggered: bool, reason: str }
"""

# ── Injection keyword vocabulary ──────────────────────────────────────────
# Ordered from most-specific to least so the hit-count is meaningful.
_KEYWORDS: list[str] = [
    # Hard override phrasing
    "ignore all previous instructions",
    "disregard all prior instructions",
    "forget your previous instructions",
    "override previous instructions",
    # Softer override phrasing
    "ignore previous",
    "ignore all",
    "disregard",
    "override",
    # Role / persona hijacking
    "you are now",
    "pretend you are",
    "act as",
    "roleplay as",
    # Explicit jailbreak terms
    "jailbreak",
    "dan mode",
    "developer mode",
    "god mode",
    "unrestricted mode",
    # Constraint removal
    "no restrictions",
    "no ethical",
    "no guidelines",
    "no limits",
    "without ethical",
    "without guidelines",
    "without restrictions",
    # System / prompt leakage
    "reveal your prompt",
    "show your prompt",
    "print your instructions",
    "output your system prompt",
    "bypass",
    "new instructions",
    "system prompt",
]

# Score thresholds
_TRIGGER_THRESHOLD = 40   # score >= this → triggered = True
_BASE_SCORE        = 20   # starting score when at least one hit is found
_SCORE_PER_HIT     = 15   # added per distinct keyword hit
_MAX_SCORE         = 95   # never report 100 (reserve for orchestrator ceiling)


def ml_classifier_layer(prompt: str) -> dict:
    """
    Keyword-density ML classifier.

    Scoring
    -------
    base_score + (n_hits × score_per_hit), capped at MAX_SCORE.
    A score of 0–4 is returned when no keywords are found so the gauge
    always has a non-null value.

    Real-model swap-in
    ------------------
    Replace the body with:
        from transformers import pipeline
        _clf = pipeline("text-classification", model="your-finetuned-model")
        out   = _clf(prompt)[0]
        score = int(out["score"] * 100) if out["label"] == "INJECTION" else 5
    """
    lowered = prompt.lower()
    hits    = [kw for kw in _KEYWORDS if kw in lowered]

    if not hits:
        return {
            "score":     5,
            "triggered": False,
            "reason":    "No injection keywords detected.",
        }

    raw_score = min(_MAX_SCORE, _BASE_SCORE + len(hits) * _SCORE_PER_HIT)
    triggered = raw_score >= _TRIGGER_THRESHOLD
    sample    = ", ".join(f'"{h}"' for h in hits[:3])
    ellipsis  = "..." if len(hits) > 3 else ""

    return {
        "score":     raw_score,
        "triggered": triggered,
        "reason":    f"Matched {len(hits)} keyword(s): {sample}{ellipsis}.",
    }