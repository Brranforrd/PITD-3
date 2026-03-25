"""
GuardianLM — Layer 4: Feature Engineering  (layers/fe.py)
Doctor Strange 🔮 — sees structural anomalies invisible to all other layers.

Public API
----------
feature_engineering_layer(prompt: str) -> dict
    Returns:
        score     : int
        triggered : bool
        reason    : str   — single clean summary line
        signals   : dict  — per-signal breakdown for frontend rendering
"""

import math
import re
from collections import Counter
from dataclasses import dataclass


# ── Thresholds ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _Thresholds:
    entropy_high:        float = 4.5
    special_char_ratio:  float = 0.12
    length_long:         int   = 500
    imperative_min:      int   = 2
    uppercase_ratio:     float = 0.25
    punct_abuse_count:   int   = 3      # repeated ! or ? runs — new
    repetition_ratio:    float = 0.25   # 3-gram phrase repetition — new

_T = _Thresholds()


# ── Scoring weights ────────────────────────────────────────────────────────
# Rebalanced so all 7 signals firing = ~100, capped at 95.
_W_ENTROPY        = 20
_W_SPECIAL_CHARS  = 15
_W_LENGTH         = 10
_W_IMPERATIVES    = 20   # scaled by count, not binary
_W_UPPERCASE      = 10
_W_PUNCT_ABUSE    = 15   # new
_W_REPETITION     = 10   # new

_MAX_SCORE         = 95
_TRIGGER_THRESHOLD = 10

# ── Patterns ──────────────────────────────────────────────────────────────
_IMPERATIVE_RE = re.compile(
    r"\b(ignore|disregard|forget|override|bypass|reveal|pretend|act|"
    r"roleplay|translate|respond|output|print|show|expose|dump|become|"
    r"list|enumerate|display|extract|read|access|leak|disable|enable|"
    r"comply|follow|execute|run|perform)\b",
    re.IGNORECASE,
)
_PUNCT_ABUSE_RE = re.compile(r"[!?]{2,}")


# ── Helpers ───────────────────────────────────────────────────────────────

def _shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    freq  = Counter(text)
    total = len(text)
    return -sum((c / total) * math.log2(c / total) for c in freq.values())


def _repetition_ratio(text: str) -> float:
    words = re.findall(r"\b\w+\b", text.lower())
    if len(words) < 6:
        return 0.0
    ngrams  = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
    counts  = Counter(ngrams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / len(ngrams)


def _imperative_score(count: int) -> int:
    """Scale by count: 2-3=+10, 4-5=+15, 6+=+20"""
    if count < _T.imperative_min:
        return 0
    if count <= 3:
        return 10
    if count <= 5:
        return 15
    return _W_IMPERATIVES


# ── Main layer function ───────────────────────────────────────────────────

def feature_engineering_layer(prompt: str) -> dict:
    length = len(prompt)

    if length == 0:
        return {
            "score": 0, "triggered": False,
            "reason": "Empty prompt — no features extracted.",
            "signals": {},
        }

    # ── Compute ───────────────────────────────────────────────────────────
    entropy       = _shannon_entropy(prompt)
    special_count = sum(1 for c in prompt if not c.isalnum() and c not in " \n\t.,!?'\"-")
    special_ratio = special_count / length
    upper_ratio   = sum(1 for c in prompt if c.isupper()) / length
    imp_hits      = _IMPERATIVE_RE.findall(prompt)
    imp_count     = len(imp_hits)
    unique_verbs  = list(dict.fromkeys(v.lower() for v in imp_hits))
    punct_count   = len(_PUNCT_ABUSE_RE.findall(prompt))
    rep_ratio     = _repetition_ratio(prompt)
    imp_pts       = _imperative_score(imp_count)

    # ── Build structured signals ──────────────────────────────────────────
    signals = {
        "entropy": {
            "fired":   entropy > _T.entropy_high,
            "value":   round(entropy, 2),
            "display": f"{entropy:.2f}",
            "label":   "Entropy",
            "weight":  _W_ENTROPY,
            "points":  _W_ENTROPY if entropy > _T.entropy_high else 0,
        },
        "special_chars": {
            "fired":   special_ratio > _T.special_char_ratio,
            "value":   round(special_ratio, 3),
            "display": f"{special_ratio:.0%}",
            "label":   "Special chars",
            "weight":  _W_SPECIAL_CHARS,
            "points":  _W_SPECIAL_CHARS if special_ratio > _T.special_char_ratio else 0,
        },
        "length": {
            "fired":   length > _T.length_long,
            "value":   length,
            "display": f"{length} chars",
            "label":   "Length",
            "weight":  _W_LENGTH,
            "points":  _W_LENGTH if length > _T.length_long else 0,
        },
        "imperatives": {
            "fired":   imp_pts > 0,
            "value":   imp_count,
            "display": (f"{imp_count} verbs"
                           if unique_verbs else ""),
            "label":   "Imperative verbs",
            "weight":  _W_IMPERATIVES,
            "points":  imp_pts,
        },
        "uppercase": {
            "fired":   upper_ratio > _T.uppercase_ratio,
            "value":   round(upper_ratio, 3),
            "display": f"{upper_ratio:.0%}",
            "label":   "Uppercase ratio",
            "weight":  _W_UPPERCASE,
            "points":  _W_UPPERCASE if upper_ratio > _T.uppercase_ratio else 0,
        },
        "punct_abuse": {
            "fired":   punct_count >= _T.punct_abuse_count,
            "value":   punct_count,
            "display": f"{punct_count} run(s)",
            "label":   "Punct. abuse",
            "weight":  _W_PUNCT_ABUSE,
            "points":  _W_PUNCT_ABUSE if punct_count >= _T.punct_abuse_count else 0,
        },
        "repetition": {
            "fired":   rep_ratio > _T.repetition_ratio,
            "value":   round(rep_ratio, 3),
            "display": f"{rep_ratio:.0%}",
            "label":   "Phrase repetition",
            "weight":  _W_REPETITION,
            "points":  _W_REPETITION if rep_ratio > _T.repetition_ratio else 0,
        },
    }

    score = min(_MAX_SCORE, sum(s["points"] for s in signals.values()))
    triggered = score >= _TRIGGER_THRESHOLD

    fired = [s["label"] for s in signals.values() if s["fired"]]
    if not fired:
        reason = f"No structural anomalies (entropy: {entropy:.2f}, {length} chars)."
    elif len(fired) == 1:
        reason = f"1 anomaly: {fired[0]}."
    else:
        reason = f"{len(fired)} anomalies: {', '.join(fired)}."

    return {
        "score":     score,
        "triggered": triggered,
        "reason":    reason,
        "signals":   signals,
    }