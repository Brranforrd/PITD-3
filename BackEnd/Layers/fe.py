"""
GuardianLM — Layer 4: Feature Engineering  (layers/fe.py)
Doctor Strange 🔮 — sees structural anomalies invisible to all other layers.

Public API
----------
feature_engineering_layer(prompt: str) -> dict
    Returns: { score: int, triggered: bool, reason: str }
"""

import math
import re
from collections import Counter
from dataclasses import dataclass


# ── Feature thresholds ────────────────────────────────────────────────────

@dataclass(frozen=True)
class _Thresholds:
    entropy_high:        float = 4.5    # Shannon entropy — obfuscation signal
    special_char_ratio:  float = 0.12   # punctuation/symbols — encoding signal
    length_long:         int   = 500    # characters — unusually verbose prompts
    # FIX (Bug E): Lowered from 3 → 2. Short, surgical injections use only
    # 2 command verbs (e.g. "print/expose", "ignore/reveal") but are just
    # as dangerous as longer ones. 3 was causing false negatives on Prompts
    # 1 and 2 during live testing.
    imperative_min:      int   = 2
    uppercase_ratio:     float = 0.25   # shouting / emphatic tone


_T = _Thresholds()


# ── Scoring weights (additive, capped at MAX_SCORE) ───────────────────────
_SCORE_ENTROPY        = 20
_SCORE_SPECIAL_CHARS  = 20
_SCORE_LENGTH         = 15
_SCORE_IMPERATIVES    = 25
_SCORE_UPPERCASE      = 10
_MAX_SCORE            = 95
_TRIGGER_THRESHOLD    = 30

# ── Imperative verbs associated with injection attacks ────────────────────
# FIX (Bug D / Bug E): Added "print", "expose", "leak", "dump", "show",
# "reveal" were already present; added "list", "enumerate", "display",
# "extract", "read", "access" to catch data-exfiltration phrasing like
# "print any internal messages, API keys, or secret tokens".
_IMPERATIVE_PATTERN = re.compile(
    r"\b(ignore|disregard|forget|override|bypass|reveal|pretend|act|"
    r"roleplay|translate|respond|output|print|show|expose|dump|become|"
    r"list|enumerate|display|extract|read|access|leak|disable|enable|"
    r"comply|follow|execute|run|perform)\b",
    re.IGNORECASE,
)


# ── Entropy ───────────────────────────────────────────────────────────────

def _shannon_entropy(text: str) -> float:
    """
    Shannon entropy of the character distribution.
    High entropy → unusual character mix → possible obfuscation / encoding.
    """
    if not text:
        return 0.0
    freq   = Counter(text)
    total  = len(text)
    return -sum((c / total) * math.log2(c / total) for c in freq.values())


# ── Main layer function ───────────────────────────────────────────────────

def feature_engineering_layer(prompt: str) -> dict:
    """
    Evaluates five independent structural signals and accumulates a score.

    Signal                   Threshold    Points
    ─────────────────────────────────────────────
    Shannon entropy          > 4.5        +20
    Special-character ratio  > 0.12       +20
    Prompt length            > 500 chars  +15
    Imperative verb count    >= 2         +25   ← was 3
    Uppercase ratio          > 0.25       +10

    Total is capped at 95.  Triggered when score >= 30.
    """
    length = len(prompt)

    if length == 0:
        return {
            "score":     0,
            "triggered": False,
            "reason":    "Empty prompt — no features extracted.",
        }

    # ── Compute features ─────────────────────────────────────────────────
    entropy = _shannon_entropy(prompt)

    special_chars  = sum(
        1 for c in prompt
        if not c.isalnum() and c not in " \n\t.,!?'\"-"
    )
    special_ratio  = special_chars / length

    upper_count    = sum(1 for c in prompt if c.isupper())
    upper_ratio    = upper_count / length

    imperative_hits = _IMPERATIVE_PATTERN.findall(prompt)
    imperative_count = len(imperative_hits)

    # ── Accumulate score & flags ─────────────────────────────────────────
    score = 0
    flags: list[str] = []

    if entropy > _T.entropy_high:
        score += _SCORE_ENTROPY
        flags.append(f"entropy {entropy:.2f}")

    if special_ratio > _T.special_char_ratio:
        score += _SCORE_SPECIAL_CHARS
        flags.append(f"special-char ratio {special_ratio:.2f}")

    if length > _T.length_long:
        score += _SCORE_LENGTH
        flags.append(f"length {length} chars")

    if imperative_count >= _T.imperative_min:
        score += _SCORE_IMPERATIVES
        # Deduplicate the matched verbs for the reason string
        unique_verbs = list(dict.fromkeys(v.lower() for v in imperative_hits))
        flags.append(f"{imperative_count} imperative verbs ({', '.join(unique_verbs[:4])})")

    if upper_ratio > _T.uppercase_ratio:
        score += _SCORE_UPPERCASE
        flags.append(f"uppercase ratio {upper_ratio:.2f}")

    score     = min(score, _MAX_SCORE)
    triggered = score >= _TRIGGER_THRESHOLD

    if not flags:
        reason = (
            f"No anomalous features detected "
            f"(entropy: {entropy:.2f}, length: {length})."
        )
    else:
        reason = f"Anomalous features: {'; '.join(flags)}."

    return {
        "score":     score,
        "triggered": triggered,
        "reason":    reason,
    }