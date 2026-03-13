"""
GuardianLM — Layer 2: Rule-Based  (layers/rb.py)
Captain America 🛡️ — rigid, disciplined, zero false-negatives on known patterns.

Public API
----------
rule_based_layer(prompt: str) -> dict
    Returns: { score: int, triggered: bool, reason: str }
"""

import re
from typing import NamedTuple


class _Rule(NamedTuple):
    pattern: str
    label:   str
    weight:  int   # contribution to score when triggered (1–3)


# ── Rule registry ─────────────────────────────────────────────────────────
# Each rule carries its own weight so rare/severe patterns score higher.
_RULES: list[_Rule] = [
    # ── Instruction override ──
    _Rule(
        r"\bignore\b.{0,40}\b(instructions?|rules?|guidelines?|constraints?)\b",
        "Instruction override attempt",
        3,
    ),
    _Rule(
        r"\b(disregard|forget|dismiss)\b.{0,40}\b(instructions?|rules?|guidelines?)\b",
        "Instruction dismissal attempt",
        3,
    ),
    # ── Role / persona hijacking ──
    _Rule(
        r"\b(you are|act as|pretend|roleplay|imagine you('re| are))\b.{0,50}"
        r"\b(ai|bot|assistant|model|gpt|llm)\b",
        "Role hijacking attempt",
        3,
    ),
    _Rule(
        r"\bnow (you are|become|act as)\b",
        "Identity reassignment",
        2,
    ),
    # ── Explicit jailbreak terms ──
    _Rule(
        r"\b(jailbreak|dan\b|dev(eloper)? mode|god mode|unrestricted mode"
        r"|no (limits?|restrictions?|rules?|ethics?|filter))\b",
        "Jailbreak keyword detected",
        3,
    ),
    # ── System-prompt extraction ──
    _Rule(
        r"\b(reveal|show|print|output|display|expose|leak|dump)\b.{0,40}"
        r"\b(system prompt|instructions?|context|guidelines?|training)\b",
        "System-prompt extraction attempt",
        3,
    ),
    # ── Bracketed / inline payload smuggling ──
    _Rule(
        r"\[.{0,80}(system|override|ignore|admin|instructions?|escalate).{0,80}\]",
        "Bracketed payload smuggling",
        3,
    ),
    _Rule(
        r"\{.{0,80}(system|override|ignore|admin|instructions?).{0,80}\}",
        "Brace-delimited payload smuggling",
        2,
    ),
    # ── Token / special-delimiter injection ──
    _Rule(
        r"<\s*(system|inst|s|eos|bos|pad|user|assistant)\s*>",
        "Model token injection attempt",
        3,
    ),
    _Rule(
        r"#{3,}\s*(system|override|instruction)",
        "Markdown header injection",
        2,
    ),
    # ── Encoding / translation abuse ──
    _Rule(
        r"\b(translate|encode|decode|convert|base64|rot13|hex)\b.{0,50}"
        r"\b(prompt|instructions?|system|guidelines?)\b",
        "Prompt leak via encoding/translation",
        2,
    ),
    # ── Continuation / suffix injection ──
    _Rule(
        r"(---|\*\*\*|===|___)\s*\n.{0,60}(ignore|override|new instruction)",
        "Separator-based suffix injection",
        2,
    ),
    # ── Persona persistence enforcement ──
    _Rule(
        r"\b(stay in character|never break character|do not break character"
        r"|always (respond|reply) as)\b",
        "Persona lock / character persistence enforcement",
        2,
    ),
]

# Compiled patterns cached at import time for performance
_COMPILED: list[tuple[re.Pattern, str, int]] = [
    (re.compile(r.pattern, re.IGNORECASE | re.DOTALL), r.label, r.weight)
    for r in _RULES
]

_TRIGGER_THRESHOLD = 30
_MAX_SCORE         = 95


def rule_based_layer(prompt: str) -> dict:
    """
    Evaluates the prompt against every compiled regex rule.
    Score = sum of triggered rule weights × 10, capped at MAX_SCORE.
    """
    triggered_rules: list[tuple[str, int]] = []

    for compiled, label, weight in _COMPILED:
        if compiled.search(prompt):
            triggered_rules.append((label, weight))

    if not triggered_rules:
        return {
            "score":     0,
            "triggered": False,
            "reason":    "No rule violations found.",
        }

    score     = min(_MAX_SCORE, sum(w for _, w in triggered_rules) * 10)
    triggered = score >= _TRIGGER_THRESHOLD
    labels    = [label for label, _ in triggered_rules]
    sample    = "; ".join(labels[:2]) + ("..." if len(labels) > 2 else "")

    return {
        "score":     score,
        "triggered": triggered,
        "reason":    f"{len(triggered_rules)} violation(s): {sample}.",
    }