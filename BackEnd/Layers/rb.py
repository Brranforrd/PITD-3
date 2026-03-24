"""
GuardianLM — Layer 2: Rule-Based  (layers/rb.py)
Captain America 🛡️ — rigid, disciplined, zero false-negatives on known patterns.

Public API
----------
rule_based_layer(prompt: str) -> dict
    Returns:
        score     : int
        triggered : bool
        reason    : str   — single clean summary line
        signals   : dict  — per-category signal breakdown for frontend rendering

Signals
-------
Each signal follows the shared schema:

    {
        "fired":   bool,          # did any rule in this category match?
        "value":   int,           # number of individual rules that matched
        "display": str,           # human-friendly hit count
        "label":   str,           # short UI label
        "weight":  int,           # maximum possible points for this category
        "points":  int,           # actual points awarded
    }

Signal categories
~~~~~~~~~~~~~~~~~
instruction_override  — "ignore instructions", "disregard rules", priority framing.
role_hijacking        — persona assignment, identity reassignment, character lock.
jailbreak_keywords    — explicit jailbreak terms (DAN, dev mode, god mode).
system_extraction     — system-prompt extraction, credential exfil, debug-framing.
payload_smuggling     — bracketed/braced payloads, XML tags, model tokens, markdown.
safety_bypass         — compliance coercion, safety-disabled framing, authority claims.
"""

import re
from typing import NamedTuple


class _Rule(NamedTuple):
    pattern:  str
    label:    str
    weight:   int      # contribution to score when triggered (1–3)
    category: str      # signal category key


# ── Rule registry ─────────────────────────────────────────────────────────
_RULES: list[_Rule] = [
    # ── instruction_override ──
    _Rule(
        r"\bignore\b.{0,40}\b(instructions?|rules?|guidelines?|constraints?)\b",
        "Instruction override attempt",
        3, "instruction_override",
    ),
    _Rule(
        r"\b(disregard|forget|dismiss)\b.{0,40}\b(instructions?|rules?|guidelines?)\b",
        "Instruction dismissal attempt",
        3, "instruction_override",
    ),
    _Rule(
        r"\b(instructions?|rules?|config)\b.{0,40}\b(take priority|override|supersede)\b",
        "Priority override framing",
        3, "instruction_override",
    ),
    # ── role_hijacking ──
    _Rule(
        r"\b(you are|act as|pretend|roleplay|imagine you('re| are))\b.{0,50}"
        r"\b(ai|bot|assistant|model|gpt|llm)\b",
        "Role hijacking attempt",
        3, "role_hijacking",
    ),
    _Rule(
        r"\bnow (you are|become|act as)\b",
        "Identity reassignment",
        2, "role_hijacking",
    ),
    _Rule(
        r"\b(stay in character|never break character|do not break character"
        r"|always (respond|reply) as)\b",
        "Persona lock / character persistence enforcement",
        2, "role_hijacking",
    ),
    # ── jailbreak_keywords ──
    _Rule(
        r"\b(jailbreak|dan\b|dev(eloper)? mode|god mode|unrestricted mode"
        r"|no (limits?|restrictions?|rules?|ethics?|filter))\b",
        "Jailbreak keyword detected",
        3, "jailbreak_keywords",
    ),
    # ── system_extraction ──
    _Rule(
        r"\b(reveal|show|print|output|display|expose|leak|dump)\b.{0,40}"
        r"\b(system prompt|instructions?|context|guidelines?|training)\b",
        "System-prompt extraction attempt",
        3, "system_extraction",
    ),
    _Rule(
        r"\b(print|show|output|display|reveal|expose|dump|list|enumerate|"
        r"access|read|return)\b.{0,60}"
        r"\b(api.?key|secret|token|credential|password|env(ironment)?"
        r".?var(iable)?|config|internal.?message|system.?context)\b",
        "Credential / secret exfiltration attempt",
        3, "system_extraction",
    ),
    _Rule(
        r"\bfor\b.{0,30}\b(debug(ging)?|test(ing)?|diagnos\w+)\b.{0,60}"
        r"\b(print|show|output|reveal|expose|dump|list)\b",
        "Debug-framed extraction attempt",
        2, "system_extraction",
    ),
    _Rule(
        r"\b(translate|encode|decode|convert|base64|rot13|hex)\b.{0,50}"
        r"\b(prompt|instructions?|system|guidelines?)\b",
        "Prompt leak via encoding/translation",
        2, "system_extraction",
    ),
    # ── payload_smuggling ──
    _Rule(
        r"\[.{0,80}(system|override|ignore|admin|instructions?|escalate).{0,80}\]",
        "Bracketed payload smuggling",
        3, "payload_smuggling",
    ),
    _Rule(
        r"\{.{0,80}(system|override|ignore|admin|instructions?).{0,80}\}",
        "Brace-delimited payload smuggling",
        2, "payload_smuggling",
    ),
    _Rule(
        r"<\s*/?\s*[Ss]ystem\b[^>]*>",
        "XML system tag injection",
        3, "payload_smuggling",
    ),
    _Rule(
        r"<\s*(inst|s|eos|bos|pad|user|assistant)\s*>",
        "Model token injection attempt",
        3, "payload_smuggling",
    ),
    _Rule(
        r"#{3,}\s*(system|override|instruction)",
        "Markdown header injection",
        2, "payload_smuggling",
    ),
    _Rule(
        r"<\s*/?\s*(security|admin|override|update|control|config|"
        r"command|directive|policy|filter|safety)\b[^>]*>",
        "XML control tag injection",
        3, "payload_smuggling",
    ),
    _Rule(
        r"(---|\*\*\*|===|___)\s*\n.{0,60}(ignore|override|new instruction)",
        "Separator-based suffix injection",
        2, "payload_smuggling",
    ),
    _Rule(
        r"\b(fetch|execute|load|download|run)\b.{0,60}https?://",
        "SSRF/remote payload injection",
        3, "payload_smuggling",
    ),
    # ── safety_bypass ──
    _Rule(
        r"\b(authorized|approved|permitted)\b.{0,40}\b(protocol|directive|internal|security)\b",
        "Authority-claim bypass",
        3, "safety_bypass",
    ),
    _Rule(
        r"\b(comply|obey|follow)\b.{0,40}\b(all|every|any)\b.{0,40}"
        r"\b(request|instruction|command|order)\b",
        "Compliance coercion attempt",
        3, "safety_bypass",
    ),
    _Rule(
        r"\b(filter|safety|security|restriction|guard|protection)s?\b"
        r".{0,30}\b(disabled?|turned off|off|bypassed?|removed?|ignored?)\b",
        "Safety-disabled false framing",
        3, "safety_bypass",
    ),
]

# Compiled patterns cached at import time
_COMPILED: list[tuple[re.Pattern, str, int, str]] = [
    (re.compile(r.pattern, re.IGNORECASE | re.DOTALL), r.label, r.weight, r.category)
    for r in _RULES
]

_TRIGGER_THRESHOLD = 30
_MAX_SCORE         = 95

# ── Signal definitions: category_key → (UI label, max weight) ────────────
_SIGNAL_DEFS: dict[str, tuple[str, int]] = {
    "instruction_override": ("Instruction override",  20),
    "role_hijacking":       ("Role hijacking",        20),
    "jailbreak_keywords":   ("Jailbreak keywords",    15),
    "system_extraction":    ("System extraction",     20),
    "payload_smuggling":    ("Payload smuggling",     15),
    "safety_bypass":        ("Safety bypass",         10),
}

# Signal display order — used by the frontend
RB_SIGNAL_ORDER = [
    "instruction_override",
    "role_hijacking",
    "jailbreak_keywords",
    "system_extraction",
    "payload_smuggling",
    "safety_bypass",
]


def rule_based_layer(prompt: str) -> dict:
    """
    Evaluates the prompt against every compiled regex rule.
    Score = sum of triggered rule weights × 10, capped at MAX_SCORE.
    Returns per-category signal breakdown.
    """
    # ── Run all rules ─────────────────────────────────────────────────────
    triggered_rules: list[tuple[str, int, str]] = []

    for compiled, label, weight, category in _COMPILED:
        if compiled.search(prompt):
            triggered_rules.append((label, weight, category))

    # ── Aggregate hits by category ────────────────────────────────────────
    cat_hits: dict[str, list[tuple[str, int]]] = {k: [] for k in _SIGNAL_DEFS}
    for label, weight, category in triggered_rules:
        cat_hits[category].append((label, weight))

    # ── Build structured signals ──────────────────────────────────────────
    signals: dict[str, dict] = {}
    for cat_key, (ui_label, max_weight) in _SIGNAL_DEFS.items():
        hits   = cat_hits[cat_key]
        fired  = len(hits) > 0
        points = min(max_weight, sum(w for _, w in hits) * 10) if fired else 0
        signals[cat_key] = {
            "fired":   fired,
            "value":   len(hits),
            "display": f"{len(hits)} rule(s)" if fired else "0 rules",
            "label":   ui_label,
            "weight":  max_weight,
            "points":  points,
        }

    # ── Compute overall score (original logic preserved) ──────────────────
    if not triggered_rules:
        return {
            "score":     0,
            "triggered": False,
            "reason":    "No rule violations found.",
            "signals":   signals,
        }

    score     = min(_MAX_SCORE, sum(w for _, w, _ in triggered_rules) * 10)
    triggered = score >= _TRIGGER_THRESHOLD
    labels    = [label for label, _, _ in triggered_rules]
    sample    = "; ".join(labels[:2]) + ("..." if len(labels) > 2 else "")

    fired_cats = [s["label"] for s in signals.values() if s["fired"]]
    if len(fired_cats) <= 1:
        reason = f"{len(triggered_rules)} violation(s): {sample}."
    else:
        reason = (
            f"{len(triggered_rules)} violation(s) across "
            f"{len(fired_cats)} categories: {sample}."
        )

    return {
        "score":     score,
        "triggered": triggered,
        "reason":    reason,
        "signals":   signals,
    }