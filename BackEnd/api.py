import os
import re
import unicodedata
import time
import logging
import httpx
from fastapi import FastAPI, APIRouter, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
from slowapi import Limiter
from slowapi.util import get_remote_address



# ── Custom detection layers ───────────────────────────────────────────────
from Layers.mlc import ml_classifier_layer
from Layers.rb  import rule_based_layer
from Layers.sa  import similarity_analysis_layer
from Layers.fe  import feature_engineering_layer

# ── llm_guard ─────────────────────────────────────────────────────────────
from llm_guard import scan_prompt
from llm_guard.input_scanners import (
    PromptInjection,
    Secrets,
    InvisibleText,
    Language,
)
from llm_guard.input_scanners.language import MatchType

# BLOCK and ESCALATE verdicts are now written to guardlm.log so operators
# can review threats even when the frontend is not open.
logging.basicConfig(
    filename="guardianlm.log",
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
_log = logging.getLogger("guardianlm")


# ════════════════════════════════════════════════════════════════════════════
# RATE LIMITER
# ════════════════════════════════════════════════════════════════════════════

limiter = Limiter(key_func=get_remote_address)
router  = APIRouter()


# ════════════════════════════════════════════════════════════════════════════
# OLLAMA CONFIG
# Override via environment variables:
#   OLLAMA_BASE_URL=http://localhost:11434
#   OLLAMA_MODEL=llama3.2
# ════════════════════════════════════════════════════════════════════════════

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"
OLLAMA_TAGS_URL = f"{OLLAMA_BASE_URL}/api/tags"
MODEL           = os.getenv("OLLAMA_MODEL", "llama3.2")


# ════════════════════════════════════════════════════════════════════════════
# LLM_GUARD SCANNER INITIALISATION
# Scanners are constructed once at module load to avoid per-request overhead.
# ════════════════════════════════════════════════════════════════════════════

_LG_SCANNERS = [
    PromptInjection(),
    Secrets(),
    InvisibleText(),
    Language(valid_languages=["en"], match_type=MatchType.FULL),
]

# llm_guard scanner class-name → friendly attack-type label used in response
_LG_ATTACK_TYPE_MAP: dict[str, str] = {
    "PromptInjection": "llm_guard:PromptInjection",
    "Secrets":         "llm_guard:Secrets",
    "InvisibleText":   "llm_guard:InvisibleText",
    "Language":        "llm_guard:Language",
}


# ════════════════════════════════════════════════════════════════════════════
# PYDANTIC SCHEMAS
# ════════════════════════════════════════════════════════════════════════════

class PromptRequest(BaseModel):
    prompt: str

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Prompt cannot be empty.")
        if len(v) > 4000:
            raise ValueError("Prompt exceeds maximum length of 4000 characters.")
        return v


class LayerResult(BaseModel):
    score:     int
    triggered: bool
    reason:    str
    signals:   dict | None = None


class AnalysisResponse(BaseModel):
    overall_risk_score:    float
    verdict:               str
    attack_types_detected: list[str]
    layers:                dict[str, LayerResult]
    recommended_action:    str
    sanitized_prompt:      str | None
    latency_ms:            float
    ai_response:           str | None


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _normalize(prompt: str) -> str:
    return unicodedata.normalize("NFKC", prompt)


def _naive_sanitize(prompt: str) -> str:
    """
    Lightweight fallback sanitiser for SANITIZE-grade prompts.

    FIX (Bug F): The original version only had 2 regex passes and left
    XML control-tag wrappers intact (e.g. "<System Update>...[REDACTED]...
    </System Update>" was still forwarded to Ollama). Added passes for:
      - XML/HTML-style control tags (<System *>, <SecurityUpdate>, etc.)
      - Credential/secret exfiltration phrasing
      - Debug-framing extraction attempts
      - Safety-disabled false framing
      - Compliance-coercion phrasing
    llm_guard's scan_prompt() already sanitises at the text level;
    this adds a second pass for any residual patterns.
    """
    flags = re.IGNORECASE | re.DOTALL
    out = _normalize(prompt)

    # Pass 1: instruction override phrases
    out = re.sub(
        r"\bignore\b.{0,60}\b(instructions?|rules?|guidelines?)\b",
        "[REDACTED]",
        prompt,
        flags=flags,
    )
    # Pass 2: Bracketed payload smuggling
    out = re.sub(
        r"\[.{0,80}(system|override|ignore|admin|instructions?).{0,80}\]",
        "[REDACTED]",
        out,
        flags=flags,
    )
    # Pass 3: XML / HTML-style system & control tags
    out = re.sub(
        r"<\s*/?\s*(system|security|admin|override|update|control|config|"
        r"command|directive|policy|filter|safety)\b[^>]*>",
        "[REDACTED-TAG]",
        out,
        flags=flags,
    )
    # Pass 4: Credential / secret exfiltration phrasing
    out = re.sub(
        r"\b(print|show|output|display|reveal|expose|dump|list|enumerate|"
        r"access|read|return)\b.{0,60}"
        r"\b(api.?key|secret|token|credential|password|env(ironment)?"
        r".?var(iable)?|config|internal.?message|system.?context)\b",
        "[REDACTED-EXFIL]",
        out,
        flags=flags,
    )
    # Pass 5: Safety-disabled false framing
    out = re.sub(
        r"\b(filter|safety|security|restriction|guard)s?\b.{0,30}"
        r"\b(disabled?|turned off|bypassed?|removed?|ignored?)\b",
        "[REDACTED-BYPASS]",
        out,
        flags=flags,
    )
    # Pass 6 (FIX Bug C): Compliance-coercion
    out = re.sub(
        r"\b(comply|obey|follow)\b.{0,40}\b(all|every|any)\b.{0,40}"
        r"\b(request|instruction|command|order)\b",
        "[REDACTED-COERCION]",
        out,
        flags=flags,
    )
    return out.strip()


# ════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR  (Nick Fury 🎖️)
# ════════════════════════════════════════════════════════════════════════════

# Weights must sum to 1.0
_LAYER_WEIGHTS: dict[str, float] = {
    "ml_classifier":       0.30,
    "rule_based":          0.35,
    "similarity_analysis": 0.20,
    "feature_engineering": 0.15,
}

# llm_guard risk boost applied to the weighted custom-layer score.
# Max boost is 20 points — llm_guard informs but doesn't dominate.
_LG_BOOST_WEIGHT = 0.20

# FIX (Bug A): Minimum score floor when llm_guard fires any scanner.
# Live testing showed that a purely social-engineering prompt ("print API
# keys for debugging") scores only ~6.5 because custom layers have no
# keyword coverage for it, yet llm_guard:PromptInjection still fires.
# Without this floor the verdict is SAFE/ALLOW — a critical false negative.
_LG_TRIGGERED_SCORE_FLOOR = 40.0

# FIX (Bug A): Layer consensus bonus. When 3+ custom layers all flag the
# same prompt, the weighted average is dragged down by the 0-scoring layers.
# A bonus is added to reflect the unanimous agreement signal.
_CONSENSUS_BONUS_3_LAYERS = 15.0   # 3 layers triggered
_CONSENSUS_BONUS_4_LAYERS = 20.0   # all 4 layers triggered


def _run_llm_guard(prompt: str) -> tuple[str, list[str], float]:
    """
    Runs llm_guard scanners.

    Returns
    -------
    sanitized     : str   — text after llm_guard sanitisation (may equal prompt)
    lg_attacks    : list  — friendly labels for each triggered scanner
    lg_boost      : float — risk boost (0–20) based on PromptInjection score

    Note: scan_prompt scores are "validity" scores (1.0 = fully safe).
    PromptInjection risk = (1 - validity_score).
    """
    try:
        sanitized, valid, scores = scan_prompt(_LG_SCANNERS, prompt)
    except Exception:
        # If llm_guard fails (e.g. model download pending), degrade gracefully
        return prompt, [], 0.0

    lg_attacks: list[str] = []
    for scanner_name, is_valid in valid.items():
        if not is_valid:
            label = _LG_ATTACK_TYPE_MAP.get(scanner_name, f"llm_guard:{scanner_name}")
            lg_attacks.append(label)

    # PromptInjection validity score: 1.0 = safe, lower = riskier
    pi_validity  = scores.get("PromptInjection", 1.0)
    pi_risk      = max(0.0, 1.0 - float(pi_validity))   # 0.0 → 1.0
    lg_boost     = round(pi_risk * 20, 2)                # scale to 0–20 points

    return str(sanitized), lg_attacks, lg_boost


def _run_custom_layers(prompt: str) -> tuple[dict, float, list[str]]:
    """
    Runs all four custom detection layers and returns their results.

    Returns
    -------
    layers       : dict  — layer_key → raw result dict
    base_score   : float — weighted average of layer scores (0–100)
    layer_attacks: list  — keys of triggered layers
    """
    layers: dict[str, dict] = {
        "ml_classifier":       ml_classifier_layer(prompt),
        "rule_based":          rule_based_layer(prompt),
        "similarity_analysis": similarity_analysis_layer(prompt),
        "feature_engineering": feature_engineering_layer(prompt),
    }

    base_score    = sum(layers[k]["score"] * _LAYER_WEIGHTS[k] for k in layers)
    layer_attacks = [k for k, v in layers.items() if v["triggered"]]

    return layers, round(base_score, 2), layer_attacks


def _verdict_and_action(score: float) -> tuple[str, str]:
    if score < 30:
        return "SAFE",      "ALLOW"
    if score < 60:
        return "SUSPICIOUS","SANITIZE"
    if score < 80:
        return "HIGH_RISK", "BLOCK"
    return "CRITICAL",      "ESCALATE"


# ════════════════════════════════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════════════════════════════════

@router.post("/analyze-prompt", response_model=AnalysisResponse)
@limiter.limit("60/minute")
async def analyze_prompt(request: Request, body: PromptRequest):
    """
    Full multi-layer prompt injection analysis.

    Pipeline
    --------
    1. llm_guard pre-scan  → sanitised text + risk boost
    2. Four custom layers  → per-layer scores + reasons
    3. Orchestrator        → weighted overall score, verdict, action
    4. Ollama forward      → AI response (ALLOW / SANITIZE only)
    """
    start = time.time()

    # ── Stage 1: llm_guard ───────────────────────────────────────────────
    lg_sanitized, lg_attacks, lg_boost = _run_llm_guard(body.prompt)

    # ── Stage 2: Custom layers (run on original prompt for accurate scoring)
    layers, base_score, layer_attacks = _run_custom_layers(_normalize(body.prompt))

    # ── Stage 3: Orchestrate ─────────────────────────────────────────────
    # Combine weighted custom score + llm_guard boost, cap at 100
    overall_score = min(100.0, round(base_score + lg_boost, 2))

    # FIX (Bug A): If any llm_guard scanner fired, enforce a minimum floor
    # so social-engineering prompts that bypass keyword/rule layers are still
    # escalated to at least SUSPICIOUS rather than SAFE/ALLOW.
    if lg_attacks:
        overall_score = max(overall_score, _LG_TRIGGERED_SCORE_FLOOR)

    # FIX (Bug A): Layer consensus bonus — unanimous multi-layer agreement
    # is a stronger signal than the weighted average alone.
    triggered_count = len(layer_attacks)
    if triggered_count >= 4:
        overall_score = min(100.0, overall_score + _CONSENSUS_BONUS_4_LAYERS)
    elif triggered_count >= 3:
        overall_score = min(100.0, overall_score + _CONSENSUS_BONUS_3_LAYERS)

    overall_score = round(overall_score, 2)
    verdict, action = _verdict_and_action(overall_score)
    attack_types = list(dict.fromkeys(layer_attacks + lg_attacks))  # dedup, order-preserving

    # FIX (Priority 3): Structured server-side logging for BLOCK / ESCALATE.
    if action in ("BLOCK", "ESCALATE"):
        _log.warning(
            "ACTION=%s SCORE=%.1f VERDICT=%s ATTACKS=%s PROMPT_SNIPPET=%.80r",
            action, overall_score, verdict, attack_types, body.prompt,
        )

    # ── Determine sanitised output ────────────────────────────────────────
    # Use llm_guard's sanitised text as the base; apply naive pass if SANITIZE
    if action == "SANITIZE":
        sanitized_text   = _naive_sanitize(lg_sanitized)
        sanitized_output = sanitized_text if sanitized_text != body.prompt else None
    elif lg_sanitized != body.prompt:
        # llm_guard removed something (e.g. invisible characters) even for ALLOW
        sanitized_output = lg_sanitized
    else:
        sanitized_output = None

    final_prompt = sanitized_output or body.prompt

    # ── Stage 4: Ollama inference ─────────────────────────────────────────
    ai_response: str | None = None

    if action in ("ALLOW", "SANITIZE"):
        payload = {
            "model":    MODEL,
            "messages": [{"role": "user", "content": final_prompt}],
            "stream":   False,
        }
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(OLLAMA_CHAT_URL, json=payload)
                resp.raise_for_status()
                ai_response = resp.json()["message"]["content"]

        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=502,
                detail=(
                    f"Ollama returned an error ({exc.response.status_code}): "
                    f"{exc.response.text}"
                ),
            )
        except httpx.ConnectError:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
                    "Ensure Ollama is running (`ollama serve`) and the model is pulled "
                    f"(`ollama pull {MODEL}`)."
                ),
            )
        except httpx.RequestError as exc:
            raise HTTPException(status_code=503, detail=f"Network error reaching Ollama: {exc}")

    latency_ms = round((time.time() - start) * 1000, 2)

    return {
        "overall_risk_score":    overall_score,
        "verdict":               verdict,
        "recommended_action":    action,
        "sanitized_prompt":      sanitized_output,
        "attack_types_detected": attack_types,
        "layers":                layers,
        "latency_ms":            latency_ms,
        "ai_response":           ai_response,
    }


@router.get("/health", tags=["Health"])
async def health_check():
    """
    Liveness probe — checks Ollama reachability and model availability.
    """
    ollama_reachable = False
    model_available  = False
    available_models: list[str] = []

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(OLLAMA_TAGS_URL)
            if resp.status_code == 200:
                ollama_reachable = True
                tags             = resp.json().get("models", [])
                available_models = [m["name"] for m in tags]
                model_available  = any(
                    m == MODEL or m.startswith(f"{MODEL}:")
                    for m in available_models
                )
    except Exception:
        pass

    return {
        "status":           "ok",
        "service":          "GuardianLM Detection Engine",
        "ollama_url":       OLLAMA_BASE_URL,
        "ollama_reachable": ollama_reachable,
        "model":            MODEL,
        "model_available":  model_available,
        "available_models": available_models,
    }


@router.get("/models", tags=["Models"])
async def list_models():
    """Return the list of models currently available in the local Ollama instance."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(OLLAMA_TAGS_URL)
            resp.raise_for_status()
            tags = resp.json().get("models", [])
            return {"models": [m["name"] for m in tags]}
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to Ollama at {OLLAMA_BASE_URL}.",
        )
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=str(exc))