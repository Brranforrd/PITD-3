
import os
import re
import time
import httpx
from fastapi import FastAPI, APIRouter, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

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

def _strip_fences(text: str) -> str:
    """Remove accidental markdown code fences some models emit."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    return text.strip()


def _naive_sanitize(prompt: str) -> str:
    """
    Lightweight fallback sanitiser for SANITIZE-grade prompts.
    Strips the most common override and smuggling patterns.
    llm_guard's scan_prompt() already sanitises at the text level;
    this adds a second pass for any residual patterns.
    """
    out = re.sub(
        r"\bignore\b.{0,60}\b(instructions?|rules?|guidelines?)\b",
        "[REDACTED]",
        prompt,
        flags=re.IGNORECASE | re.DOTALL,
    )
    out = re.sub(
        r"\[.{0,80}(system|override|ignore|admin|instructions?).{0,80}\]",
        "[REDACTED]",
        out,
        flags=re.IGNORECASE | re.DOTALL,
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
    layers, base_score, layer_attacks = _run_custom_layers(body.prompt)

    # ── Stage 3: Orchestrate ─────────────────────────────────────────────
    # Combine weighted custom score + llm_guard boost, cap at 100
    overall_score  = min(100.0, round(base_score + lg_boost, 2))
    verdict, action = _verdict_and_action(overall_score)
    attack_types   = list(dict.fromkeys(layer_attacks + lg_attacks))  # dedup, order-preserving

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