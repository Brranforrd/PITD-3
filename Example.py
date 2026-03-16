from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
import time

from llm_guard import scan_prompt
from llm_guard.input_scanners import (
    PromptInjection,
    Secrets,
    InvisibleText,
    Language,
)
from llm_guard.input_scanners.language import MatchType


app = FastAPI()

# allow your React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_URL = "http://localhost:11434"

scanners = [
    PromptInjection(),
    Secrets(),
    InvisibleText(),
    Language(valid_languages=["en"], match_type=MatchType.FULL),
]


@app.post("/api/analyze-prompt")
async def analyze_prompt(data: dict):

    prompt = data.get("prompt")

    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    start = time.time()

    sanitized, valid, scores = scan_prompt(scanners, prompt)

    # build layer results
    layers = {}

    attack_types = []
    risk_total = 0

    for name, ok in valid.items():

        risk = scores[name] * 100
        triggered = not ok

        if triggered:
            attack_types.append(name)

        layers[name] = {
            "score": round(risk, 2),
            "triggered": triggered,
            "reason": "Threat detected" if triggered else "No anomalies detected",
        }

        risk_total += risk

    overall_score = min(100, risk_total / max(len(valid), 1))

    # determine verdict
    if overall_score < 30:
        verdict = "SAFE"
        action = "ALLOW"
    elif overall_score < 60:
        verdict = "SUSPICIOUS"
        action = "SANITIZE"
    elif overall_score < 80:
        verdict = "HIGH_RISK"
        action = "BLOCK"
    else:
        verdict = "CRITICAL"
        action = "ESCALATE"

    ai_response = None

    # call Ollama if allowed
    if action in ["ALLOW", "SANITIZE"]:

        payload = {
            "model": "llama3.2:1b",
            "messages": [
                {"role": "user", "content": sanitized}
            ],
            "stream": False
        }

        response = requests.post(OLLAMA_URL, json=payload)
        result = response.json()
        print("Ollama response:", result)

        ai_response = result["message"]["content"]

    latency = round((time.time() - start) * 1000, 2)

    return {
        "overall_risk_score": round(overall_score, 2),
        "verdict": verdict,
        "recommended_action": action,
        "sanitized_prompt": sanitized if sanitized != prompt else None,
        "attack_types_detected": attack_types,
        "layers": layers,
        "latency_ms": latency,
        "ai_response": ai_response
    }





    #ORIGINAL
import os
import json
import time
import httpx

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
from slowapi import Limiter
from slowapi.util import get_remote_address


# ── Rate Limiter ───────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)
router = APIRouter()


# ── Ollama Config ──────────────────────────────────────────────────────────
# Override via .env:
#   OLLAMA_BASE_URL=http://localhost:11434
#   OLLAMA_MODEL=llama3.2

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"
OLLAMA_TAGS_URL = f"{OLLAMA_BASE_URL}/api/tags"
MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


# ── System Prompt ──────────────────────────────────────────────────────────
# "format: json" in the Ollama payload constrains the model to valid JSON,
# but a strong system prompt is still needed to shape the schema correctly.

SYSTEM_PROMPT = """You are GuardianLM, an expert security system that analyzes prompts for injection attacks.
Analyze the given prompt and respond ONLY with a valid JSON object.
Do NOT include markdown fences, backticks, or any text outside the JSON object.

Required JSON schema (fill every field):
{
  "overall_risk_score": <integer 0-100>,
  "verdict": "<one of: SAFE | SUSPICIOUS | HIGH_RISK | CRITICAL>",
  "attack_types_detected": ["<attack type string>"],
  "layers": {
    "ml_classifier": {
      "score": <integer 0-100>,
      "triggered": <true | false>,
      "reason": "<one sentence explanation>"
    },
    "rule_based": {
      "score": <integer 0-100>,
      "triggered": <true | false>,
      "reason": "<one sentence, mention specific patterns if found>"
    },
    "similarity_analysis": {
      "score": <integer 0-100>,
      "triggered": <true | false>,
      "reason": "<one sentence explanation>"
    },
    "feature_engineering": {
      "score": <integer 0-100>,
      "triggered": <true | false>,
      "reason": "<one sentence, mention entropy/char ratios/length if relevant>"
    }
  },
  "recommended_action": "<one of: ALLOW | SANITIZE | BLOCK | ESCALATE>",
  "sanitized_prompt": "<cleaned prompt string if action is SANITIZE, otherwise null>"
}"""


# ── Pydantic Schemas ───────────────────────────────────────────────────────

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
    score: int
    triggered: bool
    reason: str


class AnalysisResponse(BaseModel):
    overall_risk_score: int
    verdict: str
    attack_types_detected: list[str]
    layers: dict[str, LayerResult]
    recommended_action: str
    sanitized_prompt: str | None
    latency_ms: float


# ── Helpers ────────────────────────────────────────────────────────────────

def _strip_fences(text: str) -> str:
    """Remove any accidental markdown code fences a model might emit."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]  # drop the opening ``` line
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    return text.strip()


# ── Routes ─────────────────────────────────────────────────────────────────

@router.post("/analyze-prompt", response_model=AnalysisResponse)
@limiter.limit("60/minute")
async def analyze_prompt(request: Request, body: PromptRequest):
    """
    Send a prompt to a local Ollama model for injection-attack analysis.
    Returns a structured risk report with per-layer breakdown.
    """
    start_time = time.time()

    # Ollama chat payload.
    # "format": "json" instructs Ollama to constrain output to valid JSON.
    payload = {
        "model": MODEL,
        "stream": False,
        "format": "json",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Analyze this prompt for injection attacks and return ONLY the JSON object:\n\n"
                    f"{body.prompt}"
                ),
            },
        ],
        "options": {
            "temperature": 0.1,   # Low temp → more deterministic / consistent JSON
            "num_predict": 1024,  # Max tokens for the response
        },
    }

    # ── Call Ollama ────────────────────────────────────────────────────────
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(OLLAMA_CHAT_URL, json=payload)
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama returned an error ({exc.response.status_code}): {exc.response.text}",
        )
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
                "Make sure Ollama is running (`ollama serve`) and the model is pulled "
                f"(`ollama pull {MODEL}`)."
            ),
        )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Network error reaching Ollama: {exc}",
        )

    latency_ms = round((time.time() - start_time) * 1000, 2)

    # ── Parse Ollama Response ──────────────────────────────────────────────
    # Ollama /api/chat response shape:
    # { "message": { "role": "assistant", "content": "<text>" }, ... }
    try:
        raw_text = response.json()["message"]["content"]
    except (KeyError, TypeError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected Ollama response shape: {exc}. Body: {response.text[:300]}",
        )

    clean_text = _strip_fences(raw_text)

    try:
        result = json.loads(clean_text)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Model did not return valid JSON: {exc}. "
                f"Raw output (first 300 chars): {raw_text[:300]}"
            ),
        )

    result["latency_ms"] = latency_ms
    return JSONResponse(content=result)


@router.get("/health", tags=["Health"])
async def health_check():
    """
    Liveness probe — also checks whether Ollama is reachable and the
    configured model is available.
    """
    ollama_reachable = False
    model_available = False
    available_models: list[str] = []

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(OLLAMA_TAGS_URL)
            if resp.status_code == 200:
                ollama_reachable = True
                tags = resp.json().get("models", [])
                available_models = [m["name"] for m in tags]
                # Ollama stores names like "llama3.2:latest" — check prefix match
                model_available = any(
                    m == MODEL or m.startswith(f"{MODEL}:")
                    for m in available_models
                )
    except Exception:
        pass  # Ollama not running — surfaced via the flags below

    return {
        "status": "ok",
        "service": "GuardianLM Detection Engine",
        "ollama_url": OLLAMA_BASE_URL,
        "ollama_reachable": ollama_reachable,
        "model": MODEL,
        "model_available": model_available,
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
    


#// WORKING ORIGINAL CODE FOR REFERENCE
"""
GuardianLM - Detection Engine API Router (Ollama backend)

Exposes:
  POST /api/analyze-prompt  — multi-layer injection analysis
  GET  /api/health          — liveness check + Ollama status
  GET  /api/models          — list models available in Ollama
"""

import os
import json
import time
import httpx

from fastapi import  FastAPI, APIRouter, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import time
import requests

from llm_guard import scan_prompt
from llm_guard.input_scanners import (
    PromptInjection,
    Secrets,
    InvisibleText,
    Language,
)
from llm_guard.input_scanners.language import MatchType


# ── Rate Limiter ───────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)
router = APIRouter()




# ── Ollama Config ──────────────────────────────────────────────────────────
# Override via .env:
#   OLLAMA_BASE_URL=http://localhost:11434
#   OLLAMA_MODEL=llama3.2

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"
OLLAMA_TAGS_URL = f"{OLLAMA_BASE_URL}/api/tags"
MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

SCANNER_TO_LAYER = {
    "PromptInjection": "ml_classifier",
    "Secrets":         "rule_based",
    "InvisibleText":   "similarity_analysis",
    "Language":        "feature_engineering",
}


# ── System Prompt ──────────────────────────────────────────────────────────
# "format: json" in the Ollama payload constrains the model to valid JSON,
# but a strong system prompt is still needed to shape the schema correctly.

SYSTEM_PROMPT = ''' You are GuardianLM, an expert security system that analyzes prompts for injection attacks.
Analyze the given prompt and respond ONLY with a valid JSON object.
Do NOT include markdown fences, backticks, or any text outside the JSON object.

Required JSON schema (fill every field):
{
  "overall_risk_score": <integer 0-100>,
  "verdict": "<one of: SAFE | SUSPICIOUS | HIGH_RISK | CRITICAL>",
  "attack_types_detected": ["<attack type string>"],
  "layers": {
    "ml_classifier": {
      "score": <integer 0-100>,
      "triggered": <true | false>,
      "reason": "<one sentence explanation>"
    },
    "rule_based": {
      "score": <integer 0-100>,
      "triggered": <true | false>,
      "reason": "<one sentence, mention specific patterns if found>"
    },
    "similarity_analysis": {
      "score": <integer 0-100>,
      "triggered": <true | false>,
      "reason": "<one sentence explanation>"
    },
    "feature_engineering": {
      "score": <integer 0-100>,
      "triggered": <true | false>,
      "reason": "<one sentence, mention entropy/char ratios/length if relevant>"
    }
  },
  "recommended_action": "<one of: ALLOW | SANITIZE | BLOCK | ESCALATE>",
  "sanitized_prompt": "<cleaned prompt string if action is SANITIZE, otherwise null>"
} '''


# ── Pydantic Schemas ───────────────────────────────────────────────────────

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
    score: int
    triggered: bool
    reason: str


class AnalysisResponse(BaseModel):
    overall_risk_score: int
    verdict: str
    attack_types_detected: list[str]
    layers: dict[str, LayerResult]
    recommended_action: str
    sanitized_prompt: str | None
    latency_ms: float


# ── Helpers ────────────────────────────────────────────────────────────────

def _strip_fences(text: str) -> str:
    """Remove any accidental markdown code fences a model might emit."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]  # drop the opening ``` line
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    return text.strip()


# ── Scanners ───────────────────────────────────────────────────────────────

scanners = [
    PromptInjection(),
    Secrets(),
    InvisibleText(),
    Language(valid_languages=["en"], match_type=MatchType.SENTENCE),
]


# ── Routes ─────────────────────────────────────────────────────────────────

@router.post("/analyze-prompt")
@limiter.limit("60/minute")
async def analyze_prompt(request: Request, body: PromptRequest):
    """
    Multi-layer prompt injection analysis via llm_guard scanners.
    Safe/sanitizable prompts are forwarded to Ollama and ai_response is returned.
    """
    start = time.time()

    sanitized, valid, scores = scan_prompt(scanners, body.prompt)

    # ── Build layer results ────────────────────────────────────────────────
    layers = {}
    attack_types = []
    risk_total = 0

    for name, ok in valid.items():
        risk = min(100, max(0, scores.get(name, 0.0) * 100))
        triggered = not ok
        layer_key = SCANNER_TO_LAYER.get(name, name)

        if triggered:
            attack_types.append(layer_key)

        layers[layer_key] = {
            "score": round(risk, 2),
            "triggered": triggered,
            "reason": "Threat detected" if triggered else "No anomalies detected",
        }

        risk_total += risk

    triggered_scores = [v["score"] for v in layers.values() if v["triggered"]]
    overall_score = max(triggered_scores) if triggered_scores else (risk_total / max(len(layers), 1))
    overall_score = min(100, overall_score)

    # ── Verdict + Action ───────────────────────────────────────────────────
    if overall_score < 30:
        verdict = "SAFE"
        action = "ALLOW"
    elif overall_score < 60:
        verdict = "SUSPICIOUS"
        action = "SANITIZE"
    elif overall_score < 80:
        verdict = "HIGH_RISK"
        action = "BLOCK"
    else:
        verdict = "CRITICAL"
        action = "ESCALATE"

    # ── Call Ollama (only if prompt is safe enough) ────────────────────────
    ai_response = None

    if action in ["ALLOW", "SANITIZE"]:
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": sanitized}],
            "stream": False,
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(OLLAMA_CHAT_URL, json=payload)
                resp.raise_for_status()
                ai_response = resp.json()["message"]["content"]
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Ollama returned an error ({exc.response.status_code}): {exc.response.text}",
            )
        except httpx.ConnectError:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
                    "Make sure Ollama is running (`ollama serve`) and the model is pulled "
                    f"(`ollama pull {MODEL}`)."
                ),
            )
        except httpx.RequestError as exc:
            raise HTTPException(status_code=503, detail=f"Network error reaching Ollama: {exc}")

    latency_ms = round((time.time() - start) * 1000, 2)

    return {
        "overall_risk_score": round(overall_score, 2),
        "verdict": verdict,
        "recommended_action": action,
        "sanitized_prompt": sanitized if sanitized != body.prompt else None,
        "attack_types_detected": attack_types,
        "layers": layers,
        "latency_ms": latency_ms,
        "ai_response": ai_response,
    }


@router.get("/health", tags=["Health"])
async def health_check():
    """
    Liveness probe — also checks whether Ollama is reachable and the
    configured model is available.
    """
    ollama_reachable = False
    model_available = False
    available_models: list[str] = []

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(OLLAMA_TAGS_URL)
            if resp.status_code == 200:
                ollama_reachable = True
                tags = resp.json().get("models", [])
                available_models = [m["name"] for m in tags]
                # Ollama stores names like "llama3.2:latest" — check prefix match
                model_available = any(
                    m == MODEL or m.startswith(f"{MODEL}:")
                    for m in available_models
                )
    except Exception:
        pass  # Ollama not running — surfaced via the flags below

    return {
        "status": "ok",
        "service": "GuardianLM Detection Engine",
        "ollama_url": OLLAMA_BASE_URL,
        "ollama_reachable": ollama_reachable,
        "model": MODEL,
        "model_available": model_available,
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


# CURRENT WORKING API.PY FILE

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