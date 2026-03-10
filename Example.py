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