"""
GuardianLM - FastAPI Application Entry Point (Ollama backend)

Run from the project root:
    uvicorn BackEnd.main:app --reload --port 8000

Or from within the BackEnd directory:
    uvicorn main:app --reload --port 8000

Required env vars (set in BackEnd/.env):
    OLLAMA_BASE_URL   default: http://localhost:11434
    OLLAMA_MODEL      default: llama3.2
"""

import sys
import os

# ── Load .env from the BackEnd directory if present ────────────────────────
from pathlib import Path
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=env_file)

# ── Ensure imports work whether launched from project root or BackEnd/ ──────
sys.path.insert(0, os.path.dirname(__file__))

try:
    from BackEnd.api import router, limiter  # launched as: uvicorn BackEnd.main:app
except ImportError:
    from api import router, limiter          # launched as: uvicorn main:app (from BackEnd/)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded


# ── Application Factory ────────────────────────────────────────────────────

app = FastAPI(
    title="GuardianLM API",
    description="LLM Prompt Injection Detection System — powered by Ollama",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# ── Rate Limiting ──────────────────────────────────────────────────────────

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ── CORS Middleware ────────────────────────────────────────────────────────
# Allows the React dev server (Vite default: 5173) to reach the API.

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routers ────────────────────────────────────────────────────────────────

app.include_router(router, prefix="/api")


# ── Root Endpoint ──────────────────────────────────────────────────────────

@app.get("/", tags=["Root"])
def root():
    return {
        "status": "online",
        "system": "GuardianLM v1.0",
        "backend": "Ollama (local)",
        "docs": "/docs",
    }