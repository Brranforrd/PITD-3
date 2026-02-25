from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from Api import router

app = FastAPI(
    title="GuardianLM API",
    description="LLM Prompt Injection Detection System",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

@app.get("/")
def root():
    return {"status": "online", "system": "GuardianLM v1.0"}

