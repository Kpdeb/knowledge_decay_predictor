"""
Knowledge Decay Predictor — FastAPI Backend
Entry point: registers all routers and middleware.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

from routes import topics, quiz, predictions, schedule, auth, users
from database import engine, Base

# Create all tables on startup
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Knowledge Decay Predictor API",
    description="Predict when students forget concepts and schedule smart revisions.",
    version="1.0.0",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(auth.router,        prefix="/auth",        tags=["Auth"])
app.include_router(users.router,       prefix="/users",       tags=["Users"])
app.include_router(topics.router,      prefix="/topics",      tags=["Topics"])
app.include_router(quiz.router,        prefix="/quiz",        tags=["Quiz"])
app.include_router(predictions.router, prefix="/prediction",  tags=["Predictions"])
app.include_router(schedule.router,    prefix="/schedule",    tags=["Schedule"])


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok", "service": "backend"}
