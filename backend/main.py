from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Literal
from conversation import resolve_language

import os
import json
import traceback
import logging
from time import perf_counter

import retrieve


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Bharat AI Personal Assistant API",
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

AUDIO_DIR = os.path.join(
    BASE_DIR,
    "audio"
)

os.makedirs(
    AUDIO_DIR,
    exist_ok=True
)


# ============================================================
# STATIC AUDIO
# ============================================================

app.mount(
    "/audio",
    StaticFiles(directory=AUDIO_DIR),
    name="audio",
)


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def load_resources():

    print()
    print("=" * 60)
    print("BHARAT AI STARTING")
    print("=" * 60)

    try:

        retrieve.get_vector_store()

        status = retrieve.get_vector_store_status()

        print(
            f"Knowledge ready    : {status['ready']}"
        )

        print(
            f"Knowledge documents: {status['document_count']}"
        )

        retrieve.run_retrieval_smoke_test()

        print("Vector database loaded.")
        print("Bharat AI resources loaded.")

    except Exception as error:

        print()
        print("STARTUP ERROR")
        print("-" * 60)
        print(str(error))
        print("-" * 60)

        traceback.print_exc()

        # Do not let Railway report a healthy deployment when the
        # Bharat knowledge database is missing, empty, or unreadable.
        raise

    print("=" * 60)


# ============================================================
# REQUEST MODEL
# ============================================================

class ChatRequest(BaseModel):

    question: str

    voice: bool = False

    language: Literal["en", "hi", "ne"] = "en"

    session_id: str = "default"


# ============================================================
# STREAM EVENT
# ============================================================

def create_token_event(token: str):

    return (
        json.dumps(
            {
                "type": "token",
                "token": token,
            },
            ensure_ascii=False,
        )
        + "\n"
    )


def create_done_event():

    return (
        json.dumps(
            {
                "type": "done",
            },
            ensure_ascii=False,
        )
        + "\n"
    )


def create_error_event(message: str):

    return (
        json.dumps(
            {
                "type": "error",
                "error": message,
                "token": message,
            },
            ensure_ascii=False,
        )
        + "\n"
    )


# ============================================================
# STREAM GENERATOR
# ============================================================

def generate_stream(question: str, session_id: str, voice: bool = False, language: str = "en"):
    language = resolve_language(question, language)
    started = perf_counter()
    metrics = {}
    stream = retrieve.stream_answer(question, session_id, voice=voice, metrics=metrics, language=language)
    try:
        yield json.dumps({"type": "language", "language": language}) + "\n"
        for chunk in stream:
            if not chunk:
                continue
            if "first_token_ms" not in metrics:
                metrics["first_token_ms"] = round((perf_counter() - started) * 1000)
            yield create_token_event(str(chunk))
    except Exception:
        logging.exception("Answer streaming failed")
        yield create_error_event("I'm having trouble processing that request right now. Please try again.")
    finally:
        stream.close()
        metrics["total_ms"] = round((perf_counter() - started) * 1000)
        logging.getLogger("uvicorn.error").info("Answer timing: %s", metrics)
    yield json.dumps({"type": "metrics", **metrics}) + "\n"
    yield create_done_event()


# ============================================================
# ASK
# ============================================================

@app.post("/ask")
async def ask_question(
    request: ChatRequest,
):

    # ========================================================
    # CLEAN QUESTION
    # ========================================================

    question = (
        request.question or ""
    ).strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    # ========================================================
    # SESSION
    # ========================================================

    session_id = (
        request.session_id or "default"
    ).strip()

    if not session_id:
        session_id = "default"

    # ========================================================
    # LOG REQUEST
    # ========================================================

    print()
    print("=" * 60)
    print("NEW STREAMING QUESTION")
    print("=" * 60)

    print(
        f"Question: {question}"
    )

    print(
        f"Session: {session_id}"
    )

    print(
        f"Voice: {request.voice}"
    )

    print("=" * 60)

    # ========================================================
    # STREAM RESPONSE
    # ========================================================

    return StreamingResponse(

        generate_stream(
            question=question,
            session_id=session_id,
            voice=request.voice,
            language=request.language,
        ),

        # Your Assistant.tsx already reads the response
        # line-by-line, so NDJSON is correct.

        media_type="application/x-ndjson",

        headers={
            "Cache-Control": (
                "no-cache, "
                "no-store, "
                "must-revalidate"
            ),

            "Pragma": "no-cache",

            "Expires": "0",

            "X-Accel-Buffering": "no",

            "Connection": "keep-alive",
        },
    )


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health_check():

    vector_status = (
        retrieve.get_vector_store_status()
    )

    return {
        "status": (
            "ok"
            if vector_status["ready"]
            else "degraded"
        ),
        "service": "Bharat AI",
        "streaming": True,
        "knowledge_ready": vector_status["ready"],
        "knowledge_documents": vector_status["document_count"],
        "collection": vector_status["collection"],
        "environment": os.getenv(
            "RAILWAY_ENVIRONMENT_NAME",
            "local"
        ),
    }


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "service": "Bharat AI Personal Assistant API",
        "status": "running",
        "streaming": True,
        "endpoints": {
            "ask": "/ask",
            "health": "/health",
            "audio": "/audio",
        },
    }