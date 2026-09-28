from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import os
import json
import traceback

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
        print("Loading embedding model and ChromaDB...")

        retrieve.get_vector_store()

        print("Vector database loaded.")
        print("Bharat AI resources loaded.")

    except Exception as error:

        print()
        print("STARTUP WARNING")
        print("-" * 60)
        print(str(error))
        print("-" * 60)

        traceback.print_exc()

    print("=" * 60)


# ============================================================
# REQUEST MODEL
# ============================================================

class ChatRequest(BaseModel):

    question: str

    voice: bool = False

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
            },
            ensure_ascii=False,
        )
        + "\n"
    )


# ============================================================
# STREAM GENERATOR
# ============================================================

def generate_stream(
    question: str,
    session_id: str,
    voice: bool = False,
):

    print()
    print("=" * 60)
    print("SESSION:", session_id)
    print("=" * 60)

    try:

        # ====================================================
        # CALL RETRIEVAL + LLM STREAM
        # ====================================================

        stream = retrieve.stream_answer(
            question,
            session_id,
        )

        # ====================================================
        # FORWARD EVERY MODEL CHUNK IMMEDIATELY
        # ====================================================

        for chunk in stream:

            if chunk is None:
                continue

            if not isinstance(chunk, str):
                chunk = str(chunk)

            if not chunk:
                continue

            # IMPORTANT:
            #
            # No artificial delay.
            # No fake typing.
            # No word-by-word generation.
            #
            # The chunk comes directly from retrieve.py
            # and is immediately sent to the browser.

            yield create_token_event(chunk)

        # ====================================================
        # STREAM COMPLETE
        # ====================================================

        print()
        print("STREAM COMPLETED")
        print("=" * 60)

        yield create_done_event()

    except Exception as error:

        print()
        print("=" * 60)
        print("STREAMING ERROR")
        print("=" * 60)
        print(str(error))
        print("=" * 60)

        traceback.print_exc()

        yield create_error_event(
            "I'm having trouble processing that request right now."
        )

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

    return {
        "status": "ok",
        "service": "Bharat AI",
        "streaming": True,
        "environment": "local",
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