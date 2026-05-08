"""FastAPI application for real-time live translation using ADK Gemini Live API."""

import asyncio
import logging
import warnings
from pathlib import Path

import csv
import io

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Load environment variables from .env file BEFORE importing agent
load_dotenv(Path(__file__).parent / ".env")

# Ensure non-Vertex AI mode for Gemini API key auth
# These env vars cause the SDK to route through aiplatform.googleapis.com
import os

os.environ.pop("GOOGLE_GENAI_USE_VERTEXAI", None)
os.environ.pop("GOOGLE_CLOUD_PROJECT", None)
os.environ.pop("GOOGLE_CLOUD_LOCATION", None)

# Patch ADK to use v1beta for Gemini API live connections.
# ADK (as of 1.32.0) still defaults `_live_api_version` to "v1alpha" for AI
# Studio API-key auth, but `gemini-3.1-flash-live-preview` is only on v1beta.
# See google_llm.py:_live_api_version. Tracked in google/adk-python#5075.
from google.adk.models.google_llm import Gemini

Gemini._live_api_version = "v1beta"

# Import agent after loading environment variables
# pylint: disable=wrong-import-position
import sys

sys.path.insert(0, str(Path(__file__).parent))
from translator_agent.agent import (  # noqa: E402
    DICT_PATH,
    LANGUAGES,
    POPULAR_LANGUAGES,
    agent,
    create_agent,
    load_glossary_pairs,
)

MAX_GLOSSARY_BYTES = 256 * 1024  # 256 KB cap on uploaded CSV

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress Pydantic serialization warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

APP_NAME = "live-translation"

# ========================================
# Phase 1: Application Initialization
# ========================================

app = FastAPI()

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

session_service = InMemorySessionService()
runner = Runner(app_name=APP_NAME, agent=agent, session_service=session_service)


@app.get("/")
async def root():
    """Serve the index.html page."""
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/api/languages")
async def get_languages():
    """Return available languages with popular ones highlighted."""
    return {"languages": LANGUAGES, "popular": POPULAR_LANGUAGES}


@app.get("/api/glossary")
async def get_glossary():
    """Return the currently active glossary pairs."""
    pairs = load_glossary_pairs()
    return {"pairs": [{"source": s, "target": t} for s, t in pairs]}


@app.post("/api/glossary")
async def upload_glossary(file: UploadFile):
    """Replace the glossary with the uploaded CSV (source,target per line)."""
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must have a .csv extension.")

    raw = await file.read()
    if len(raw) > MAX_GLOSSARY_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {MAX_GLOSSARY_BYTES} bytes.",
        )

    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=400, detail=f"CSV must be UTF-8: {exc}"
        ) from exc

    pairs: list[tuple[str, str]] = []
    for line_no, row in enumerate(csv.reader(io.StringIO(text)), start=1):
        if not row or all(not c.strip() for c in row):
            continue
        if len(row) < 2 or not row[0].strip() or not row[1].strip():
            raise HTTPException(
                status_code=400,
                detail=f"Line {line_no} must be 'source,target' with both fields.",
            )
        pairs.append((row[0].strip(), row[1].strip()))

    try:
        with open(DICT_PATH, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(pairs)
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not write glossary file: {exc}",
        ) from exc

    logger.info("Glossary updated: %d entries", len(pairs))
    return {
        "pairs": [{"source": s, "target": t} for s, t in pairs],
        "applies_on": "next-session",
    }


@app.websocket("/ws/{user_id}/{session_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    user_id: str,
    session_id: str,
    source: str = "en",
    target: str = "ja",
) -> None:
    """WebSocket endpoint for bidirectional streaming translation."""
    logger.debug(
        f"WebSocket connection request: user_id={user_id}, session_id={session_id}, "
        f"source={source}, target={target}"
    )
    await websocket.accept()
    logger.debug("WebSocket connection accepted")

    # ========================================
    # Phase 2: Session Initialization
    # ========================================

    # Create per-connection agent and runner for the selected language pair
    connection_agent = create_agent(source, target)
    connection_runner = Runner(
        app_name=APP_NAME, agent=connection_agent, session_service=session_service
    )

    model_name = connection_agent.model
    # Native audio models: contain "native-audio" or "live-preview" in name
    is_native_audio = (
        "native-audio" in model_name.lower() or "live-preview" in model_name.lower()
    )

    if is_native_audio:
        run_config = RunConfig(
            streaming_mode=StreamingMode.BIDI,
            response_modalities=["AUDIO"],
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            session_resumption=types.SessionResumptionConfig(),
        )
        logger.debug(
            f"Native audio model detected: {model_name}, using AUDIO response modality"
        )
    else:
        run_config = RunConfig(
            streaming_mode=StreamingMode.BIDI,
            response_modalities=["TEXT"],
            input_audio_transcription=None,
            output_audio_transcription=None,
            session_resumption=types.SessionResumptionConfig(),
        )
        logger.debug(
            f"Half-cascade model detected: {model_name}, using TEXT response modality"
        )

    session = await session_service.get_session(
        app_name=APP_NAME, user_id=user_id, session_id=session_id
    )
    if not session:
        await session_service.create_session(
            app_name=APP_NAME, user_id=user_id, session_id=session_id
        )

    live_request_queue = LiveRequestQueue()

    # ========================================
    # Phase 3: Active Session
    # ========================================

    async def upstream_task() -> None:
        """Receives messages from WebSocket and sends to LiveRequestQueue."""
        logger.debug("upstream_task started")
        while True:
            message = await websocket.receive()

            if "bytes" in message:
                audio_data = message["bytes"]
                logger.debug(f"Received binary audio chunk: {len(audio_data)} bytes")
                audio_blob = types.Blob(
                    mime_type="audio/pcm;rate=16000", data=audio_data
                )
                live_request_queue.send_realtime(audio_blob)

            elif "text" in message:
                logger.debug("Ignoring text message (translator is audio-only)")

    async def downstream_task() -> None:
        """Receives Events from run_live() and sends to WebSocket."""
        logger.debug("downstream_task started")
        async for event in connection_runner.run_live(
            user_id=user_id,
            session_id=session_id,
            live_request_queue=live_request_queue,
            run_config=run_config,
        ):
            event_json = event.model_dump_json(exclude_none=True, by_alias=True)
            logger.debug(f"[SERVER] Event: {event_json}")
            await websocket.send_text(event_json)
        logger.debug("run_live() generator completed")

    try:
        await asyncio.gather(upstream_task(), downstream_task())
    except WebSocketDisconnect:
        logger.debug("Client disconnected normally")
    except Exception as e:
        logger.error(f"Unexpected error in streaming tasks: {e}", exc_info=True)
    finally:
        # ========================================
        # Phase 4: Session Termination
        # ========================================
        logger.debug("Closing live_request_queue")
        live_request_queue.close()
