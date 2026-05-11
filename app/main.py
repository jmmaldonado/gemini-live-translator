"""FastAPI application for real-time live translation using ADK Gemini Live API."""

import asyncio
import json
import logging
import warnings
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
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
    LANGUAGES,
    POPULAR_LANGUAGES,
    agent,
    create_agent,
    load_default_glossary,
)

MAX_GLOSSARY_ENTRIES = 1000  # safety cap on per-session glossary length
SETUP_TIMEOUT_SEC = 5  # how long to wait for the client's setup message

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


@app.get("/api/glossary/defaults")
async def get_default_glossary():
    """Return the seed glossary baked into the image (used when localStorage is empty)."""
    pairs = load_default_glossary()
    return {"pairs": [{"source": s, "target": t} for s, t in pairs]}


def _parse_setup(raw: str) -> list[tuple[str, str]]:
    """Parse the client's setup message and return validated glossary pairs."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    pairs: list[tuple[str, str]] = []
    for entry in (data.get("glossary") or [])[:MAX_GLOSSARY_ENTRIES]:
        if not isinstance(entry, dict):
            continue
        src = (entry.get("source") or "").strip()
        tgt = (entry.get("target") or "").strip()
        if src and tgt:
            pairs.append((src, tgt))
    return pairs


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

    # Wait for the client's setup message (carries the per-session glossary).
    # Falls back to the on-disk default glossary if the client doesn't send one
    # within SETUP_TIMEOUT_SEC (older clients, network hiccups).
    glossary_pairs: list[tuple[str, str]] | None = None
    try:
        setup_raw = await asyncio.wait_for(
            websocket.receive_text(), timeout=SETUP_TIMEOUT_SEC
        )
        glossary_pairs = _parse_setup(setup_raw)
        logger.debug("Setup received: %d glossary entries", len(glossary_pairs))
    except asyncio.TimeoutError:
        logger.warning(
            "No setup message within %ds; using default glossary.", SETUP_TIMEOUT_SEC
        )
    except WebSocketDisconnect:
        logger.debug("Client disconnected before sending setup")
        return

    # Create per-connection agent and runner for the selected language pair
    connection_agent = create_agent(source, target, glossary_pairs)
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
