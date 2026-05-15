"""Long-running soak test for the live translator.

Generates random English sentences, converts to audio via Cloud TTS,
sends through the translator WebSocket, transcribes the response via
Cloud STT, and verifies semantic correctness with Gemini Flash Lite.

Runs on a single persistent WebSocket to exercise session resumption,
GoAway handling, and translation quality over extended periods.

Usage:
    uv run python test_long.py [--url ws://localhost:8000] [--duration 3600] [--source en] [--target ja]
"""

import argparse
import asyncio
import base64
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import websockets
from dotenv import load_dotenv
from google import genai
from google.cloud import speech, texttospeech

load_dotenv(Path(__file__).parent / "app" / ".env")

os.environ.pop("GOOGLE_GENAI_USE_VERTEXAI", None)
os.environ.pop("GOOGLE_CLOUD_PROJECT", None)
os.environ.pop("GOOGLE_CLOUD_LOCATION", None)

CHUNK_SIZE = 512
CHUNK_INTERVAL = 0.016
RESPONSE_TIMEOUT = 30
SILENCE_AFTER_SPEECH = 2.0
GENAI_MODEL = "gemini-2.5-flash-lite"

TOPICS = [
    "technology and software engineering",
    "travel and geography",
    "food and cooking",
    "business and finance",
    "science and nature",
    "sports and fitness",
    "art and music",
    "history and culture",
    "health and medicine",
    "education and learning",
    "weather and seasons",
    "daily life and routines",
    "news and current events",
    "philosophy and ethics",
]


@dataclass
class IterationResult:
    index: int
    original: str
    output_transcription: str | None = None
    stt_transcription: str | None = None
    passed: bool = False
    score: float = 0.0
    reason: str = ""
    error: str | None = None
    elapsed: float = 0.0


@dataclass
class Stats:
    iterations: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    total_score: float = 0.0
    results: list[IterationResult] = field(default_factory=list)


def stamp() -> str:
    return time.strftime("%H:%M:%S")


def generate_sentence(client: genai.Client, topic: str) -> str:
    resp = client.models.generate_content(
        model=GENAI_MODEL,
        contents=(
            f"Generate exactly one natural English sentence (10-20 words) about "
            f"{topic}. Output only the sentence, no quotes or explanation."
        ),
    )
    return resp.text.strip().strip('"')


def text_to_pcm(tts_client: texttospeech.TextToSpeechClient, text: str) -> bytes:
    resp = tts_client.synthesize_speech(
        input=texttospeech.SynthesisInput(text=text),
        voice=texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Neural2-J",
        ),
        audio_config=texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
        ),
    )
    # Strip the 44-byte WAV header to get raw PCM
    audio = resp.audio_content
    if audio[:4] == b"RIFF":
        audio = audio[44:]
    # Pad 1s silence before and after for VAD
    silence = b"\x00\x00" * 16000
    return silence + audio + silence


def pcm_to_text(
    stt_client: speech.SpeechClient,
    pcm_data: bytes,
    sample_rate: int,
    language: str,
) -> str:
    resp = stt_client.recognize(
        config=speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate,
            language_code=language,
            enable_automatic_punctuation=True,
        ),
        audio=speech.RecognitionAudio(content=pcm_data),
    )
    return " ".join(r.alternatives[0].transcript for r in resp.results if r.alternatives)


def verify_translation(
    client: genai.Client,
    original: str,
    translated: str,
    source: str,
    target: str,
) -> tuple[bool, float, str]:
    resp = client.models.generate_content(
        model=GENAI_MODEL,
        contents=(
            f"You are a translation quality evaluator. Compare the original "
            f"{source} sentence with its {target} translation.\n\n"
            f"Original ({source}): {original}\n"
            f"Translation ({target}): {translated}\n\n"
            f"Score the semantic accuracy from 0 to 10 (10 = perfect). "
            f"Reply in exactly this format:\n"
            f"SCORE: <number>\n"
            f"PASS: <yes/no>\n"
            f"REASON: <one sentence>"
        ),
    )
    text = resp.text.strip()
    score = 0.0
    passed = False
    reason = text
    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith("SCORE:"):
            try:
                score = float(line.split(":", 1)[1].strip().split("/")[0])
            except ValueError:
                pass
        elif line.upper().startswith("PASS:"):
            passed = "yes" in line.lower()
        elif line.upper().startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()
    return passed, score, reason


LANG_TO_STT = {
    "ja": "ja-JP",
    "en": "en-US",
    "zh": "zh-CN",
    "es": "es-ES",
    "fr": "fr-FR",
    "de": "de-DE",
    "ko": "ko-KR",
    "pt": "pt-BR",
    "hi": "hi-IN",
    "ar": "ar-SA",
}


async def run_iteration(
    ws,
    genai_client: genai.Client,
    tts_client: texttospeech.TextToSpeechClient,
    stt_client: speech.SpeechClient,
    index: int,
    source: str,
    target: str,
) -> IterationResult:
    topic = TOPICS[index % len(TOPICS)]
    t0 = time.monotonic()

    try:
        sentence = generate_sentence(genai_client, topic)
    except Exception as e:
        return IterationResult(index=index, original="", error=f"generate: {e}")

    try:
        pcm_data = text_to_pcm(tts_client, sentence)
    except Exception as e:
        return IterationResult(index=index, original=sentence, error=f"tts: {e}")

    output_transcription_final: list[str] = []
    output_transcription_partial: list[str] = []
    audio_chunks: list[bytes] = []
    turn_complete = asyncio.Event()

    async def receive_responses():
        try:
            while not turn_complete.is_set():
                raw = await asyncio.wait_for(ws.recv(), timeout=RESPONSE_TIMEOUT)
                msg = json.loads(raw)

                ot = msg.get("outputTranscription")
                if ot and ot.get("text"):
                    if ot.get("finished"):
                        output_transcription_final.append(ot["text"])
                    else:
                        output_transcription_partial.append(ot["text"])

                content = msg.get("content", {})
                for part in content.get("parts", []):
                    inline = part.get("inlineData")
                    if inline and inline.get("data"):
                        audio_chunks.append(base64.b64decode(inline["data"]))

                if msg.get("turnComplete"):
                    turn_complete.set()
        except asyncio.TimeoutError:
            pass
        except websockets.ConnectionClosed:
            pass

    recv_task = asyncio.create_task(receive_responses())

    # Send audio
    offset = 0
    while offset < len(pcm_data):
        chunk = pcm_data[offset : offset + CHUNK_SIZE]
        try:
            await ws.send(chunk)
        except websockets.ConnectionClosed:
            recv_task.cancel()
            return IterationResult(
                index=index, original=sentence, error="ws closed during send"
            )
        offset += CHUNK_SIZE
        await asyncio.sleep(CHUNK_INTERVAL)

    # Trailing silence for VAD
    silence = b"\x00" * CHUNK_SIZE
    for _ in range(int(SILENCE_AFTER_SPEECH / CHUNK_INTERVAL)):
        try:
            await ws.send(silence)
        except websockets.ConnectionClosed:
            break
        await asyncio.sleep(CHUNK_INTERVAL)

    # Wait for response
    try:
        await asyncio.wait_for(turn_complete.wait(), timeout=RESPONSE_TIMEOUT)
    except asyncio.TimeoutError:
        pass

    recv_task.cancel()
    try:
        await recv_task
    except asyncio.CancelledError:
        pass

    output_text = (
        output_transcription_final[-1]
        if output_transcription_final
        else "".join(output_transcription_partial) or None
    )

    # STT on returned audio
    stt_text = None
    if audio_chunks:
        combined_pcm = b"".join(audio_chunks)
        stt_lang = LANG_TO_STT.get(target, f"{target}-{target.upper()}")
        try:
            stt_text = pcm_to_text(stt_client, combined_pcm, 24000, stt_lang)
        except Exception as e:
            stt_text = f"(stt error: {e})"

    # Verify translation
    translated = output_text or stt_text or ""
    if not translated:
        return IterationResult(
            index=index,
            original=sentence,
            output_transcription=output_text,
            stt_transcription=stt_text,
            error="no response",
            elapsed=time.monotonic() - t0,
        )

    try:
        passed, score, reason = verify_translation(
            genai_client, sentence, translated, source, target
        )
    except Exception as e:
        return IterationResult(
            index=index,
            original=sentence,
            output_transcription=output_text,
            stt_transcription=stt_text,
            error=f"verify: {e}",
            elapsed=time.monotonic() - t0,
        )

    return IterationResult(
        index=index,
        original=sentence,
        output_transcription=output_text,
        stt_transcription=stt_text,
        passed=passed,
        score=score,
        reason=reason,
        elapsed=time.monotonic() - t0,
    )


async def main():
    parser = argparse.ArgumentParser(description="Long-running translation soak test")
    parser.add_argument("--url", default="ws://localhost:8000", help="WebSocket base URL")
    parser.add_argument("--duration", type=int, default=3600, help="Test duration in seconds")
    parser.add_argument("--source", default="en", help="Source language code")
    parser.add_argument("--target", default="ja", help="Target language code")
    args = parser.parse_args()

    ws_url = f"{args.url}/ws/soak-test/soak-session-001?source={args.source}&target={args.target}"

    genai_client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    tts_client = texttospeech.TextToSpeechClient()
    stt_client = speech.SpeechClient()

    stats = Stats()
    start = time.monotonic()

    print(f"[{stamp()}] Starting soak test: {args.source} -> {args.target}, duration={args.duration}s")
    print(f"[{stamp()}] Connecting to {ws_url}")

    async with websockets.connect(ws_url) as ws:
        await ws.send(json.dumps({"glossary": []}))
        print(f"[{stamp()}] Connected, setup sent")

        while time.monotonic() - start < args.duration:
            stats.iterations += 1
            result = await run_iteration(
                ws, genai_client, tts_client, stt_client,
                stats.iterations, args.source, args.target,
            )
            stats.results.append(result)

            if result.error:
                stats.errors += 1
                print(
                    f"[{stamp()}] #{result.index} ERROR ({result.elapsed:.1f}s) | "
                    f'"{result.original[:50]}" | {result.error}'
                )
            elif result.passed:
                stats.passed += 1
                stats.total_score += result.score
                display = result.output_transcription or result.stt_transcription or ""
                if len(display) > 40:
                    display = display[:37] + "..."
                print(
                    f"[{stamp()}] #{result.index} PASS ({result.score:.0f}/10) "
                    f'({result.elapsed:.1f}s) | "{result.original[:50]}" -> "{display}"'
                )
            else:
                stats.failed += 1
                stats.total_score += result.score
                display = result.output_transcription or result.stt_transcription or ""
                if len(display) > 40:
                    display = display[:37] + "..."
                print(
                    f"[{stamp()}] #{result.index} FAIL ({result.score:.0f}/10) "
                    f'({result.elapsed:.1f}s) | "{result.original[:50]}" -> "{display}"'
                    f" | {result.reason}"
                )

            elapsed = time.monotonic() - start
            remaining = args.duration - elapsed
            if remaining > 0:
                print(
                    f"         [{elapsed:.0f}s / {args.duration}s elapsed, "
                    f"{remaining:.0f}s remaining]",
                    flush=True,
                )

    # Summary
    elapsed = time.monotonic() - start
    scored = stats.passed + stats.failed
    avg_score = stats.total_score / scored if scored else 0
    print(f"\n[{stamp()}] === SUMMARY ===")
    print(
        f"Duration: {elapsed:.0f}s | Iterations: {stats.iterations} | "
        f"Passed: {stats.passed}/{stats.iterations} "
        f"({100 * stats.passed / stats.iterations:.1f}%) | "
        f"Avg score: {avg_score:.1f}/10 | Errors: {stats.errors}"
    )

    sys.exit(0 if stats.errors == 0 and stats.passed > 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
