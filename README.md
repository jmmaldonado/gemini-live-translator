# ADK Live Translator

Real-time audio translation app powered by ADK Gemini Live API Toolkit. Speak in any language and get immediate audio translation in your chosen target language.

Supports 97 languages including English, Japanese, Chinese, Spanish, French, German, Portuguese, Korean, Hindi, Arabic, and many more.

![Demo](demo.gif)

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- [Gemini API key](https://aistudio.google.com/apikey)

## Setup

```bash
uv sync
```

Set your Gemini API key in `app/.env`:

```
GOOGLE_API_KEY=your-api-key
```


## Custom Glossary

Pin specific terms to a fixed translation so the model always renders them the same way. **The glossary is per browser** — it's stored in your browser's `localStorage` and sent to the server only when you start a session. Different visitors can run different glossaries at the same time without affecting each other; nothing is persisted server-side.

1. Click **Glossary** in the header. On first visit the modal seeds itself from the default glossary baked into the app (`app/dict.csv`). Entries are shown in a three-column table: **Source / Pronunciation / Transcript**.
2. Click **Choose .csv file** and pick a UTF-8 CSV (max 256 KB, max 1000 entries). Each line is `source,target[,transcription]`:

   ```csv
   Kubernetes,クバネティス,Kubernetes
   Cloud Run,クラウドラン,Cloud Run
   Gemini,ジェミニ,Gemini
   Vertex AI,バーテックスエーアイ,Vertex AI
   ```

   The optional **third column** is a display override. The model still pronounces the term using `target` (so the audio sounds right), but the on-screen transcript renders `transcription` in place of `target`. Useful for proper nouns where you want a phonetic pronunciation but a Latin display label. When omitted, the transcript shows `target` as-is. The replacement happens client-side, so it never affects what the model emits — only what you see.

3. Click **Load & replace**. The CSV is parsed in the browser and stored locally. Use **Reset to defaults** to discard your customisations and re-fetch the seed glossary from the server.
4. The new entries take effect on the **next** session — click **Start Audio** again, or change languages, to open a fresh WebSocket. Live sessions keep the glossary they started with.

Status feedback appears below the entry count: green for success, red for parse errors (a line missing the comma, file too large, etc.) — in the error case the previous glossary stays in place.

### Changing the default glossary

Edit `app/dict.csv` and redeploy. The endpoint `GET /api/glossary/defaults` returns those defaults. Browsers that already have a cached glossary in `localStorage` will keep using it — they need to either click **Reset to defaults** in the modal, or the app needs to bump the storage key (`GLOSSARY_KEY` in `app/static/js/app.js`) to force a one-time re-seed on next load.

## Run

```bash
uv run uvicorn app.main:app --reload
```

Open http://localhost:8000 and click **Start Audio** to begin translating.

## Architecture

Uses ADK's 4-phase bidi-streaming lifecycle over WebSocket:

1. **App init** — FastAPI server, default Agent (`gemini-3.1-flash-live-preview`), Runner, in-memory SessionService.
2. **Session init** — On WebSocket connect the server reads a JSON setup message (`{glossary: [...]}`) sent by the browser as the first frame, then constructs a per-connection Agent whose system instruction includes that glossary. RunConfig is set up with AUDIO modality, input/output transcription, and session resumption.
3. **Active streaming** — Concurrent upstream (mic → `LiveRequestQueue`) and downstream (`run_live` → WebSocket) tasks. The frontend swaps `target` → `transcription` on incoming output-transcription text before rendering.
4. **Termination** — `LiveRequestQueue.close()` on disconnect.

## Model

Uses `gemini-3.1-flash-live-preview` with the Gemini API (`generativelanguage.googleapis.com`). This is a native audio model supporting real-time audio input/output with transcription. The app sends audio via `realtime_input` (audio blobs).

The system instruction is built in `app/translator_agent/agent.py` as:

```
You are a real-time translator from {source} to {target}. Listen to the
incoming audio and immediately output the translated version in {target},
maintaining the speaker's original tone and urgency.

Use the following glossary for specific terms. When you hear these words,
always use the paired translation:
- <source> → <target>
...
```

Only the first two CSV columns reach the model; the third (transcript display) is purely a frontend post-processing rule.

## Deployment to Cloud Run

### 1. Prerequisites

- [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) (`gcloud`) installed and configured
- A Google Cloud project with Cloud Run API enabled

### 2. Configure Environment

Set your Gemini API key in `app/.env`:

```
GOOGLE_API_KEY=your-api-key
```

Export it before deploying:

```bash
set -a
source app/.env
set +a
```

### 3. Deploy

```bash
gcloud run deploy live-translation \
  --source . \
  --project YOUR_PROJECT \
  --region us-central1 \
  --allow-unauthenticated \
  --timeout 3600 \
  --min-instances 1 \
  --max-instances 1 \
  --set-env-vars "GOOGLE_API_KEY=${GOOGLE_API_KEY}"
```

Key flags:
- `--timeout 3600` — Live API sessions can be long-lived (up to 1 hour)
- `--min-instances 1` — avoids cold start latency for WebSocket connections
- `--max-instances 1` — sufficient for demo; increase for production


## ADK/SDK Compatibility Patches

ADK 1.32.0 still needs two small adjustments in `app/main.py` to talk to `gemini-3.1-flash-live-preview` over the Gemini API:

1. **Remove Vertex AI env vars** — The genai SDK auto-detects `GOOGLE_CLOUD_PROJECT`/`GOOGLE_CLOUD_LOCATION` and routes to `aiplatform.googleapis.com`. We pop these env vars to force Gemini API key routing via `generativelanguage.googleapis.com`. (SDK behavior, not an ADK bug.)

2. **API version `v1alpha` → `v1beta`** — ADK still defaults `Gemini._live_api_version` to `v1alpha` for AI Studio API-key auth, but `gemini-3.1-flash-live-preview` is only served on `v1beta`. Tracked in [google/adk-python#5075](https://github.com/google/adk-python/issues/5075).

Previously tracked issues that are **fixed upstream** and no longer require local workarounds:

- **Audio + text routing for Gemini 3.1 Live** ([#5018](https://github.com/google/adk-python/issues/5018)) — fixed in v1.29.0 (commit [`ee69661`](https://github.com/google/adk-python/commit/ee69661a616056fa89e0ec2188aaa59bd714d8c9)). ADK now routes input through `send_realtime_input` for the 3.1 model, so the prior "audio-only / no `client_content`" workaround is gone.
- **Session resumption / transparent reconnection** ([#4996](https://github.com/google/adk-python/issues/4996)) — fixed in v1.32.0. The reconnection loop in `base_llm_flow.py` now iterates on `ConnectionClosed` and recoverable `APIError`s, sets `session_resumption.transparent = True`, and skips replaying history when a resumption handle exists.
