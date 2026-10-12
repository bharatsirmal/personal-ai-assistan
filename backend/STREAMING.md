# Text and voice streaming

The frontend speaks completed sentences while the model continues generating.
Speech uses the browser's installed voices. A Stop response button cancels the
fetch and queued speech; the microphone can also interrupt a response. Voice
conversation resumes listening only after both text generation and playback end.

## Configuration

The existing Groq model is retained. Optional backend environment variables:

- `GROQ_REASONING_EFFORT=low` (default; sent only to supported GPT-OSS models)
- `GROQ_MAX_TOKENS=800` (default; reasoning and answer need enough room)
- `RAG_TOP_K=6` (default for focused questions)

Overview requests still retrieve full topic sources. Specific project questions
use bounded retrieval; XAI questions expand to the Brain Tumor project name.
Voice responses normally use 2–4 short sentences; users can request more detail.

## Timing

The final NDJSON `metrics` event and backend `Answer timing` log report retrieval,
model first-token, overall first-token and total milliseconds, plus context chunk
count and prompt characters. Browser debug logs report time from request dispatch
to first text and actual first audio playback. These do not measure the delay
between finishing speech and the browser finalizing the transcript.

Measure warm requests separately from backend/model startup. No production latency
or microphone-device benchmark is implied by passing offline tests.

## Verification and release

Backend: `venv/Scripts/python.exe -m unittest test_pdf_knowledge`

Portfolio frontend (Node 22.18+):

```
node --test scripts/voice-stream.test.mjs
node node_modules/typescript/bin/tsc --noEmit --incremental false
```

Deploy both repositories: this backend and the sibling `profile` frontend.
Retain the updated PDFs in `data` so startup can synchronize their index.

In Chrome or Edge, allow microphone/audio access and verify a long answer starts
speaking before the last text arrives. Test Stop response, mic interruption,
switching chats mid-answer, and consecutive voice turns. Browser/OS speech voices
and recognition endpointing still affect audio quality and latency; this uses the
existing browser speech pipeline, not a native audio model.
