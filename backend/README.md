# Bharat AI backend

FastAPI service for Bharat's portfolio assistant. It retrieves knowledge from the PDFs in `data/`, streams answers, and accepts recruiter requirements and attachments. The frontend is maintained separately in the sibling `profile` repository.

## Run locally

Run these commands from this directory (Windows PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
# Configure GROQ_API_KEY in .env before starting.
.\venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

API documentation: `http://127.0.0.1:8000/docs`. Health endpoint: `/health`.
Keep `.env` private. Knowledge PDFs and `chroma_db/` are application data; do not treat them as temporary build output.

## Structure

```text
main.py             HTTP endpoints and streaming
retrieve.py         Retrieval, embedding setup and model calls
conversation.py     Language and conversational routing
prompt.py           Assistant instructions
recruiter.py        Recruiter context and attachment parsing
memory.py           In-memory conversation state
ingest.py           PDF ingestion
data/               Knowledge PDFs
chroma_db/          Persisted knowledge index
audio/              Generated audio compatibility endpoint
docs/               Recruiter and streaming documentation
examples/           Sample JSON requests
tests/              Offline regression tests
  fixtures/         Curated question/answer evaluation data
scripts/            Explicit live smoke checks
```

The Python entry points stay in this directory so existing `uvicorn main:app` deployment commands continue working. Speech playback uses the browser; the unused Piper implementation and old generated audio samples have been removed.

## Verification

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests
```

The offline suite mocks model calls. The separate `python -m scripts.smoke_rag` command is a **live** check that sends knowledge context to the configured AI provider; run it deliberately after configuring credentials.

See [recruiter support](docs/RECRUITER.md) and [streaming](docs/STREAMING.md).
