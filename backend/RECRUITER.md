# Recruiter conversations

Open **Recruiter · Add job description** above the chat composer. Upload a searchable
PDF, DOCX or UTF-8 TXT, paste requirements, or dictate them and review the transcript.
Attach them, then ask questions through the normal text or voice controls. English,
Hindi and Nepali use the existing language detection and device speech support.
The language dropdown remains hidden.

Questions are open-ended. There is no fixed list of supported hiring questions:
the model receives the current employer requirements, Bharat's documented evidence,
the latest question and recent conversation. It can assess fit, explain the basis,
identify gaps, discuss hypothetical scenarios or suggest interview checks. Instructions
require it to distinguish evidence from inference and unknowns, and never manufacture
skills, employment history, scores or guarantees. An assessment remains model-generated;
verify important hiring claims through an interview and references.

## Limits and lifetime

- Files: 5 MB; PDFs: up to 30 pages; extracted/pasted text: 24,000 characters.
- DOCX includes body paragraphs and table cells. Older binary DOC files must be
  converted to DOCX. Image-only content, text embedded in images and complex visual
  layouts are not OCR-processed. PDFs with textless pages are rejected to avoid
  silently dropping requirements.
- One job description per chat. Replace it to start a new assessment; remove it to
  return to ordinary chat. Replacement/removal clears backend conversation memory
  so previous role assumptions do not contaminate the new assessment. Earlier
  messages remain visible in browser chat history.
- Job descriptions stay in process memory for one hour from attachment, with a
  maximum of 256 simultaneous recruiter contexts. Original uploads are not saved,
  served publicly or inserted into the permanent Bharat Chroma collection.
- Extracted employer text and candidate evidence are sent to the configured Groq
  model when a question is asked. Upload parsing itself does not call a model.
- A server restart clears job context. The existing browser chat history still
  stores messages. A revision check returns HTTP 409 for expired/replaced context
  instead of silently generating an assessment without the expected requirements.
- Use **one backend worker/replica**, matching the current process-local conversation
  memory. Before horizontal scaling, move recruiter contexts AND conversation memory
  to a shared expiring store. No new database credentials are required for this version.
- Session identifiers act as unguessable conversation capabilities, not user logins.
  The frontend creates new identifiers with `crypto.randomUUID()`; never share them.

## API

All recruiter routes require the same `session_id` used by `/ask`.

- `POST /recruiter/context`: JSON `{ "session_id": "...", "text": "..." }`.
- `POST /recruiter/upload?session_id=...&filename=job.pdf`: raw file bytes. Raw streaming
  enforces the upload cap without requiring multipart dependencies.
- `GET /recruiter/context?session_id=...`: name, revision and expiry; no document text.
- `DELETE /recruiter/context?session_id=...`: remove employer context and clear memory.
- `POST /ask`: existing fields plus optional `recruiter_revision` from the attachment
  response. Updated clients send it to guard against lost or replaced job descriptions.
  The NDJSON token/language/metrics/done stream remains compatible.

The personal corpus is currently small (33 chunks). Recruiter mode supplies its full
evidence rather than relying on a question-keyword match. This supports unfamiliar
role criteria and multilingual follow-ups without a translation model call. Regular
chat retains its focused retrieval. For a substantially larger candidate corpus,
replace this with requirement-aware retrieval and a cached evidence summary.

`RECRUITER_MAX_TOKENS` defaults to 2200 for detailed text assessments. Existing
`GROQ_MAX_TOKENS` still controls normal chat. Voice answers keep the existing short
spoken style and begin speaking as complete sentences arrive.

## Deploy and verify

Deploy the backend and frontend together. No additional Python dependencies or
API keys are needed. Keep the current backend on a single worker/replica.

Run `venv/Scripts/python.exe -m unittest test_recruiter test_conversation test_pdf_knowledge`
in the backend and `node --test scripts/voice-stream.test.mjs`, TypeScript checking
and `npm run build` in the frontend. Recruiter integration tests use synthetic job
documents and mocked generation; they do not send private PDFs to an external model.

Smoke-check: attach a job description, ask a detailed hiring question, ask an unrelated
scenario follow-up, switch language, replace the job, switch chats, remove it and try
an unreadable file. Expired/replaced attachments should request reattachment.
