# Recruiter conversations

The **+** button opens the file picker directly for PDF, DOCX, UTF-8 TXT, PNG, JPEG
and WebP. A selected attachment appears in the composer; it is uploaded when the
message is sent, and its card stays on that user message. Cards open the original
file or image from browser IndexedDB on the same browser. Clearing browser data
removes those original files; file blobs are not put in localStorage or chat requests.

Paste or dictate job requirements in the normal message box. The mic is on its right.
Candidate requirement messages are classified with the existing Groq text model and
recognized job descriptions are retained for later questions. Ordinary short questions
skip classification. English,
Hindi and Nepali use the existing language detection and device speech support.
The language dropdown remains hidden.

Sent user messages have an **Edit** action. **Save & resend** replaces that turn and
its later replies, sends only the preceding conversation as context and generates
a fresh answer. Editing a message with a file restores that file's context when needed.
Replacing the job description also resets the context boundary
in browser history, so old-role assessments do not return after reloading the chat.

Answers render Markdown tables, lists and code blocks. Tables scroll horizontally
on narrow screens. Common HTML formatting tags are normalized to Markdown outside
code examples; raw HTML execution is disabled.

Questions are open-ended. There is no fixed list of supported hiring questions:
the model receives the current employer requirements, Bharat's documented evidence,
the latest question and recent conversation. It can assess fit, explain the basis,
identify gaps, discuss hypothetical scenarios or suggest interview checks. Instructions
require it to distinguish evidence from inference and unknowns, and never manufacture
skills, employment history, scores or guarantees. An assessment remains model-generated;
verify important hiring claims through an interview and references.

## Limits and lifetime

- Files: 5 MB; PDFs: up to 30 pages; extracted text: 24,000 characters. Normal chat
  messages accept up to 12,000 characters; longer requirements can be uploaded as TXT.
- DOCX includes body paragraphs and table cells. Older binary DOC files must be
  converted to DOCX. Scanned PDFs are still rejected; export pages as images or use
  a searchable PDF. Image uploads use Groq vision to extract text and describe content.
  OCR and descriptions can be imperfect, especially for small text and complex diagrams.
- One active reference/job description per chat. Send a new file or type new requirements
  to replace it; start a new chat for a fresh context. Removing an unsent file only
  removes the local draft. Replacement clears backend conversation memory
  so previous role assumptions do not contaminate the new assessment. Earlier
  messages remain visible in browser chat history.
- Job descriptions stay in process memory for one hour from attachment, with a
  maximum of 256 simultaneous recruiter contexts. Original uploads are not saved on
  the server, served publicly or inserted into the permanent Bharat Chroma collection.
  Originals are stored locally in browser IndexedDB for attachment previews.
- Extracted employer text and candidate evidence are sent to the configured Groq
  model when a question is asked. PDF/DOCX/TXT parsing stays local to the backend.
  Image processing sends the uploaded image to Groq, as explicitly approved by the
  project owner. A compact image-derived description is used for later questions;
  the original image is not retained by this backend. UI file selection alone does
  not send anything to Groq: processing begins when Send is pressed.
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
  `X-Context-Revision` reports context recognized from a normally typed job description.

The personal corpus is currently small (33 chunks). Recruiter mode supplies its full
evidence rather than relying on a question-keyword match. This supports unfamiliar
role criteria and multilingual follow-ups without a translation model call. Regular
chat retains its focused retrieval. For a substantially larger candidate corpus,
replace this with requirement-aware retrieval and a cached evidence summary.

`RECRUITER_MAX_TOKENS` defaults to 2200 for detailed text assessments. Existing
`GROQ_MAX_TOKENS` still controls normal chat. Voice answers keep the existing short
spoken style and begin speaking as complete sentences arrive.

## Deploy and verify

Deploy the backend and frontend together. Pillow is included in requirements for image
validation. The existing Groq key is used for vision. `GROQ_VISION_MODEL` defaults to
`qwen/qwen3.8-27b`, currently documented in [Groq's vision guide](https://console.groq.com/docs/vision).
Your account must have access to that model. Keep one backend worker/replica.

Run `venv/Scripts/python.exe -m unittest test_attachments test_chat_edit test_recruiter test_conversation test_pdf_knowledge`
in the backend and `node --test scripts/attachments.test.mjs scripts/chat-format.test.mjs scripts/voice-stream.test.mjs`, TypeScript checking
and `npm run build` in the frontend. Recruiter integration tests use synthetic job
documents and mocked generation; they do not send private PDFs to an external model.

Smoke-check: attach a job description, ask a detailed hiring question, ask an unrelated
scenario follow-up, switch language, replace the job, switch chats, remove it and try
an unreadable file. Expired/replaced attachments should request reattachment.
