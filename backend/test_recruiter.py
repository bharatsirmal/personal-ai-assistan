import json
import unittest
from io import BytesIO
from zipfile import ZipFile
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
import main
import recruiter
import retrieve


SESSION = "session_synthetic_test:chat_123"
JD = "Required: Python and PostgreSQL. Preferred: Kubernetes. Own production support."


def docx_bytes():
    output = BytesIO()
    with ZipFile(output, "w") as archive:
        archive.writestr("word/document.xml", '''<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>Required: Python.</w:t></w:r></w:p><w:tbl><w:tr><w:tc><w:p><w:r><w:t>Preferred: PostgreSQL.</w:t></w:r></w:p></w:tc></w:tr></w:tbl></w:body></w:document>''')
    return output.getvalue()


def pdf_bytes():
    stream = b"BT /F1 12 Tf 40 100 Td (Required Python and PostgreSQL skills.) Tj ET"
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
               b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 400 200] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
               b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
               b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"]
    result = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(result))
        result.extend(f"{index} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = len(result)
    result.extend(b"xref\n0 6\n0000000000 65535 f \n")
    for offset in offsets[1:]:
        result.extend(f"{offset:010d} 00000 n \n".encode())
    result.extend(f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode())
    return bytes(result)


class RecruiterTests(unittest.TestCase):
    def setUp(self):
        recruiter._contexts.clear()
        self.client = TestClient(main.app)  # No startup/model/network needed.

    def test_real_pdf_docx_and_text_extraction(self):
        self.assertIn("PostgreSQL", recruiter.extract_document(pdf_bytes(), "job.pdf"))
        self.assertEqual(recruiter.extract_document(docx_bytes(), "job.DOCX"), "Required: Python.\nPreferred: PostgreSQL.")
        self.assertEqual(recruiter.extract_document(JD.encode(), "job.txt"), JD)

    def test_bad_uploads_and_size_limit(self):
        for data, name in [(b"wrong", "job.docx"), (b"wrong", "job.pdf"), (b"wrong", "job.doc"), (b"", "job.txt"), (b"x" * (recruiter.MAX_BYTES + 1), "job.txt")]:
            with self.assertRaises(ValueError):
                recruiter.extract_document(data, name)
        with self.assertRaises(ValueError):
            recruiter.save_context(SESSION, "x" * (recruiter.MAX_CHARS + 1))

    def test_session_isolation_replace_delete_and_private_metadata(self):
        response = self.client.post("/recruiter/context", json={"session_id": SESSION, "text": JD})
        self.assertEqual(response.status_code, 200)
        old = response.json()["context"]["revision"]
        self.assertNotIn("text", response.json()["context"])
        self.assertIsNone(self.client.get("/recruiter/context", params={"session_id": SESSION + "other"}).json()["context"])
        response = self.client.post("/recruiter/upload", params={"session_id": SESSION, "filename": "job.docx"}, content=docx_bytes())
        self.assertEqual(response.status_code, 200)
        self.assertNotEqual(old, response.json()["context"]["revision"])
        self.assertEqual(self.client.delete("/recruiter/context", params={"session_id": SESSION}).status_code, 200)
        self.assertIsNone(recruiter.get_context(SESSION))

    def test_expiry_and_stale_revision_rejected_before_model(self):
        value = recruiter.save_context(SESSION, JD)
        with patch.object(recruiter, "time", return_value=value["expires_at"] + 1):
            response = self.client.post("/ask", json={"session_id": SESSION, "question": "How would he handle an outage?", "recruiter_revision": value["revision"]})
        self.assertEqual(response.status_code, 409)
        self.assertIsNone(recruiter.get_context(SESSION))

    def test_validation_does_not_replace_good_context(self):
        value = recruiter.save_context(SESSION, JD)
        response = self.client.post("/recruiter/upload", params={"session_id": SESSION, "filename": "bad.docx"}, content=b"invalid")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(recruiter.get_context(SESSION)["revision"], value["revision"])
        self.assertEqual(self.client.post("/recruiter/context", json={"session_id": "default", "text": JD}).status_code, 400)

    def test_upload_endpoint_enforces_byte_limit_and_replaced_revision(self):
        value = recruiter.save_context(SESSION, JD)
        with patch.object(recruiter, "MAX_BYTES", 100):
            response = self.client.post("/recruiter/upload", params={"session_id": SESSION, "filename": "job.txt"}, content=b"x" * 101)
        self.assertEqual(response.status_code, 413)
        recruiter.save_context(SESSION, "New role: Required Java and Spring experience.")
        response = self.client.post("/ask", json={"session_id": SESSION, "question": "Can we hire him?", "recruiter_revision": value["revision"]})
        self.assertEqual(response.status_code, 409)

    def test_old_stream_cannot_repopulate_memory_after_job_replacement(self):
        value = recruiter.save_context(SESSION, JD)
        store = Mock()
        store.get.return_value = {"documents": [], "metadatas": []}
        def changing_answer(*args, **kwargs):
            yield "Old assessment."
            recruiter.save_context(SESSION, "New role: Required Java and Spring experience.")
        with patch.object(retrieve, "get_vector_store", return_value=store), \
             patch.object(retrieve, "stream_generate_answer", side_effect=changing_answer), \
             patch.object(retrieve.memory, "add_exchange") as save:
            list(retrieve.stream_answer("What would he do?", SESSION, job_context=value))
            save.assert_not_called()

    def test_arbitrary_multilingual_questions_have_both_sources_and_history(self):
        store = Mock()
        store.get.return_value = {"documents": ["Bharat built a Python project. No production support history is documented."], "metadatas": [{"source": "Synthetic Resume.pdf"}]}
        job = recruiter.save_context(SESSION, JD)
        for question, language in [("How should he approach our first production outage?", "en"), ("उसके पहले 30 दिनों का plan क्या होगा?", "hi"), ("उहाँलाई कुन जिम्मेवारी दिन सकिन्छ?", "ne"), ("What if we remove Kubernetes as a requirement?", "en")]:
            with self.subTest(question=question), patch.object(retrieve, "get_vector_store", return_value=store), \
                 patch.object(retrieve, "stream_generate_answer", return_value=iter(["Conditional assessment."])) as model, \
                 patch.object(retrieve, "build_memory_context", return_value="Earlier recruiter: focus on production support"), \
                 patch.object(retrieve, "english_search_query") as translation:
                answer = "".join(retrieve.stream_answer(question, SESSION, language=language, job_context=job))
                prompt = model.call_args.args[0]
                self.assertIn(JD, prompt)
                self.assertIn("Synthetic Resume.pdf", prompt)
                self.assertIn("production support", prompt)
                self.assertIn(question, prompt)
                self.assertTrue(model.call_args.kwargs["recruiter_mode"])
                self.assertTrue(answer)
                translation.assert_not_called()

    def test_ask_stream_uses_snapshot_and_normal_chat_has_no_job(self):
        value = recruiter.save_context(SESSION, JD)
        with patch.object(retrieve, "stream_answer", return_value=iter(["Evidence-based answer."])) as stream:
            response = self.client.post("/ask", json={"session_id": SESSION, "question": "What is the hiring basis?", "recruiter_revision": value["revision"]})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(stream.call_args.kwargs["job_context"]["text"], JD)
            events = [json.loads(line) for line in response.text.splitlines()]
            self.assertEqual(events[-1]["type"], "done")
        with patch.object(retrieve, "stream_answer", return_value=iter(["Normal answer."])) as stream:
            self.client.post("/ask", json={"session_id": SESSION + "other", "question": "What is Python?"})
            self.assertIsNone(stream.call_args.kwargs["job_context"])


if __name__ == "__main__":
    unittest.main()
