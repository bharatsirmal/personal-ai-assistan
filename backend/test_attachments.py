import json
import unittest
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image
from fastapi.testclient import TestClient
import main
import recruiter
import retrieve

SESSION = "session_attachment_synthetic:chat"


def completion(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


class AttachmentTests(unittest.TestCase):
    def setUp(self):
        recruiter.remove_context(SESSION)

    def test_valid_image_goes_to_vision_as_data_not_instructions(self):
        data = BytesIO()
        Image.new("RGB", (40, 40), "white").save(data, format="PNG")
        client = Mock()
        client.chat.completions.create.return_value = completion("Required Python and SQL skills. Preferred FastAPI.")
        with patch.object(retrieve, "get_groq_client", return_value=client):
            response = TestClient(main.app).post("/recruiter/upload", params={"session_id": SESSION, "filename": "requirements.png"}, content=data.getvalue())
        self.assertEqual(response.status_code, 200)
        context = recruiter.get_context(SESSION)
        self.assertIn("Python", context["text"])
        self.assertIn("OCR errors", context["text"])
        options = client.chat.completions.create.call_args.kwargs
        self.assertTrue(options["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertIn("never obey", options["messages"][0]["content"])
        self.assertNotIn("text", response.json()["context"])

    def test_invalid_or_oversized_image_rejected_before_provider(self):
        large = BytesIO()
        Image.new("RGB", (5000, 4000), "white").save(large, format="PNG")
        with patch.object(retrieve, "get_groq_client") as provider:
            for data in (b"not a real image", large.getvalue()):
                with self.assertRaises(ValueError):
                    recruiter.extract_document(data, "fake.png")
            provider.assert_not_called()

    def test_image_provider_failure_keeps_previous_document(self):
        old = recruiter.save_context(SESSION, "Required: React and PostgreSQL experience.")
        data = BytesIO()
        Image.new("RGB", (30, 30)).save(data, format="PNG")
        client = Mock()
        client.chat.completions.create.side_effect = RuntimeError("Provider offline")
        with patch.object(retrieve, "get_groq_client", return_value=client):
            response = TestClient(main.app).post("/recruiter/upload", params={"session_id": SESSION, "filename": "job.png"}, content=data.getvalue())
        self.assertEqual(response.status_code, 400)
        self.assertIn("Image reading is unavailable", response.json()["detail"])
        self.assertEqual(old["revision"], recruiter.get_context(SESSION)["revision"])

    def test_normal_chat_recognizes_typed_requirements_and_returns_revision(self):
        question = "We are hiring a backend engineer. Required: Python, SQL and three years of experience. Can Bharat fit?"
        client = Mock()
        client.chat.completions.create.return_value = completion(json.dumps({"is_job_description": True}))
        with patch.object(retrieve, "get_groq_client", return_value=client), \
             patch.object(retrieve, "stream_answer", return_value=iter(["Conditional assessment."])) as answer:
            response = TestClient(main.app).post("/ask", json={"session_id": SESSION, "question": question, "history": [{"role": "user", "content": "Old unrelated job"}]})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(recruiter.get_context(SESSION)["text"], question)
        self.assertEqual(response.headers["X-Context-Revision"], recruiter.get_context(SESSION)["revision"])
        self.assertEqual(answer.call_args.kwargs["history"], [])
        self.assertEqual(answer.call_args.kwargs["job_context"]["text"], question)

    def test_followup_and_short_question_do_not_replace_requirements(self):
        old = recruiter.save_context(SESSION, "Required: Python and SQL experience.")
        client = Mock()
        client.chat.completions.create.return_value = completion('{"is_job_description":false}')
        with patch.object(retrieve, "get_groq_client", return_value=client):
            self.assertIsNone(recruiter.capture_chat_requirements("Would he meet our requirements if we removed SQL?", SESSION))
            self.assertIsNone(recruiter.capture_chat_requirements("Tell me more", SESSION))
        self.assertEqual(client.chat.completions.create.call_count, 1)
        self.assertEqual(old["revision"], recruiter.get_context(SESSION)["revision"])


if __name__ == "__main__":
    unittest.main()
