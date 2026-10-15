import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

import retrieve
from conversation import resolve_language, clock_context, is_analysis


class ConversationTests(unittest.TestCase):
    def test_language_commands_and_followups(self):
        for question, previous, expected in [
            ("Can you talk in Hindi?", "en", "hi"),
            ("Can you speak Nepali?", "hi", "ne"),
            ("Switch back to English", "ne", "en"),
            ("भारत pressure कैसे handle करेगा?", "en", "hi"),
            ("भारतको कमजोरी के छ?", "en", "ne"),
            ("अच्छा बताओ", "en", "hi"),
            ("tell me more", "hi", "hi"),
            ("What is Hindi?", "en", "en"),
            ("Don't speak Hindi", "en", "en"),
        ]:
            self.assertEqual(resolve_language(question, previous), expected, question)

    def test_translation_preserves_original_question_for_answer(self):
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(choices=[
            SimpleNamespace(message=SimpleNamespace(content='{"query":"How would Bharat handle a deadline?"}'))
        ])
        with patch.object(retrieve, "get_groq_client", return_value=client):
            query = retrieve.english_search_query("भारत deadline कैसे handle करेगा?", "hi")
        self.assertTrue(is_analysis(query))
        prompt = retrieve.build_final_prompt("भारत deadline कैसे handle करेगा?", "Evidence", "", language="hi")
        self.assertIn("Response language: Hindi", prompt)
        self.assertIn("भारत deadline कैसे handle करेगा?", prompt)

    def test_general_question_reaches_model_without_pdf_match(self):
        with patch.object(retrieve, "get_vector_store", return_value=Mock()), \
             patch.object(retrieve, "retrieve_knowledge", return_value=[]), \
             patch.object(retrieve, "stream_generate_answer", return_value=iter(["React is a UI library."])) as model, \
             patch.object(retrieve.memory, "add_exchange"):
            result = "".join(retrieve.stream_answer("What is React?"))
        self.assertIn("UI library", result)
        self.assertIn("No relevant documented personal facts", model.call_args.args[0])

    def test_situational_question_adds_actual_personal_evidence(self):
        store = Mock()
        store.get.return_value = {
            "documents": ["Documented improvement area: public speaking."],
            "metadatas": [{"source": "Bharat_Sirmal_01_Personal.pdf"}],
        }
        with patch.object(retrieve, "get_vector_store", return_value=store), \
             patch.object(retrieve, "retrieve_knowledge", return_value=[]), \
             patch.object(retrieve, "stream_generate_answer", return_value=iter(["He could practise presentations."])) as model, \
             patch.object(retrieve.memory, "add_exchange"):
            result = "".join(retrieve.stream_answer("How would Bharat handle an interview?"))
        self.assertIn("could", result)
        self.assertIn("public speaking", model.call_args.args[0])
        self.assertIn("not a verified personal fact", retrieve.SYSTEM_PROMPT)

    def test_clock_has_both_explicit_timezones(self):
        context = clock_context()
        self.assertIn("Asia/Kolkata", context)
        self.assertIn("Asia/Kathmandu", context)


if __name__ == "__main__":
    unittest.main()
