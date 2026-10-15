"""Offline regressions: python -m unittest tests.test_pdf_knowledge."""
import unittest
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.documents import Document

import ingest
import retrieve
import main


class PdfKnowledgeTests(unittest.TestCase):
    def test_specific_xai_question_keeps_retrieval_bounded(self):
        store = Mock()
        store.similarity_search_with_score.return_value = [(Document(page_content="MRI Grad-CAM"), 0.8)]
        retrieve.retrieve_knowledge(store, "explain the bharat xai project")
        store.get.assert_not_called()
        self.assertEqual(store.similarity_search_with_score.call_args.kwargs["k"], retrieve.TOP_K)
        self.assertIn("Brain Tumor", store.similarity_search_with_score.call_args.args[0])

    def test_reasoning_and_stream_cleanup(self):
        client = Mock()
        stream = Mock()
        stream.__iter__ = Mock(return_value=iter([
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Hello"))])
        ]))
        client.chat.completions.create.return_value = stream
        with patch.object(retrieve, "get_groq_client", return_value=client), \
             patch.object(retrieve, "GROQ_MODEL", "openai/gpt-oss-120b"):
            response = retrieve.stream_groq("question")
            self.assertEqual(next(response), "Hello")
            response.close()
        stream.close.assert_called_once()
        options = client.chat.completions.create.call_args.kwargs
        self.assertEqual(options["reasoning_effort"], "low")
        self.assertEqual(options["messages"][0]["role"], "system")

    def test_error_is_not_an_answer_token(self):
        def failing_stream(*args, **kwargs):
            yield "Useful partial answer."
            raise RuntimeError("offline failure")
        with patch.object(retrieve, "stream_answer", side_effect=failing_stream):
            events = [json.loads(line) for line in main.generate_stream("question", "test", True)]
        self.assertEqual([e["type"] for e in events], ["language", "token", "error", "metrics", "done"])
        self.assertIn("first_token_ms", events[3])

    def test_voice_instruction_and_bounded_history(self):
        prompt = retrieve.build_final_prompt("Explain XAI", "MRI", "x" * 12000, voice=True)
        self.assertIn("2-4 short", prompt)
        self.assertLess(len(prompt), 5000)

    def test_resume_question_searches_resume_and_keeps_continuation(self):
        store = Mock()
        store.get.return_value = {"ids": ["first", "continuation"]}
        docs = [
            (Document(page_content="SKILLS Python"), 0.8),
            (Document(page_content="SQL, Excel"), 1.9),
        ]
        store.similarity_search_with_score.return_value = docs
        self.assertEqual(retrieve.retrieve_knowledge(store, "Tell me about Bharat's resume"), docs)
        self.assertEqual(store.similarity_search_with_score.call_args.kwargs,
                         {"k": 2, "filter": {"source": {"$in": ["Resume.pdf"]}}})

    def test_no_relevant_topic_does_not_bypass_threshold(self):
        store = Mock()
        store.get.return_value = {"ids": ["x"]}
        store.similarity_search_with_score.return_value = [(Document(page_content="other"), 3.0)]
        self.assertEqual(retrieve.retrieve_knowledge(store, "Bharat's projects"), [])

    def test_sync_replaces_old_content_and_is_idempotent(self):
        store = Mock()
        store.get.return_value = {"ids": ["old"]}
        chunks = [Document(page_content="updated resume", metadata={"source": "Resume.pdf"})]
        ingest.synchronize_vector_database(store, chunks)
        new_ids = store.add_documents.call_args.kwargs["ids"]
        store.delete.assert_called_once_with(ids=["old"])
        store.reset_mock()
        store.get.return_value = {"ids": new_ids}
        ingest.synchronize_vector_database(store, chunks)
        store.add_documents.assert_not_called()
        store.delete.assert_not_called()

    def test_failed_embedding_keeps_old_chunks(self):
        store = Mock()
        store.get.return_value = {"ids": ["old"]}
        store.add_documents.side_effect = RuntimeError("embedding failed")
        with self.assertRaises(RuntimeError):
            ingest.synchronize_vector_database(store, [
                Document(page_content="new", metadata={"source": "Resume.pdf"})
            ])
        store.delete.assert_not_called()

    def test_stream_sends_pdf_context_to_generator(self):
        store = Mock()
        with patch.object(retrieve, "get_vector_store", return_value=store), \
             patch.object(retrieve, "retrieve_knowledge", return_value=[
                 (Document(page_content="YAOP and FrameForge", metadata={"source": "Resume.pdf"}), 0.8)
             ]), \
             patch.object(retrieve, "build_memory_context", return_value=""), \
             patch.object(retrieve.memory, "add_exchange"), \
             patch.object(retrieve, "stream_generate_answer", return_value=iter(["Answer"])) as generate:
            self.assertEqual("".join(retrieve.stream_answer("List Bharat's projects")), "Answer")
            self.assertIn("YAOP and FrameForge", generate.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
