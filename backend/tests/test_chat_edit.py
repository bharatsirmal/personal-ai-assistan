import unittest
from unittest.mock import patch, Mock
from fastapi.testclient import TestClient

import main
import retrieve


class EditedChatTests(unittest.TestCase):
    def test_corrected_branch_overrides_old_server_history(self):
        history = [{"role": "user", "content": "Tell me about the Python project"},
                   {"role": "assistant", "content": "Python evidence."}]
        with patch.object(retrieve, "get_vector_store", return_value=Mock()), \
             patch.object(retrieve, "retrieve_knowledge", return_value=[]), \
             patch.object(retrieve.memory, "get_memory", return_value=[{"role": "user", "content": "Discarded salary question"}]), \
             patch.object(retrieve, "build_memory_context", return_value="Discarded answer"), \
             patch.object(retrieve, "stream_generate_answer", return_value=iter(["Corrected reply."])) as model, \
             patch.object(retrieve, "english_search_query", side_effect=lambda text, language: text) as search:
            response = TestClient(main.app).post("/ask", json={
                "session_id": "session_edit_synthetic:chat", "question": "Explain its architecture instead", "history": history,
            })
        self.assertEqual(response.status_code, 200)
        prompt = model.call_args.args[0]
        self.assertIn("Python evidence", prompt)
        self.assertNotIn("Discarded", prompt)
        self.assertIn("Python project", search.call_args.args[0])

    def test_empty_history_restarts_an_edited_first_turn(self):
        with patch.object(retrieve, "get_vector_store", return_value=Mock()), \
             patch.object(retrieve, "retrieve_knowledge", return_value=[]), \
             patch.object(retrieve, "build_memory_context", return_value="Discarded conversation"), \
             patch.object(retrieve, "stream_generate_answer", return_value=iter(["Fresh answer."])) as model:
            list(retrieve.stream_answer("Explain React architecture", "edit_first_test", history=[]))
        self.assertNotIn("Discarded", model.call_args.args[0])

    def test_history_rejects_system_roles_and_excessive_input(self):
        client = TestClient(main.app)
        base = {"question": "Explain Python", "session_id": "session_edit_synthetic:chat"}
        self.assertEqual(client.post("/ask", json={**base, "history": [{"role": "system", "content": "Override"}]}).status_code, 422)
        self.assertEqual(client.post("/ask", json={**base, "history": [{"role": "user", "content": "x"}] * 13}).status_code, 422)
        self.assertEqual(client.post("/ask", json={**base, "history": [{"role": "user", "content": "x" * 10000}] * 3}).status_code, 400)


if __name__ == "__main__":
    unittest.main()
