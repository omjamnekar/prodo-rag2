import json
import unittest
from main import app

class TestMainRoutes(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True # type: ignore

    def test_home(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data.decode('utf-8'), "Hello, PRODO RAG on Vercel!")

    def test_health(self):
        response = self.app.get('/rag/health')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.data.decode('utf-8')), {"status": "ok"})

    def test_rag_query(self):
        """Test the RAG query endpoint."""
        payload = {
            "repoId": "test-repo",
            "prompt": "Test prompt",
            "top_k": 5,
            "metadata": {}
        }
        response = self.app.post('/rag/query', json=payload)
        self.assertEqual(response.status_code, 200)

    def test_index_repo_call(self):
        payload = {
            "repoId": "test-repo",
            "files": [
                {"filename": "test.py", "content": "print('Hello World')"}
            ],
            "metadata": {}
        }
        response = self.app.post('/rag/index', json=payload)
        self.assertEqual(response.status_code, 200)

    def test_rag_reset(self):
        response = self.app.delete('/rag/reset?repoId=test-repo')
        self.assertEqual(response.status_code, 200)

    def test_delete_repo_vectors(self):
        response = self.app.delete('/rag/delete?repoId=test-repo')
        self.assertEqual(response.status_code, 200)


if __name__ == '__main__':
    unittest.main()