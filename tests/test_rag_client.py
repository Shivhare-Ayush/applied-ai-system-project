import unittest
from unittest.mock import Mock, patch

import requests

from services.rag_client import RAGClient


class TestRAGClient(unittest.TestCase):
    def setUp(self) -> None:
        self.client = RAGClient(base_url="http://example.com")

    @patch("services.rag_client.requests.post")
    def test_query_uses_query_text_payload(self, mock_post: Mock) -> None:
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "augmented_prompt": "Context:\nFact A\n\nQuery: test",
            "sources": ["Fact A"],
            "scores": [0.91],
        }
        mock_post.return_value = mock_response

        result = self.client.query("test")

        self.assertIn("augmented_prompt", result)
        mock_post.assert_called_once_with(
            "http://example.com/query",
            json={"query_text": "test"},
            timeout=30,
        )

    def test_query_rejects_empty_input(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self.client.query("   ")
        self.assertIn("cannot be empty", str(ctx.exception).lower())

    def test_query_rejects_overly_long_input(self) -> None:
        long_query = "x" * 5001
        with self.assertRaises(ValueError) as ctx:
            self.client.query(long_query)
        self.assertIn("maximum length", str(ctx.exception).lower())

    @patch("services.rag_client.requests.post")
    def test_query_wraps_timeout_error(self, mock_post: Mock) -> None:
        mock_post.side_effect = requests.Timeout("timeout")

        with self.assertRaises(requests.Timeout) as ctx:
            self.client.query("hello")

        self.assertIn("did not respond", str(ctx.exception))

    @patch("services.rag_client.requests.post")
    def test_query_wraps_connection_error(self, mock_post: Mock) -> None:
        mock_post.side_effect = requests.ConnectionError("unreachable")

        with self.assertRaises(requests.ConnectionError) as ctx:
            self.client.query("hello")

        self.assertIn("Could not reach RAG server", str(ctx.exception))

    @patch("services.rag_client.requests.post")
    def test_query_includes_server_response_in_http_error(self, mock_post: Mock) -> None:
        mock_response = Mock()
        mock_response.status_code = 422
        mock_response.text = '{"detail":"bad payload"}'
        mock_response.raise_for_status.side_effect = requests.HTTPError("422")
        mock_post.return_value = mock_response

        with self.assertRaises(requests.HTTPError) as ctx:
            self.client.query("hello")

        self.assertIn("HTTP 422", str(ctx.exception))
        self.assertIn("Server response", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
