"""RAG client for communicating with the retrieval-augmented generation server."""

import requests


class RAGClient:
    """Handles HTTP communication with the RAG server.

    The server exposes a single POST /query endpoint that accepts a plain-text
    query and returns an augmented prompt together with the retrieved sources
    and their vector similarity scores.
    """

    def __init__(self, base_url: str, timeout: int = 30) -> None:
        """Initialise the client.

        Args:
            base_url: Root URL of the RAG server, e.g. ``http://host:8080``.
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def query(self, user_query: str) -> dict:
        """Send *user_query* to the RAG server and return the raw response.

        Returns:
            A dict with keys:
              - ``augmented_prompt`` (str): The prompt pre-formatted with a
                "Context:" section followed by the original "Query:" section.
              - ``sources`` (list[str]): The retrieved document chunks.
              - ``scores`` (list[float]): Vector similarity scores that
                correspond positionally to *sources*.

        Raises:
            requests.HTTPError: When the server responds with a non-2xx status.
            requests.ConnectionError: When the server is unreachable.
            requests.Timeout: When the request exceeds *timeout* seconds.
        """
        url = f"{self.base_url}/query"
        payload = {"query": user_query}
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
        except requests.Timeout as exc:
            raise requests.Timeout(
                f"RAG server at {self.base_url} did not respond within"
                f" {self.timeout}s for query: {user_query!r}"
            ) from exc
        except requests.ConnectionError as exc:
            raise requests.ConnectionError(
                f"Could not reach RAG server at {self.base_url}."
                f" Check the server URL and network connectivity."
            ) from exc
        except requests.HTTPError as exc:
            raise requests.HTTPError(
                f"RAG server returned HTTP {response.status_code} for"
                f" query: {user_query!r}"
            ) from exc
        return response.json()
