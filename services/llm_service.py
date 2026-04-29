"""LLM service that wraps Google Gemini 1.5 Flash for generation and verification."""

import google.generativeai as genai

# Verifier instruction embedded in the verification prompt.
_VERIFIER_INSTRUCTION = (
    "You are a strict fact-checker. Your only job is to determine whether the"
    " answer below is grounded exclusively in the Context section of the"
    " supplied prompt. Do not use any prior knowledge.\n\n"
    "Reply with exactly one of:\n"
    "  GROUNDED   — every claim in the answer can be traced to the Context.\n"
    "  HALLUCINATION — the answer contains information not present in the Context.\n\n"
    "After the verdict, add a single sentence explaining your reasoning."
)


class LLMService:
    """Provides text generation and grounding verification via Gemini 1.5 Flash.

    Both *generate* and *verify* use the same underlying model so that the
    verifier applies the same world-knowledge cut-off as the generator.
    """

    MODEL_ID = "gemini-1.5-flash"

    def __init__(self, api_key: str) -> None:
        """Configure the Gemini SDK with *api_key*.

        Args:
            api_key: A valid Google AI Studio API key.
        """
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(self.MODEL_ID)

    def generate(self, augmented_prompt: str) -> str:
        """Generate an answer from *augmented_prompt*.

        The prompt is expected to already contain a "Context:" section and a
        "Query:" section as formatted by the RAG server.

        Args:
            augmented_prompt: The full prompt string produced by the RAG server.

        Returns:
            The model's text response.
        """
        try:
            response = self._model.generate_content(augmented_prompt)
            return response.text
        except Exception as exc:
            raise RuntimeError(
                f"Gemini 1.5 Flash generation failed: {exc}"
            ) from exc

    def verify(self, augmented_prompt: str, answer: str) -> str:
        """Check whether *answer* is grounded in the context of *augmented_prompt*.

        The verifier is instructed to reply with either ``GROUNDED`` or
        ``HALLUCINATION``, followed by a one-sentence explanation.

        Args:
            augmented_prompt: The original prompt (includes the Context section).
            answer: The answer produced by the generator step.

        Returns:
            The verifier's raw text response (starts with ``GROUNDED`` or
            ``HALLUCINATION``).
        """
        verification_prompt = (
            f"{_VERIFIER_INSTRUCTION}\n\n"
            f"--- Original Prompt ---\n{augmented_prompt}\n\n"
            f"--- Answer to Verify ---\n{answer}"
        )
        try:
            response = self._model.generate_content(verification_prompt)
            return response.text
        except Exception as exc:
            raise RuntimeError(
                f"Gemini 1.5 Flash verification failed: {exc}"
            ) from exc
