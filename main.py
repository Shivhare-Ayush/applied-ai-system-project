"""Agentic AI Orchestrator — entry point.

Workflow
--------
1. Accept a user query from the command line (or a hard-coded default for
   quick testing).
2. Call the RAG server to retrieve grounding context and obtain an
   ``augmented_prompt``.
3. Pass the ``augmented_prompt`` to Gemini 1.5 Flash to generate an answer.
4. Run the Verifier step: ask the LLM to check whether the answer is grounded
   exclusively in the retrieved context.
5. If the Verifier flags a hallucination, perform one Self-Correction attempt
   by re-generating with an explicit instruction to restrict the answer to the
   context.
6. Log vector similarity scores and the final validation verdict.
"""

import logging
import os
import sys

from dotenv import load_dotenv

from services import LLMService, RAGClient

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

RAG_BASE_URL = os.getenv("RAG_BASE_URL", "http://34.122.127.176:8080")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

_SELF_CORRECTION_PREFIX = (
    "Your previous answer contained information not found in the provided"
    " Context. Rewrite the answer using ONLY the information in the Context"
    " section below. Do not add any knowledge from outside the Context.\n\n"
)


# ---------------------------------------------------------------------------
# Orchestration helpers
# ---------------------------------------------------------------------------


def _is_hallucination(verdict: str) -> bool:
    """Return True when the verifier verdict indicates a hallucination."""
    return verdict.strip().upper().startswith("HALLUCINATION")


def _log_rag_metadata(sources: list, scores: list) -> None:
    """Emit one log line per retrieved source with its vector similarity score."""
    logger.info("RAG retrieval — %d source(s) returned:", len(sources))
    for idx, (source, score) in enumerate(zip(sources, scores), start=1):
        logger.info("  [%d] score=%.4f | %s", idx, score, source[:120])


# ---------------------------------------------------------------------------
# Main orchestration pipeline
# ---------------------------------------------------------------------------


def run(user_query: str) -> str:
    """Execute the full orchestration pipeline for *user_query*.

    Steps:
      1. RAG retrieval
      2. LLM generation
      3. Verifier (grounding check)
      4. Optional self-correction (one attempt)

    Returns:
        The final answer string.
    """
    if not GEMINI_API_KEY:
        logger.error(
            "GEMINI_API_KEY is not set. Add it to your .env file or environment."
        )
        sys.exit(1)

    rag = RAGClient(base_url=RAG_BASE_URL)
    llm = LLMService(api_key=GEMINI_API_KEY)

    # ------------------------------------------------------------------
    # Step 1 — RAG retrieval
    # ------------------------------------------------------------------
    logger.info("Querying RAG server: %s", RAG_BASE_URL)
    try:
        rag_response = rag.query(user_query)
    except Exception as exc:
        logger.error("RAG retrieval failed: %s", exc)
        sys.exit(1)

    augmented_prompt: str = rag_response["augmented_prompt"]
    sources: list = rag_response.get("sources", [])
    scores: list = rag_response.get("scores", [])

    _log_rag_metadata(sources, scores)
    logger.info("Augmented prompt received (%d chars).", len(augmented_prompt))

    # ------------------------------------------------------------------
    # Step 2 — LLM generation
    # ------------------------------------------------------------------
    logger.info("Generating answer with Gemini 1.5 Flash …")
    try:
        answer = llm.generate(augmented_prompt)
    except RuntimeError as exc:
        logger.error("LLM generation failed: %s", exc)
        sys.exit(1)
    logger.info("Initial answer generated (%d chars).", len(answer))

    # ------------------------------------------------------------------
    # Step 3 — Verifier (grounding check)
    # ------------------------------------------------------------------
    logger.info("Running Verifier step …")
    try:
        verdict = llm.verify(augmented_prompt, answer)
    except RuntimeError as exc:
        logger.warning("Verifier step failed (%s). Skipping correction.", exc)
        return answer
    logger.info("Verifier verdict: %s", verdict.splitlines()[0])

    # ------------------------------------------------------------------
    # Step 4 — Self-correction (one attempt on hallucination)
    # ------------------------------------------------------------------
    if _is_hallucination(verdict):
        logger.warning(
            "Hallucination detected. Performing one self-correction attempt."
        )
        correction_prompt = _SELF_CORRECTION_PREFIX + augmented_prompt
        try:
            answer = llm.generate(correction_prompt)
        except RuntimeError as exc:
            logger.error("Self-correction generation failed: %s", exc)
            return answer
        logger.info("Self-corrected answer generated (%d chars).", len(answer))

        # Re-verify after self-correction for logging purposes.
        try:
            second_verdict = llm.verify(augmented_prompt, answer)
        except RuntimeError as exc:
            logger.warning(
                "Post-correction verification failed (%s). "
                "Returning self-corrected answer without re-verification.",
                exc,
            )
            return answer
        logger.info(
            "Post-correction Verifier verdict: %s", second_verdict.splitlines()[0]
        )
        if _is_hallucination(second_verdict):
            logger.warning(
                "Self-corrected answer still flagged as hallucination. "
                "Returning best available answer."
            )
        else:
            logger.info("Self-corrected answer passed grounding verification.")
    else:
        logger.info("Answer passed grounding verification. No correction needed.")

    return answer


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and run the orchestration pipeline."""
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
    else:
        user_query = "What are the main features of the system?"
        logger.info("No query provided — using default: %r", user_query)

    final_answer = run(user_query)

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(final_answer)


if __name__ == "__main__":
    main()
