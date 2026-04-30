"""Reliability helpers for retrieval confidence estimation."""

from typing import Iterable


def compute_confidence(sources: list, scores: Iterable[float]) -> float:
    """Estimate answer confidence from retrieval breadth and similarity scores.

    The score is intentionally simple and bounded in [0.0, 1.0]:
      - Coverage component: how many sources were retrieved (up to 3).
      - Similarity component: average retrieval score, clamped to [0, 1].

    Args:
        sources: Retrieved source chunks.
        scores: Similarity scores aligned with sources.

    Returns:
        A confidence value in the range [0.0, 1.0].
    """
    source_count = len(sources)
    if source_count == 0:
        return 0.0

    score_values = [max(0.0, min(1.0, float(score))) for score in scores]
    avg_score = sum(score_values) / len(score_values) if score_values else 0.0

    # Up to 3 sources contributes to retrieval coverage.
    coverage = min(source_count / 3.0, 1.0)

    confidence = (0.6 * coverage) + (0.4 * avg_score)
    return max(0.0, min(1.0, confidence))


def confidence_label(confidence: float) -> str:
    """Convert a numeric confidence score into a qualitative label."""
    if confidence >= 0.75:
        return "HIGH"
    if confidence >= 0.4:
        return "MEDIUM"
    return "LOW"
