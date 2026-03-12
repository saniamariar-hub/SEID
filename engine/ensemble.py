"""
SEID Engine - Ensemble Module
==============================
Weighted ensemble scoring for combining TF-IDF and RoBERTa predictions.
"""

import logging
from typing import Tuple

logger = logging.getLogger("SEIDEngine.ensemble")


# =========================
# DEFAULT WEIGHTS
# =========================
DEFAULT_TFIDF_WEIGHT = 0.4
DEFAULT_ROBERTA_WEIGHT = 0.6


def validate_weights(tfidf_weight: float, roberta_weight: float) -> bool:
    """
    Validate that ensemble weights sum to 1.0.

    Args:
        tfidf_weight: Weight for TF-IDF model
        roberta_weight: Weight for RoBERTa model

    Returns:
        True if valid

    Raises:
        ValueError: If weights don't sum to 1.0
    """
    if abs((tfidf_weight + roberta_weight) - 1.0) > 0.001:
        raise ValueError(
            f"Ensemble weights must sum to 1.0. "
            f"Got tfidf={tfidf_weight}, roberta={roberta_weight}, "
            f"sum={tfidf_weight + roberta_weight}"
        )
    return True


def compute_ensemble_score(
    tfidf_prob: float,
    roberta_prob: float,
    tfidf_weight: float = DEFAULT_TFIDF_WEIGHT,
    roberta_weight: float = DEFAULT_ROBERTA_WEIGHT,
    use_roberta: bool = True
) -> float:
    """
    Compute weighted ensemble score.

    Formula: final_score = roberta_weight * roberta_prob + tfidf_weight * tfidf_prob

    Args:
        tfidf_prob: Probability from TF-IDF model (0.0 to 1.0)
        roberta_prob: Probability from RoBERTa model (0.0 to 1.0)
        tfidf_weight: Weight for TF-IDF model (default: 0.4)
        roberta_weight: Weight for RoBERTa model (default: 0.6)
        use_roberta: Whether RoBERTa is enabled

    Returns:
        Weighted ensemble score (0.0 to 1.0)
    """
    # If RoBERTa is disabled, return TF-IDF only
    if not use_roberta:
        return tfidf_prob

    # Compute weighted ensemble
    final_score = (roberta_weight * roberta_prob) + (tfidf_weight * tfidf_prob)

    return final_score


def get_component_scores(
    tfidf_prob: float,
    roberta_prob: float,
    use_roberta: bool = True
) -> Tuple[float, float, float]:
    """
    Get all component scores for logging/debugging.

    Args:
        tfidf_prob: TF-IDF probability
        roberta_prob: RoBERTa probability
        use_roberta: Whether RoBERTa is enabled

    Returns:
        Tuple of (tfidf_score, roberta_score, ensemble_score)
    """
    ensemble = compute_ensemble_score(
        tfidf_prob=tfidf_prob,
        roberta_prob=roberta_prob,
        use_roberta=use_roberta
    )

    return (tfidf_prob, roberta_prob if use_roberta else 0.0, ensemble)

