"""
SEID Engine - Engine Package
=============================
Social Engineering & Intrusion Detection Engine modules.

This package contains modular components for:
- Text preprocessing and sanitization
- TF-IDF and RoBERTa inference
- Ensemble scoring
- Risk tier classification
"""

from .preprocessing import preprocess, sanitize_input, MAX_INPUT_LENGTH, MAX_CHAR_LENGTH
from .risk_tiers import (
    SecurityMode,
    Channel,
    RISK_THRESHOLDS,
    MODE_THRESHOLDS,
    get_risk_tier,
    get_threshold_for_mode,
    parse_mode,
    parse_channel
)
from .ensemble import (
    DEFAULT_TFIDF_WEIGHT,
    DEFAULT_ROBERTA_WEIGHT,
    validate_weights,
    compute_ensemble_score,
    get_component_scores
)
from .inference import TFIDFInference, RoBERTaInference

__all__ = [
    # Preprocessing
    "preprocess",
    "sanitize_input",
    "MAX_INPUT_LENGTH",
    "MAX_CHAR_LENGTH",
    # Risk Tiers
    "SecurityMode",
    "Channel",
    "RISK_THRESHOLDS",
    "MODE_THRESHOLDS",
    "get_risk_tier",
    "get_threshold_for_mode",
    "parse_mode",
    "parse_channel",
    # Ensemble
    "DEFAULT_TFIDF_WEIGHT",
    "DEFAULT_ROBERTA_WEIGHT",
    "validate_weights",
    "compute_ensemble_score",
    "get_component_scores",
    # Inference
    "TFIDFInference",
    "RoBERTaInference",
]

