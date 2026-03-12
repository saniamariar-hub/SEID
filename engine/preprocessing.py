"""
SEID Engine - Preprocessing Module
===================================
Input sanitization and text preprocessing for inference.
"""

import re
import logging
from typing import Any

logger = logging.getLogger("SEIDEngine.preprocessing")


# =========================
# CONSTANTS
# =========================
MAX_INPUT_LENGTH = 256  # tokens (RoBERTa truncation)
MAX_CHAR_LENGTH = 10000  # characters (safety truncation)


def sanitize_input(text: Any) -> str:
    """
    Sanitize and validate input text.

    Handles:
    - None/empty input
    - Non-string types
    - Null characters
    - Extremely long input (truncation)

    Args:
        text: Raw input (any type)

    Returns:
        Sanitized string
    """
    # Handle None or empty
    if text is None:
        logger.warning("Received None input, returning empty string")
        return ""

    # Convert to string
    if not isinstance(text, str):
        text = str(text)

    # Strip null characters
    text = text.replace('\x00', '')

    # Truncate if too long (character-based first pass)
    if len(text) > MAX_CHAR_LENGTH:
        logger.warning(f"Input truncated from {len(text)} to {MAX_CHAR_LENGTH} chars")
        text = text[:MAX_CHAR_LENGTH]

    return text


def preprocess(text: Any) -> str:
    """
    Preprocess input text for inference.

    Args:
        text: Raw input text

    Returns:
        Cleaned text ready for model inference
    """
    # Sanitize input first
    text = sanitize_input(text)

    # Handle empty after sanitization
    if not text.strip():
        return ""

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

