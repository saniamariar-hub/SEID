"""
SEID Engine - Risk Tiers Module
=================================
Risk classification based on prediction scores.
"""

from enum import Enum
from typing import Dict


# =========================
# ENUMS
# =========================
class SecurityMode(Enum):
    """Security operation modes with associated thresholds."""
    BALANCED = "balanced"
    HIGH_RECALL = "high_recall"
    LOW_FP = "low_fp"


class Channel(Enum):
    """Communication channel types."""
    EMAIL = "email"
    SMS = "sms"
    UNKNOWN = "unknown"


# =========================
# THRESHOLDS (PRODUCTION)
# =========================

# Risk tier thresholds for classification
RISK_THRESHOLDS: Dict[str, float] = {
    "low": 0.2,
    "medium": 0.5,
    "high": 0.8
}

# Security mode thresholds (updated based on final evaluation)
MODE_THRESHOLDS: Dict[SecurityMode, float] = {
    SecurityMode.BALANCED: 0.95,
    SecurityMode.HIGH_RECALL: 0.50,
    SecurityMode.LOW_FP: 0.95
}


# =========================
# FUNCTIONS
# =========================
def get_risk_tier(score: float) -> str:
    """
    Determine risk tier based on score.

    Args:
        score: Ensemble score (0.0 to 1.0)

    Returns:
        Risk level: "Low", "Medium", "High", or "Critical"
    """
    if score < RISK_THRESHOLDS["low"]:
        return "Low"
    elif score < RISK_THRESHOLDS["medium"]:
        return "Medium"
    elif score < RISK_THRESHOLDS["high"]:
        return "High"
    else:
        return "Critical"


def get_threshold_for_mode(mode: SecurityMode) -> float:
    """
    Get decision threshold for a security mode.

    Args:
        mode: SecurityMode enum

    Returns:
        Decision threshold (0.0 to 1.0)
    """
    return MODE_THRESHOLDS.get(mode, MODE_THRESHOLDS[SecurityMode.BALANCED])


def parse_mode(mode: str) -> SecurityMode:
    """
    Parse security mode from string.

    Args:
        mode: Mode string ("balanced", "high_recall", "low_fp")

    Returns:
        SecurityMode enum

    Raises:
        ValueError: If mode string is invalid
    """
    if isinstance(mode, SecurityMode):
        return mode

    mode_map = {
        "balanced": SecurityMode.BALANCED,
        "high_recall": SecurityMode.HIGH_RECALL,
        "low_fp": SecurityMode.LOW_FP
    }

    if mode.lower() not in mode_map:
        raise ValueError(f"Invalid mode: {mode}. Use: balanced, high_recall, low_fp")

    return mode_map[mode.lower()]


def parse_channel(channel: str) -> Channel:
    """
    Parse channel from string.

    Args:
        channel: Channel string ("email", "sms", "unknown")

    Returns:
        Channel enum
    """
    if isinstance(channel, Channel):
        return channel

    channel_map = {
        "email": Channel.EMAIL,
        "sms": Channel.SMS,
        "unknown": Channel.UNKNOWN
    }

    return channel_map.get(channel.lower(), Channel.UNKNOWN)

