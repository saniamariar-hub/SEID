"""
SEID Engine - Social Engineering & Intrusion Detection
=======================================================
Production-grade inference engine combining TF-IDF and RoBERTa models
for malicious content detection (phishing/smishing).

Author: Applied ML / Security NLP Team
Version: 3.1.0

Architecture:
    models/
        tfidf_model/
        roberta_malicious_classifier/
    engine/
        preprocessing.py
        inference.py
        ensemble.py
        risk_tiers.py
"""

import logging
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from pathlib import Path

# Engine modules
from engine import (
    # Preprocessing
    preprocess,
    sanitize_input,
    MAX_INPUT_LENGTH,
    MAX_CHAR_LENGTH,
    # Risk Tiers
    SecurityMode,
    Channel,
    MODE_THRESHOLDS,
    RISK_THRESHOLDS,
    get_risk_tier,
    get_threshold_for_mode,
    parse_mode,
    parse_channel,
    # Ensemble
    DEFAULT_TFIDF_WEIGHT,
    DEFAULT_ROBERTA_WEIGHT,
    validate_weights,
    compute_ensemble_score,
    # Inference
    TFIDFInference,
    RoBERTaInference,
)

# =========================
# LOGGING CONFIGURATION
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SEIDEngine")


# =========================
# STARTUP INFO DATA CLASS
# =========================
@dataclass
class StartupInfo:
    """Structured startup validation info."""
    tfidf_loaded: bool = False
    roberta_enabled: bool = False
    roberta_path_exists: bool = False
    torch_installed: bool = False
    transformers_installed: bool = False
    device: str = "cpu"
    mode: str = "tfidf_only"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def check_dependencies() -> Dict[str, bool]:
    """
    Check if torch and transformers are installed.

    Returns:
        Dict with torch_installed and transformers_installed flags
    """
    result = {
        "torch_installed": False,
        "transformers_installed": False,
        "device": "cpu"
    }

    try:
        import torch
        result["torch_installed"] = True
        result["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        pass

    try:
        import transformers
        result["transformers_installed"] = True
    except ImportError:
        pass

    return result


def check_roberta_path(model_path: str) -> bool:
    """
    Verify RoBERTa model directory exists and contains required files.

    Args:
        model_path: Path to RoBERTa model directory

    Returns:
        True if path exists and contains model files
    """
    path = Path(model_path)
    if not path.exists():
        return False

    # Check for essential model files
    required_files = ["config.json"]
    for f in required_files:
        if not (path / f).exists():
            return False

    return True


# =========================
# DATA CLASS
# =========================
@dataclass
class PredictionResult:
    """Structured prediction result."""
    probability: float
    risk_tier: str
    is_malicious: bool
    tfidf_score: float
    roberta_score: float
    channel: str
    mode: str
    threshold: float
    timestamp: str
    explanations: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =========================
# MODEL PATHS (PRODUCTION)
# =========================
DEFAULT_TFIDF_VECTORIZER_PATH = "./models/tfidf_model/tfidf_vectorizer.joblib"
DEFAULT_TFIDF_MODEL_PATH = "./models/tfidf_model/logistic_regression.joblib"
DEFAULT_ROBERTA_PATH = "./models/roberta_malicious_classifier"


# =========================
# MAIN ENGINE CLASS
# =========================
class SEIDEngine:
    """
    Social Engineering & Intrusion Detection Engine v3.0

    Production-grade ensemble classifier with:
    - Channel-aware predictions (email, sms)
    - Multiple security modes (balanced, high_recall, low_fp)
    - Batch processing support
    - Structured JSON logging
    - Input safety handling
    - Safe model loading (graceful fallback)

    Ensemble Formula:
        final_score = 0.6 * roberta_score + 0.4 * tfidf_score
    """

    def __init__(
        self,
        tfidf_vectorizer_path: str = DEFAULT_TFIDF_VECTORIZER_PATH,
        tfidf_model_path: str = DEFAULT_TFIDF_MODEL_PATH,
        roberta_model_path: str = DEFAULT_ROBERTA_PATH,
        tfidf_weight: float = DEFAULT_TFIDF_WEIGHT,
        roberta_weight: float = DEFAULT_ROBERTA_WEIGHT,
        use_roberta: bool = True,
        default_mode: str = "balanced"
    ):
        """
        Initialize the SEID Engine with startup validation.

        Args:
            tfidf_vectorizer_path: Path to TF-IDF vectorizer
            tfidf_model_path: Path to Logistic Regression model
            roberta_model_path: Path to RoBERTa model directory
            tfidf_weight: Weight for TF-IDF (default: 0.4)
            roberta_weight: Weight for RoBERTa (default: 0.6)
            use_roberta: Enable RoBERTa (default: True)
            default_mode: Default security mode
        """
        # Initialize startup info for validation tracking
        self.startup_info = StartupInfo()

        # Validate and store weights
        validate_weights(tfidf_weight, roberta_weight)
        self.tfidf_weight = tfidf_weight
        self.roberta_weight = roberta_weight
        self.default_mode = parse_mode(default_mode)

        logger.info("=" * 60)
        logger.info("Initializing SEID Engine v3.1")
        logger.info("=" * 60)

        # === STARTUP VALIDATION ===
        # Step 1: Check dependencies (torch, transformers)
        deps = check_dependencies()
        self.startup_info.torch_installed = deps["torch_installed"]
        self.startup_info.transformers_installed = deps["transformers_installed"]
        self.startup_info.device = deps["device"]

        logger.info(f"[Validation] torch installed: {deps['torch_installed']}")
        logger.info(f"[Validation] transformers installed: {deps['transformers_installed']}")
        logger.info(f"[Validation] Device: {deps['device']}")

        # Step 2: Check RoBERTa model path
        self.startup_info.roberta_path_exists = check_roberta_path(roberta_model_path)
        logger.info(f"[Validation] RoBERTa path exists: {self.startup_info.roberta_path_exists}")

        logger.info(f"Ensemble weights: TF-IDF={tfidf_weight}, RoBERTa={roberta_weight}")
        logger.info(f"Default mode: {self.default_mode.value}")

        # Initialize TF-IDF inference
        self.tfidf = TFIDFInference(
            vectorizer_path=tfidf_vectorizer_path,
            model_path=tfidf_model_path
        )
        self.startup_info.tfidf_loaded = True

        # Initialize RoBERTa inference (optional, safe loading)
        self.roberta = None
        self.use_roberta = False
        self.device = deps["device"]

        if use_roberta:
            # Only attempt RoBERTa loading if dependencies are available
            if deps["torch_installed"] and deps["transformers_installed"]:
                if self.startup_info.roberta_path_exists:
                    self.roberta = RoBERTaInference(
                        model_path=roberta_model_path,
                        max_length=MAX_INPUT_LENGTH
                    )
                    self.use_roberta = self.roberta.available
                    self.startup_info.roberta_enabled = self.use_roberta

                    if self.use_roberta:
                        logger.info("[Validation] RoBERTa model loaded successfully")
                    else:
                        logger.warning("[Validation] RoBERTa loading failed. Using TF-IDF only.")
                else:
                    logger.warning(f"[Validation] RoBERTa path not found: {roberta_model_path}")
            else:
                missing = []
                if not deps["torch_installed"]:
                    missing.append("torch")
                if not deps["transformers_installed"]:
                    missing.append("transformers")
                logger.warning(f"[Validation] Missing dependencies: {', '.join(missing)}. RoBERTa disabled.")
        else:
            logger.info("RoBERTa disabled by configuration. Using TF-IDF only.")

        # Set mode in startup info
        self.startup_info.mode = "ensemble" if self.use_roberta else "tfidf_only"

        # Log structured startup info
        logger.info("=" * 60)
        logger.info("STARTUP VALIDATION COMPLETE")
        logger.info(self.startup_info.to_json())
        logger.info("=" * 60)
        logger.info(f"SEID Engine v3.1 initialized - Mode: {self.startup_info.mode.upper()}")
        logger.info("=" * 60)

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status for API endpoint.

        Returns:
            Dict with status, roberta_enabled, device
        """
        return {
            "status": "ok",
            "roberta_enabled": self.use_roberta,
            "device": self.device,
            "mode": self.startup_info.mode,
            "tfidf_loaded": self.startup_info.tfidf_loaded
        }

    def predict(
        self,
        text: str,
        channel: str = "unknown",
        mode: Optional[str] = None,
        include_explanation: bool = False
    ) -> Dict[str, Any]:
        """
        Get prediction with channel awareness and security mode.

        Args:
            text: Input text to classify
            channel: Communication channel ("email", "sms", "unknown")
            mode: Security mode ("balanced", "high_recall", "low_fp")
            include_explanation: Whether to include explanation placeholders

        Returns:
            Dictionary with structured prediction results:
            {
                "probability": float,
                "risk_tier": str,
                "is_malicious": bool,
                "tfidf_score": float,
                "roberta_score": float,
                "channel": str,
                "mode": str,
                "threshold": float,
                "timestamp": str
            }
        """
        # Parse inputs
        channel_enum = parse_channel(channel)
        mode_enum = parse_mode(mode) if mode else self.default_mode
        threshold = get_threshold_for_mode(mode_enum)
        timestamp = datetime.now(timezone.utc).isoformat()

        # Get probabilities from models
        tfidf_prob = self.tfidf.predict(text)
        roberta_prob = self.roberta.predict(text) if self.use_roberta else 0.0

        # Compute ensemble score
        final_score = compute_ensemble_score(
            tfidf_prob=tfidf_prob,
            roberta_prob=roberta_prob,
            tfidf_weight=self.tfidf_weight,
            roberta_weight=self.roberta_weight,
            use_roberta=self.use_roberta
        )

        # Determine classification
        is_malicious = final_score >= threshold
        risk_tier = get_risk_tier(final_score)

        # Build result
        result = PredictionResult(
            probability=round(final_score, 4),
            risk_tier=risk_tier,
            is_malicious=is_malicious,
            tfidf_score=round(tfidf_prob, 4),
            roberta_score=round(roberta_prob, 4),
            channel=channel_enum.value,
            mode=mode_enum.value,
            threshold=threshold,
            timestamp=timestamp,
            explanations=[] if include_explanation else None
        )

        # Structured logging
        self._log_prediction(result, text)

        return result.to_dict()

    # Alias for backward compatibility
    def predict_proba(self, text: str, **kwargs) -> Dict[str, Any]:
        """Alias for predict() - backward compatibility."""
        return self.predict(text, **kwargs)

    def _log_prediction(self, result: PredictionResult, text: Any) -> None:
        """Log prediction in structured JSON format."""
        text_len = len(text) if text is not None and isinstance(text, str) else 0
        log_entry = {
            "timestamp": result.timestamp,
            "probability": result.probability,
            "risk_tier": result.risk_tier,
            "is_malicious": result.is_malicious,
            "mode": result.mode,
            "channel": result.channel,
            "text_length": text_len
        }
        logger.info(f"Prediction: {json.dumps(log_entry)}")

    def predict_batch(
        self,
        texts: List[str],
        channel: str = "unknown",
        mode: Optional[str] = None,
        include_explanation: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Batch prediction for multiple texts.

        Args:
            texts: List of input texts to classify
            channel: Communication channel for all texts
            mode: Security mode for all predictions
            include_explanation: Whether to include explanations

        Returns:
            List of prediction result dictionaries
        """
        logger.info(f"Batch prediction started: {len(texts)} texts")

        results = []
        for i, text in enumerate(texts):
            try:
                result = self.predict(
                    text=text,
                    channel=channel,
                    mode=mode,
                    include_explanation=include_explanation
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Batch item {i} failed: {e}")
                results.append({
                    "error": str(e),
                    "index": i,
                    "probability": None,
                    "risk_tier": "Error"
                })

        logger.info(f"Batch prediction complete: {len(results)} results")
        return results

    def explain(
        self,
        text: str,
        channel: str = "unknown",
        mode: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate full prediction with explanation.

        Args:
            text: Input text to analyze
            channel: Communication channel
            mode: Security mode

        Returns:
            Dictionary with full prediction details and explanations
        """
        return self.predict(
            text=text,
            channel=channel,
            mode=mode,
            include_explanation=True
        )

    def __call__(
        self,
        text: str,
        channel: str = "unknown",
        mode: Optional[str] = None
    ) -> Dict[str, Any]:
        """Callable interface for single prediction."""
        return self.predict(text=text, channel=channel, mode=mode)


# =========================
# STANDALONE USAGE
# =========================
if __name__ == "__main__":
    print("=" * 70)
    print("SEID ENGINE v3.0 - PRODUCTION DEMO")
    print("=" * 70)

    try:
        # Initialize engine with production settings
        engine = SEIDEngine(
            use_roberta=True,  # Try to load RoBERTa
            default_mode="balanced",
            tfidf_weight=0.4,
            roberta_weight=0.6
        )

        # Test samples with channel information
        test_samples = [
            ("Hi John, please find attached the quarterly report.", "email"),
            ("URGENT: Your account suspended. Click here now!", "email"),
            ("Congratulations! You've won $1000. Text CLAIM to 55555", "sms"),
            ("Meeting at 3pm tomorrow.", "email"),
            ("Your PayPal needs verification at paypa1-secure.com", "email"),
            ("", "email"),  # Empty input test
        ]

        # === DEMO 1: Security Modes with Updated Thresholds ===
        print("\n" + "=" * 70)
        print("DEMO 1: SECURITY MODES (PRODUCTION THRESHOLDS)")
        print("  balanced=0.95, high_recall=0.50, low_fp=0.95")
        print("=" * 70)

        sample_text = "URGENT: Verify your account immediately at secure-bank.com"
        for mode in ["balanced", "high_recall", "low_fp"]:
            result = engine.predict(sample_text, channel="email", mode=mode)
            print(f"\nMode: {mode.upper()} (threshold={result['threshold']})")
            print(f"  Probability: {result['probability']:.4f}")
            print(f"  Malicious: {result['is_malicious']}")
            print(f"  Risk Tier: {result['risk_tier']}")

        # === DEMO 2: Channel-Aware Predictions ===
        print("\n" + "=" * 70)
        print("DEMO 2: CHANNEL-AWARE PREDICTIONS")
        print("=" * 70)

        for text, channel in test_samples[:5]:
            result = engine.predict(text, channel=channel, mode="high_recall")
            status = "🚨 MALICIOUS" if result['is_malicious'] else "✅ SAFE"
            print(f"\n[{channel.upper()}] {text[:50]}...")
            print(f"  Probability: {result['probability']:.4f} | {status} | {result['risk_tier']}")

        # === DEMO 3: Batch Prediction ===
        print("\n" + "=" * 70)
        print("DEMO 3: BATCH PREDICTION")
        print("=" * 70)

        batch_texts = [s[0] for s in test_samples]
        batch_results = engine.predict_batch(batch_texts, channel="email", mode="high_recall")

        print(f"\nProcessed {len(batch_results)} texts:\n")
        for i, result in enumerate(batch_results):
            prob = result.get('probability', 'N/A')
            risk = result.get('risk_tier', 'Error')
            print(f"  [{i+1}] Probability: {prob} | Risk: {risk}")

        # === DEMO 4: Input Safety ===
        print("\n" + "=" * 70)
        print("DEMO 4: INPUT SAFETY HANDLING")
        print("=" * 70)

        edge_cases = [
            (None, "None input"),
            ("", "Empty string"),
            ("   ", "Whitespace only"),
            ("Test\x00with\x00nulls", "Null characters"),
            (12345, "Non-string input"),
        ]

        for test_input, description in edge_cases:
            result = engine.predict(test_input, channel="unknown")
            print(f"  {description}: Probability={result['probability']:.4f}")

        # === DEMO 5: Model Status ===
        print("\n" + "=" * 70)
        print("DEMO 5: MODEL STATUS")
        print("=" * 70)
        print(f"  TF-IDF Model: ✅ Loaded")
        print(f"  RoBERTa Model: {'✅ Loaded' if engine.use_roberta else '⚠️ Not available (TF-IDF only)'}")
        print(f"  Ensemble Weights: TF-IDF={engine.tfidf_weight}, RoBERTa={engine.roberta_weight}")

        print("\n" + "=" * 70)
        print("DEMO COMPLETE - SEID Engine v3.0 Ready for Production")
        print("=" * 70)

    except FileNotFoundError as e:
        print(f"\nERROR: Model files not found: {e}")
        print("Ensure models are in ./models/ directory:")
        print("  - ./models/tfidf_model/tfidf_vectorizer.joblib")
        print("  - ./models/tfidf_model/logistic_regression.joblib")
        print("  - ./models/roberta_malicious_classifier/ (optional)")

