"""
SEID Engine - Inference Module
================================
Core inference wrappers for TF-IDF and RoBERTa models.
"""

import logging
from typing import Optional, Any
from pathlib import Path

import joblib

from .preprocessing import preprocess, MAX_INPUT_LENGTH

logger = logging.getLogger("SEIDEngine.inference")


# =========================
# TF-IDF INFERENCE
# =========================
class TFIDFInference:
    """TF-IDF + Logistic Regression inference wrapper."""

    def __init__(
        self,
        vectorizer_path: str = "./models/tfidf_model/tfidf_vectorizer.joblib",
        model_path: str = "./models/tfidf_model/logistic_regression.joblib"
    ):
        """
        Initialize TF-IDF inference.

        Args:
            vectorizer_path: Path to saved TF-IDF vectorizer
            model_path: Path to saved Logistic Regression model
        """
        self.vectorizer_path = vectorizer_path
        self.model_path = model_path
        self.vectorizer = None
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load TF-IDF vectorizer and Logistic Regression model."""
        try:
            logger.info(f"Loading TF-IDF vectorizer from {self.vectorizer_path}")
            self.vectorizer = joblib.load(self.vectorizer_path)

            logger.info(f"Loading Logistic Regression model from {self.model_path}")
            self.model = joblib.load(self.model_path)

            logger.info("TF-IDF model loaded successfully")
        except FileNotFoundError as e:
            logger.error(f"TF-IDF model files not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load TF-IDF model: {e}")
            raise

    def predict(self, text: Any) -> float:
        """
        Get malicious probability from TF-IDF model.

        Args:
            text: Input text (raw)

        Returns:
            Probability of malicious class (0.0 to 1.0)
        """
        text_clean = preprocess(text)
        if not text_clean:
            return 0.0

        X = self.vectorizer.transform([text_clean])
        prob = self.model.predict_proba(X)[0, 1]
        return float(prob)


# =========================
# ROBERTA INFERENCE
# =========================
class RoBERTaInference:
    """RoBERTa transformer inference wrapper."""

    def __init__(
        self,
        model_path: str = "./models/roberta_malicious_classifier",
        max_length: int = MAX_INPUT_LENGTH
    ):
        """
        Initialize RoBERTa inference.

        Args:
            model_path: Path to saved RoBERTa model directory
            max_length: Maximum token length for truncation
        """
        self.model_path = model_path
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        self.device = None
        self.available = False

        self._load_model()

    def _load_model(self) -> None:
        """Load RoBERTa model and tokenizer from HuggingFace."""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            logger.info(f"Loading RoBERTa model from {self.model_path}")

            # Check if model path exists
            if not Path(self.model_path).exists():
                logger.warning(f"RoBERTa model path not found: {self.model_path}")
                self.available = False
                return

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.eval()

            # Set device (CUDA if available)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.available = True

            logger.info(f"RoBERTa model loaded successfully (device: {self.device})")

        except ImportError:
            logger.warning("PyTorch/Transformers not installed. RoBERTa disabled.")
            self.available = False
        except Exception as e:
            logger.warning(f"Failed to load RoBERTa model: {e}. RoBERTa disabled.")
            self.available = False

    def predict(self, text: Any) -> float:
        """
        Get malicious probability from RoBERTa model.

        Args:
            text: Input text (raw)

        Returns:
            Probability of malicious class (0.0 to 1.0)
        """
        if not self.available or self.model is None:
            return 0.0

        import torch

        text_clean = preprocess(text)
        if not text_clean:
            return 0.0

        encoding = self.tokenizer(
            text_clean,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            prob = probs[0, 1].item()

        return float(prob)

