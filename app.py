"""
SEID Engine - FastAPI Application
===================================
Production REST API for Social Engineering Detection.

Run with:
    uvicorn app:app --reload --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health         - Health check
    POST /predict        - Single prediction
    POST /batch_predict  - Batch predictions
"""

import logging
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from seid_engine import SEIDEngine

# =========================
# LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SEIDEngine.API")


# =========================
# PYDANTIC MODELS
# =========================
class PredictRequest(BaseModel):
    """Single prediction request."""
    text: str = Field(..., min_length=1, max_length=50000, description="Text to analyze")
    channel: str = Field(default="unknown", description="Channel: email, sms, unknown")
    mode: str = Field(default="balanced", description="Mode: balanced, high_recall, low_fp")
    
    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        return v
    
    @field_validator("channel")
    @classmethod
    def validate_channel(cls, v: str) -> str:
        valid_channels = {"email", "sms", "unknown"}
        if v.lower() not in valid_channels:
            raise ValueError(f"Channel must be one of: {valid_channels}")
        return v.lower()
    
    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        valid_modes = {"balanced", "high_recall", "low_fp"}
        if v.lower() not in valid_modes:
            raise ValueError(f"Mode must be one of: {valid_modes}")
        return v.lower()


class PredictResponse(BaseModel):
    """Single prediction response."""
    probability: float
    risk_tier: str
    mode: str
    channel: str
    is_malicious: bool
    roberta_enabled: bool


class BatchMessage(BaseModel):
    """Single message in batch request."""
    text: str = Field(..., min_length=1, max_length=50000)
    channel: str = Field(default="unknown")
    
    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        return v


class BatchPredictRequest(BaseModel):
    """Batch prediction request."""
    messages: List[BatchMessage] = Field(..., min_length=1, max_length=100)
    mode: str = Field(default="balanced")
    
    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        valid_modes = {"balanced", "high_recall", "low_fp"}
        if v.lower() not in valid_modes:
            raise ValueError(f"Mode must be one of: {valid_modes}")
        return v.lower()


class BatchPredictResponse(BaseModel):
    """Single item in batch response."""
    probability: float
    risk_tier: str
    is_malicious: bool


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    roberta_enabled: bool
    device: str
    mode: str


# =========================
# GLOBAL ENGINE INSTANCE
# =========================
engine: Optional[SEIDEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - load model once at startup."""
    global engine
    logger.info("=" * 60)
    logger.info("SEID Engine API Starting...")
    logger.info("=" * 60)
    
    try:
        engine = SEIDEngine(use_roberta=True)
        logger.info("Engine loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load engine: {e}")
        raise
    
    yield
    
    logger.info("SEID Engine API Shutting down...")


# =========================
# FASTAPI APP
# =========================
app = FastAPI(
    title="SEID Engine API",
    description="Social Engineering & Intrusion Detection API",
    version="3.1.0",
    lifespan=lifespan
)

# =========================
# CORS MIDDLEWARE
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# ENDPOINTS
# =========================
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.

    Returns status, RoBERTa availability, and device info.
    """
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Engine not initialized"
        )

    health = engine.get_health_status()
    return HealthResponse(
        status=health["status"],
        roberta_enabled=health["roberta_enabled"],
        device=health["device"],
        mode=health["mode"]
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(request: PredictRequest):
    """
    Single text prediction.

    Analyzes text for social engineering indicators.
    """
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Engine not initialized"
        )

    try:
        result = engine.predict(
            text=request.text,
            channel=request.channel,
            mode=request.mode
        )

        return PredictResponse(
            probability=result["probability"],
            risk_tier=result["risk_tier"],
            mode=result["mode"],
            channel=result["channel"],
            is_malicious=result["is_malicious"],
            roberta_enabled=engine.use_roberta
        )

    except ValueError as e:
        logger.warning(f"Invalid input: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed"
        )


@app.post("/batch_predict", response_model=List[BatchPredictResponse], tags=["Prediction"])
async def batch_predict(request: BatchPredictRequest):
    """
    Batch prediction for multiple messages.

    Analyzes multiple texts in a single request.
    Maximum 100 messages per request.
    """
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Engine not initialized"
        )

    try:
        results = []
        for msg in request.messages:
            result = engine.predict(
                text=msg.text,
                channel=msg.channel,
                mode=request.mode
            )
            results.append(BatchPredictResponse(
                probability=result["probability"],
                risk_tier=result["risk_tier"],
                is_malicious=result["is_malicious"]
            ))

        logger.info(f"Batch prediction complete: {len(results)} items")
        return results

    except ValueError as e:
        logger.warning(f"Invalid batch input: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch prediction failed"
        )


# =========================
# ROOT ENDPOINT
# =========================
@app.get("/", tags=["Info"])
async def root():
    """API information endpoint."""
    return {
        "name": "SEID Engine API",
        "version": "3.1.0",
        "description": "Social Engineering & Intrusion Detection",
        "endpoints": {
            "health": "GET /health",
            "predict": "POST /predict",
            "batch_predict": "POST /batch_predict"
        }
    }


# =========================
# LOCAL DEVELOPMENT
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

