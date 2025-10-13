"""
Lead Score API Router
=====================
Update your existing app/api/v1/score.py with these endpoints
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from app.ml.score import score_predictor
from app.core.logging import ml_logger


router = APIRouter(prefix="/ml/v1", tags=["Lead Scoring"])


# ============================================
# REQUEST/RESPONSE MODELS
# ============================================

class ScoreRequest(BaseModel):
    """Lead score prediction request."""
    messages_in_session: int = Field(..., ge=0, description="Number of messages")
    user_msg: str = Field(..., min_length=1, description="Latest user message")
    conversation_duration_minutes: float = Field(..., ge=0)
    user_response_time_avg_seconds: float = Field(..., ge=0)
    user_initiated_conversation: bool
    is_returning_customer: bool
    time_of_day: str = Field(..., description="business_hours, extended_hours, or off_hours")
    
    @validator('time_of_day')
    def validate_time_of_day(cls, v):
        valid = ['business_hours', 'extended_hours', 'off_hours']
        if v not in valid:
            raise ValueError(f"time_of_day must be one of: {valid}")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "messages_in_session": 10,
                "user_msg": "I want to buy. How much?",
                "conversation_duration_minutes": 15.5,
                "user_response_time_avg_seconds": 45.0,
                "user_initiated_conversation": True,
                "is_returning_customer": False,
                "time_of_day": "business_hours"
            }
        }


class ScoreResponse(BaseModel):
    """Lead score prediction response."""
    score: float = Field(..., description="Probability 0.0-1.0")
    category: str = Field(..., description="hot, warm, or cold")
    confidence: float = Field(..., description="Confidence 0.0-1.0")


class TrainRequest(BaseModel):
    """Training request."""
    data_path: str = Field(..., description="Path to training CSV")
    grid_search: bool = Field(False, description="Perform grid search?")
    validation_split: float = Field(0.2, ge=0.1, le=0.5)


class TrainResponse(BaseModel):
    """Training response."""
    success: bool
    metrics: dict
    message: str


# ============================================
# ENDPOINTS
# ============================================

@router.post("/predict_score", response_model=ScoreResponse)
async def predict_score(request: ScoreRequest):
    """
    Predict lead qualification score.
    
    Returns:
    - score >= 0.75: Hot lead 🔥
    - score >= 0.50: Warm lead 🌤️
    - score < 0.50: Cold lead ❄️
    """
    try:
        features = request.dict()
        result = await score_predictor.predict(features)
        
        ml_logger.info(f"Score: {result['category']} ({result['score']:.3f})")
        
        return ScoreResponse(**result)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        ml_logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@router.post("/train_score", response_model=TrainResponse)
async def train_score(request: TrainRequest):
    """
    Train the lead score model.
    
    Requires CSV with columns:
    - messages_in_session
    - user_msg
    - conversation_duration_minutes
    - user_response_time_avg_seconds
    - user_initiated_conversation
    - is_returning_customer
    - time_of_day
    - score (0 or 1)
    """
    try:
        metrics = await score_predictor.train(
            data_path=request.data_path,
            validation_split=request.validation_split,
            grid_search=request.grid_search
        )
        
        return TrainResponse(
            success=True,
            metrics=metrics,
            message=f"Model trained successfully. F1: {metrics['f1_score']:.3f}"
        )
        
    except Exception as e:
        ml_logger.error(f"Training error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/score_health")
async def score_health():
    """Health check for score predictor."""
    model_loaded = score_predictor.pipeline is not None
    
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "service": "lead_score_predictor"
    }