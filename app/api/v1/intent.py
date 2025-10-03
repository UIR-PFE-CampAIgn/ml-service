from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Dict, Any

from app.ml.intent import IntentPredictor
from app.schemas import IntentPredictionRequest, IntentPredictionResponse

router = APIRouter()


class IntentRequest(BaseModel):
    text: str


class IntentResponse(BaseModel):
    intent: str
    confidence: float


@router.get("/predict_intent")
async def predict_intent(text: str) -> IntentResponse:
    """
    Predict intent from input text using TF-IDF + SVM pipeline.
    
    Args:
        text: Input text to classify
        
    Returns:
        Predicted intent and confidence score
    """
    predictor = IntentPredictor()
    result = await predictor.predict(text)
    
    return IntentResponse(
        intent=result["intent"],
        confidence=result["confidence"]
    )