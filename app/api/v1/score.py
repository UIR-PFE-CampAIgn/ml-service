from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Any, List

from app.ml.score import ScorePredictor

router = APIRouter()


class ScoreRequest(BaseModel):
    features: Dict[str, Any]
    model_type: str = "logistic_regression"  # or "xgboost"


class ScoreResponse(BaseModel):
    score: float
    probability: float
    model_used: str


@router.get("/predict_score")
async def predict_score(
    features: Dict[str, Any],
    model_type: str = "logistic_regression"
) -> ScoreResponse:
    """
    Predict score using LogReg or XGBoost pipeline.
    
    Args:
        features: Input features dictionary
        model_type: Model type to use ("logistic_regression" or "xgboost")
        
    Returns:
        Predicted score, probability, and model information
    """
    predictor = ScorePredictor(model_type=model_type)
    result = await predictor.predict(features)
    
    return ScoreResponse(
        score=result["score"],
        probability=result["probability"],
        model_used=model_type
    )