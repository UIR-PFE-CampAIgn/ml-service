from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# Intent Prediction Schemas
class IntentPredictionRequest(BaseModel):
    text: str = Field(..., description="Input text for intent classification")


class IntentPrediction(BaseModel):
    intent: str
    confidence: float


class IntentPredictionResponse(BaseModel):
    intent: str
    confidence: float
    top_predictions: List[IntentPrediction]


# Score Prediction Schemas
class ScorePredictionRequest(BaseModel):
    features: Dict[str, Any] = Field(
        ..., description="Feature dictionary for score prediction"
    )
    model_type: str = Field(
        default="logistic_regression", description="Model type to use"
    )


class ScorePredictionResponse(BaseModel):
    score: float
    probability: float
    probabilities: Dict[str, float]
    model_used: str


## RAG Schemas removed

##campagin generator shemas

class CampaignRequest(BaseModel):
    prompt: str
    timezone: str = "UTC"


class MessageTemplate(BaseModel):
    message: str
    target_segment: str
    approach: str
    personalization_tips: str


class SendSchedule(BaseModel):
    segment: str
    send_datetime: str
    reasoning: str
    priority: str


class CampaignStrategy(BaseModel):
    target_segments: List[str]
    reasoning: str
    campaign_type: str
    key_message: str
    expected_response_rates: Dict[str, str]


class CampaignResponse(BaseModel):
    strategy: CampaignStrategy
    templates: List[MessageTemplate]
    schedule: List[SendSchedule]
    insights: Dict[str, List[str]]
# Training Schemas
class ModelType(str, Enum):
    INTENT = "intent"
    SCORE = "score"


class TrainingRequest(BaseModel):
    model_type: ModelType
    training_data_path: str = Field(..., description="Path to training data file")
    hyperparameters: Optional[Dict[str, Any]] = Field(
        default=None, description="Model hyperparameters"
    )
    validation_split: float = Field(
        default=0.2, ge=0.1, le=0.5, description="Validation split ratio"
    )


class TrainingResponse(BaseModel):
    job_id: str
    status: str
    message: str


class TrainingStatus(BaseModel):
    job_id: str
    status: str  # queued, running, completed, failed
    progress: float = Field(ge=0.0, le=1.0)
    metrics: Optional[Dict[str, float]] = None
    error: Optional[str] = None


# Health Check Schema
class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str
    services: Dict[str, str]


# Error Schemas
class ErrorResponse(BaseModel):
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None


# Batch Prediction Schemas
class BatchIntentRequest(BaseModel):
    texts: List[str] = Field(
        ..., min_items=1, description="List of texts for batch intent prediction"
    )


class BatchIntentResponse(BaseModel):
    predictions: List[IntentPredictionResponse]


class BatchScoreRequest(BaseModel):
    features_list: List[Dict[str, Any]] = Field(
        ..., min_items=1, description="List of feature dictionaries"
    )
    model_type: str = Field(
        default="logistic_regression", description="Model type to use"
    )


class BatchScoreResponse(BaseModel):
    predictions: List[ScorePredictionResponse]


# Model Info Schemas
class ModelInfo(BaseModel):
    name: str
    version: str
    algorithm: str
    metrics: Dict[str, float]
    created_at: str
    training_samples: int


class ModelsListResponse(BaseModel):
    models: Dict[str, List[str]]  # model_name -> list of versions


class ModelDetailsResponse(BaseModel):
    model_info: ModelInfo
    metadata: Dict[str, Any]
