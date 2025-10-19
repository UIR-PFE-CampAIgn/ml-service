from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Tuple
import uuid
from enum import Enum

from app.ml.intent import IntentPredictor
from app.ml.score import ScorePredictor
from app.ml.train_intent import train_and_save_intent

router = APIRouter()


class ModelType(str, Enum):
    INTENT = "intent"
    SCORE = "score"


class TrainingRequest(BaseModel):
    model_type: ModelType
    training_data_path: str

    # Intent training args (mapped to train_and_save_intent)
    label_col: str = Field(default="intent")
    text_col: Optional[str] = Field(default=None)
    c_values: Optional[List[float]] = Field(default=None)
    # Represent ngram ranges as list of [min, max] pairs, e.g. [[1,1],[1,2]]
    ngram_ranges: Optional[List[List[int]]] = Field(default=None)
    metrics_output_path: Optional[str] = Field(default=None)
    cv: int = Field(default=3)
    scoring: str = Field(default="f1_macro")
    model_name: str = Field(default="intent")
    save_to_registry: Optional[bool] = Field(default=True)

    # Score training args
    hyperparameters: Optional[Dict[str, Any]] = None
    validation_split: float = 0.2


class TrainingResponse(BaseModel):
    job_id: str
    status: str
    message: str


class TrainingStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    metrics: Optional[Dict[str, float]] = None
    error: Optional[str] = None


# In-memory job tracking (in production, use Redis or database)
training_jobs = {}


@router.post("/train")
async def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks
) -> TrainingResponse:
    """
    Start model training job.
    
    Args:
        request: Training configuration
        background_tasks: FastAPI background tasks
        
    Returns:
        Training job ID and initial status
    """
    job_id = str(uuid.uuid4())
    
    # Initialize job status
    training_jobs[job_id] = TrainingStatus(
        job_id=job_id,
        status="queued",
        progress=0.0
    )
    
    # Start training in background
    background_tasks.add_task(
        run_training_job,
        job_id=job_id,
        request=request
    )
    
    return TrainingResponse(
        job_id=job_id,
        status="queued",
        message="Training job started successfully"
    )


@router.get("/train/{job_id}/status")
async def get_training_status(job_id: str) -> TrainingStatus:
    """
    Get status of a training job.
    
    Args:
        job_id: Training job ID
        
    Returns:
        Current training status and metrics
    """
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    return training_jobs[job_id]


async def run_training_job(job_id: str, request: TrainingRequest):
    """
    Background task to run model training.
    
    Args:
        job_id: Training job ID
        request: Training configuration
    """
    try:
        # Update status to running
        training_jobs[job_id].status = "running"
        training_jobs[job_id].progress = 0.1
        
        if request.model_type == ModelType.INTENT:
            # Mirror scripts/train_intent.py argument mapping
            # Convert [[a,b], [c,d]] to [(a,b), (c,d)] if provided
            ngram_ranges: Optional[List[Tuple[int, int]]] = None
            if request.ngram_ranges:
                try:
                    ngram_ranges = [(int(a), int(b)) for a, b in request.ngram_ranges]
                except Exception as _:
                    ngram_ranges = None

            result = train_and_save_intent(
                data_path=request.training_data_path,
                label_col=request.label_col,
                text_col=request.text_col,
                c_values=request.c_values,
                ngram_ranges=ngram_ranges,
                metrics_output_path=request.metrics_output_path,
                cv=request.cv,
                scoring=request.scoring,
                model_name=request.model_name,
                save_to_registry=True if request.save_to_registry is None else request.save_to_registry,
            )
            metrics = result.get("metrics", {})
        elif request.model_type == ModelType.SCORE:
            predictor = ScorePredictor()
            metrics = await predictor.train(
                data_path=request.training_data_path,
                hyperparameters=request.hyperparameters or {},
                validation_split=request.validation_split,
            )
        
        # Update completion status
        training_jobs[job_id].status = "completed"
        training_jobs[job_id].progress = 1.0
        training_jobs[job_id].metrics = metrics
        
    except Exception as e:
        training_jobs[job_id].status = "failed"
        training_jobs[job_id].error = str(e)
