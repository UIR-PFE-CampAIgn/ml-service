from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from contextlib import asynccontextmanager
import uvicorn
from datetime import datetime

from app.core.config import settings
from app.core.logging import setup_logging, app_logger
from app.api.v1 import intent, score, train
from app.schemas import HealthResponse


# Application lifespan manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    setup_logging()
    app_logger.info("ML Service starting up...")
    
    # Initialize services here if needed
    try:
        # Test model registry connection
        from app.core.model_registry import model_registry
        models = model_registry.list_models()
        app_logger.info(f"Model registry connected. Available models: {list(models.keys())}")
    except Exception as e:
        app_logger.warning(f"Model registry connection failed: {e}")
    
    app_logger.info("ML Service startup complete")
    
    yield
    
    # Shutdown
    app_logger.info("ML Service shutting down...")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Machine Learning Service API with Intent Classification, Score Prediction, and RAG capabilities",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure for production
)

# Include API routers
app.include_router(
    intent.router,
    prefix="/api/v1",
    tags=["intent"],
    responses={404: {"description": "Not found"}}
)

app.include_router(
    score.router,
    prefix="/api/v1",
    tags=["score"],
    responses={404: {"description": "Not found"}}
)

#app.include_router(
#    rag.router,
#    prefix="/api/v1",
#    tags=["rag"],
#    responses={404: {"description": "Not found"}}
#)

app.include_router(
    train.router,
    prefix="/api/v1",
    tags=["training"],
    responses={404: {"description": "Not found"}}
)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with basic service information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs" if settings.debug else "disabled"
    }


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Check model registry
        model_registry_status = "healthy"
        try:
            from app.core.model_registry import model_registry
            model_registry.list_models()
        except Exception:
            model_registry_status = "unhealthy"
        
        # Check vector database (for RAG)
        vector_db_status = "healthy"
        #try:
            #from app.ml.rag import RAGChain
            #rag_chain = RAGChain()
            #rag_chain.get_vectorstore_stats()
        #except Exception:
        #    vector_db_status = "unhealthy"
        
        services = {
            "model_registry": model_registry_status,
            "vector_database": vector_db_status,
        }
        
        overall_status = "healthy" if all(
            status == "healthy" for status in services.values()
        ) else "degraded"
        
        return HealthResponse(
            status=overall_status,
            version=settings.app_version,
            timestamp=datetime.now().isoformat(),
            services=services
        )
        
    except Exception as e:
        app_logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


# Model management endpoints
@app.get("/api/v1/models")
async def list_models():
    """List available models and their versions."""
    try:
        from app.core.model_registry import model_registry
        models = model_registry.list_models()
        return {"models": models}
    except Exception as e:
        app_logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve models")


@app.get("/api/v1/models/{model_name}")
async def get_model_info(model_name: str, version: str = "latest"):
    """Get information about a specific model."""
    try:
        from app.core.model_registry import model_registry
        
        metadata = model_registry.load_metadata(model_name, version)
        if not metadata:
            raise HTTPException(status_code=404, detail="Model not found")
        
        return {"model_info": metadata}
        
    except Exception as e:
        app_logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve model information")


# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle ValueError exceptions."""
    app_logger.error(f"ValueError: {exc}")
    return HTTPException(status_code=400, detail=str(exc))


@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request, exc):
    """Handle FileNotFoundError exceptions."""
    app_logger.error(f"FileNotFoundError: {exc}")
    return HTTPException(status_code=404, detail="Resource not found")


# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )