from pydantic import Field
from typing import Optional
from pydantic_settings import BaseSettings
import os


class Settings(BaseSettings):
    """Application settings using Pydantic BaseSettings."""
    
    # API Settings
    app_name: str = "ML Service"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    
    # Model Storage Settings
    model_storage_type: str = Field(default="s3", env="MODEL_STORAGE_TYPE")  # s3 or minio
    s3_bucket: str = Field(default="ml-models", env="S3_BUCKET")
    s3_region: str = Field(default="us-east-1", env="S3_REGION")
    s3_endpoint_url: Optional[str] = Field(default=None, env="S3_ENDPOINT_URL")  # For MinIO
    aws_access_key_id: Optional[str] = Field(default=None, env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(default=None, env="AWS_SECRET_ACCESS_KEY")
    
    # Database Settings (for RAG)
    vector_db_type: str = Field(default="chroma", env="VECTOR_DB_TYPE")
    vector_db_path: str = Field(default="./data/vectordb", env="VECTOR_DB_PATH")
    
    # ML Model Settings
    intent_model_path: str = Field(default="models/intent/latest", env="INTENT_MODEL_PATH")
    score_model_path: str = Field(default="models/score/latest", env="SCORE_MODEL_PATH")
    
    # LLM Settings (for RAG)
    llm_provider: str = Field(default="ollama", env="LLM_PROVIDER")  # ollama, openai, etc.
    llm_model: str = Field(default="llama2", env="LLM_MODEL")
    llm_base_url: str = Field(default="http://localhost:11434", env="LLM_BASE_URL")
    llm_api_key: Optional[str] = Field(default=None, env="LLM_API_KEY")
    
    # Logging Settings
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    
    # CORS Settings
    cors_origins: list = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        env="CORS_ORIGINS"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()