# ML Service - Comprehensive Endpoint Documentation

## Overview

This Machine Learning Service is a FastAPI-based microservice that provides three core AI capabilities: **Intent Classification**, **Score Prediction**, and **Retrieval-Augmented Generation (RAG)**. The service uses a modular architecture with Docker containerization, supporting multiple ML algorithms and providing both training and inference endpoints.

## Architecture

The service operates on a three-layer architecture:
1. **API Layer** (`/api/v1/`) - FastAPI routers handling HTTP requests/responses
2. **ML Layer** (`/ml/`) - Core machine learning logic and model implementations  
3. **Core Layer** (`/core/`) - Configuration, logging, and model registry management

## Endpoint Categories

### 1. Health & System Management Endpoints

#### **GET /** - Root Service Information
Returns basic service metadata including name, version, and status.

#### **GET /health** - Health Check
Performs comprehensive health monitoring by checking:
- Model registry connectivity and available models
- Vector database status for RAG functionality  
- Overall service health (healthy/degraded based on component status)

#### **GET /api/v1/models** - List Available Models
Retrieves all trained models from the model registry with their versions and metadata.

#### **GET /api/v1/models/{model_name}** - Get Model Information
Returns detailed information about a specific model including training metrics, hyperparameters, and metadata.

---

### 2. Intent Classification Endpoints

#### **GET /api/v1/predict_intent**
**Purpose**: Classifies user input text into predefined intent categories using TF-IDF feature extraction and Support Vector Machine (SVM) classification.

**How it works**:
1. Preprocesses input text (lowercasing, cleaning)
2. Applies TF-IDF vectorization to extract numerical features
3. Uses trained SVM model to predict intent class
4. Returns predicted intent with confidence score

**Use cases**: Chatbots, customer service automation, query routing

---

### 3. Score Prediction Endpoints  

#### **GET /api/v1/predict_score**
**Purpose**: Predicts binary or continuous scores from structured feature data using either Logistic Regression or XGBoost algorithms.

**How it works**:
1. Accepts feature dictionary and model type selection
2. Standardizes features using pre-trained scaler
3. Applies selected algorithm (LogisticRegression or XGBoost)
4. Returns predicted score, probability, and model information

**Use cases**: Risk assessment, quality scoring, recommendation systems, fraud detection

---

### 4. RAG (Retrieval-Augmented Generation) Endpoints

#### **POST /api/v1/rag_answer**
**Purpose**: Provides intelligent question-answering by combining document retrieval with large language model generation.

**How it works**:
1. Embeds user query using Sentence Transformers model
2. Searches vector database (ChromaDB) for most relevant document chunks
3. Retrieves top-k similar documents based on cosine similarity
4. Constructs context prompt with retrieved documents
5. Streams generated answer from LLM (Ollama or OpenAI) in real-time
6. Returns structured streaming response with source attribution

**Supported LLM providers**: Ollama (local), OpenAI (cloud)
**Use cases**: Document Q&A, knowledge base search, research assistance

---

### 5. Model Training Endpoints

#### **POST /api/v1/train** - Start Training Job
**Purpose**: Initiates asynchronous model training for intent classification or score prediction models.

**How it works**:
1. Validates training request and data path
2. Creates unique job ID and initializes status tracking  
3. Executes training in background using FastAPI BackgroundTasks
4. Performs data loading, preprocessing, and model training
5. Applies hyperparameter tuning if enabled (GridSearchCV)
6. Evaluates model performance on validation set
7. Saves trained model and metadata to model registry

#### **GET /api/v1/train/{job_id}/status** - Get Training Status
Monitors training job progress with real-time status updates, metrics, and error reporting.

---

## Technical Implementation Details

### Intent Classification Pipeline
- **Algorithm**: TF-IDF + SVM with RBF kernel
- **Features**: Text vectorization with n-gram support (1-2 grams)
- **Preprocessing**: Lowercasing, stop word removal, feature selection
- **Hyperparameters**: Tunable C parameter, kernel selection, feature limits

### Score Prediction Pipeline  
- **Algorithms**: Logistic Regression with L1/L2 regularization, XGBoost ensemble
- **Features**: Standardized numerical features, categorical encoding
- **Preprocessing**: StandardScaler normalization, one-hot encoding
- **Hyperparameters**: Regularization strength, tree parameters, learning rates

### RAG Pipeline
- **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2)
- **Vector Store**: ChromaDB with persistence
- **Text Splitting**: Recursive character splitting for document chunks
- **Retrieval**: Similarity search with configurable context limits
- **Generation**: Streaming responses from local/cloud LLMs

### Model Registry & Persistence
- Centralized model storage with versioning support
- Automatic metadata tracking (metrics, hyperparameters, timestamps)
- SQLite-based model registry with binary model serialization
- Model loading with fallback and error handling

### Monitoring & Logging
- Structured JSON logging for ML operations
- Health checks for all service dependencies  
- Training job status tracking with progress indicators
- Error handling and graceful degradation

## Data Flow Architecture

1. **Request Processing**: FastAPI receives and validates requests
2. **Model Loading**: Dynamic model loading from registry based on request
3. **Feature Engineering**: Data preprocessing and feature extraction  
4. **ML Inference**: Model prediction with confidence/probability scores
5. **Response Formatting**: Structured JSON responses with metadata
6. **Error Handling**: Comprehensive exception handling with user-friendly messages

This service provides a production-ready ML platform supporting both real-time inference and model training workflows, with comprehensive monitoring and scalability features for enterprise deployment.