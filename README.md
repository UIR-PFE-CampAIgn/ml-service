# ML Service

A comprehensive Machine Learning service built with FastAPI that provides intent classification and score prediction capabilities.

## Features

- **Intent Classification**: TF-IDF + SVM pipeline for text intent classification
- **Score Prediction**: LogisticRegression and XGBoost pipelines for feature-based scoring
- **Model Training**: Asynchronous model training with job tracking
- **Model Registry**: S3/MinIO-based model storage and versioning
- **REST API**: FastAPI-based REST API with automatic documentation
- **Docker Support**: Containerized deployment with Docker Compose
- **CI/CD**: GitHub Actions workflows for testing and deployment

## Architecture

```
ml-service/
├─ app/
│  ├─ api/                     # FastAPI routers
│  │   ├─ v1/
│  │   │   ├─ intent.py        # GET /predict_intent
│  │   │   ├─ score.py         # GET /predict_score
│  │   │   └─ train.py         # POST /train
│  ├─ core/
│  │   ├─ config.py            # pydantic BaseSettings
│  │   ├─ model_registry.py    # S3/MinIO helpers
│  │   └─ logging.py
│  ├─ ml/
│  │   ├─ intent.py            # TF-IDF + SVM pipeline
│  │   ├─ score.py             # LogReg / XGB pipeline
│  ├─ schemas/                 # pydantic request/response models
│  └─ main.py                  # FastAPI instance
├─ tests/
│  └─ unit/
├─ Dockerfile
├─ docker-compose.yml          # local dev (MinIO)
├─ requirements.txt
├─ .github/workflows/ci.yml
└─ README.md
```

## Quick Start

### Using Docker Compose (Recommended)

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd ml-service
   ```

2. **Start services with Docker Compose**

   ```bash
   docker-compose up -d
   ```

   This will start:

   - ML Service (FastAPI) on `http://localhost:8082`
   - MinIO (S3-compatible storage) on `http://localhost:9000`

3. **Access the API documentation**
   - Swagger UI: `http://localhost:8082/docs`
   - ReDoc: `http://localhost:8082/redoc`

### Local Development

1. **Install Python dependencies**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Set environment variables**

   ```bash
   export DEBUG=true
   export LOG_LEVEL=INFO
   export MODEL_STORAGE_TYPE=local
   # No LLM configuration required
   ```

3. **Run the service**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8082
   ```

## API Usage

### Intent Classification

```bash
# Predict intent for a single text
curl -X GET "http://localhost:8082/api/v1/predict_intent?text=Hello%20how%20are%20you"
```

### Score Prediction

```bash
# Predict score using features
curl -X GET "http://localhost:8082/api/v1/predict_score" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {"feature1": 1.0, "feature2": 0.5},
    "model_type": "logistic_regression"
  }'
```

### Model Training

```bash
# Start training a new model
curl -X POST "http://localhost:8082/api/v1/train" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "intent",
    "training_data_path": "/path/to/training_data.csv",
    "hyperparameters": {"C": 1.0, "kernel": "rbf"},
    "validation_split": 0.2
  }'

# Check training status
curl -X GET "http://localhost:8082/api/v1/train/{job_id}/status"
```

## Configuration

The service is configured using environment variables and/or a `.env` file:

```env
# API Settings
DEBUG=false
HOST=0.0.0.0
PORT=8082

# Model Storage (S3/MinIO)
MODEL_STORAGE_TYPE=s3
S3_BUCKET=ml-models
S3_REGION=us-east-1
S3_ENDPOINT_URL=http://localhost:9000
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key

# (RAG removed — no LLM or vector DB settings required)

# Logging
LOG_LEVEL=INFO
```

## Data Formats

### Intent Classification Data

Training data should be a CSV file with `text` and `intent` columns:

```csv
text,intent
"Hello how are you",greeting
"What's the weather like",weather_query
"Book a flight to Paris",booking_request
```

### Score Prediction Data

Training data should be a CSV file with feature columns and a target column:

```csv
feature1,feature2,feature3,score
1.0,0.5,2.1,0.8
0.3,1.2,1.7,0.6
2.1,0.8,0.9,0.9
```

## Model Management

### Listing Models

```bash
curl -X GET "http://localhost:8082/api/v1/models"
```

### Model Information

```bash
curl -X GET "http://localhost:8082/api/v1/models/intent?version=latest"
```

## Development

### Running Tests

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests (requires running services)
pytest tests/integration/ -v

# All tests with coverage
pytest --cov=app --cov-report=html
```

### Code Quality

```bash
# Format code
black app/
isort app/

# Lint code
flake8 app/

# Security check
bandit -r app/
```

### Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

## Deployment

### Docker

```bash
# Build image
docker build -t ml-service:latest .

# Run container
docker run -p 8082:8082 ml-service:latest
```

### Kubernetes

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-service
  template:
    metadata:
      labels:
        app: ml-service
    spec:
      containers:
        - name: ml-service
          image: ml-service:latest
          ports:
            - containerPort: 8082
          env:
            - name: MODEL_STORAGE_TYPE
              value: "s3"
            - name: S3_BUCKET
              value: "ml-models"
```

## Monitoring

### Health Check

```bash
curl -X GET "http://localhost:8082/health"
```

### Metrics

The service provides basic health and status information. For production monitoring, consider integrating with:

- **Prometheus**: Add metrics collection
- **Grafana**: Visualization dashboards
- **Sentry**: Error tracking
- **DataDog**: APM and logs

## Performance

### Optimization Tips

1. **Model Caching**: Models are cached in memory after first load
2. **Batch Processing**: Use batch endpoints for multiple predictions
3. **Async Operations**: Training and heavy operations run asynchronously
4. **Resource Limits**: Configure appropriate memory and CPU limits

### Scaling

- **Horizontal**: Deploy multiple service instances behind a load balancer
- ## **Vertical**: Increase CPU and memory allocations

## Troubleshooting

### Common Issues

1. **Model Loading Errors**

   - Check S3/MinIO connectivity
   - Verify bucket permissions
   - Ensure model files exist

2. --

3. **Memory Issues**
   - Monitor model memory usage
   - Consider model quantization
   - Implement model unloading

### Logs

Logs are available in:

- Console output (development)
- `./logs/ml-service.log` (file logging)
- Container logs: `docker logs <container_id>`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run quality checks
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions and support:

- Create an issue in the repository
- Check the documentation
- Review the API docs at `/docs`
