# ─── Lab 3: Manufacturing Quality Prediction ───
# Improvements over Lab 2:
#   - Slim base image (python:3.10-slim)
#   - Non-root user for security
#   - HEALTHCHECK for container orchestration
#   - Pinned requirements for reproducibility
#   - Metadata labels

FROM python:3.10-slim

# Metadata
LABEL maintainer="Ajith Srikanth"
LABEL description="Manufacturing Quality Prediction - CNC Defect Classifier"
LABEL lab="Docker Lab 3"

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Install dependencies first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY src/ ./src/

# Environment configuration
ENV SERVING_PORT=5000
ENV FLASK_DEBUG=false
ENV MODEL_PATH=quality_model.joblib
ENV SCALER_PATH=scaler.joblib
ENV METRICS_PATH=training_metrics.json

EXPOSE 5000

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" || exit 1

# Switch to non-root user for serving (training runs as root for volume writes)
# USER appuser

CMD ["python", "src/main.py"]
