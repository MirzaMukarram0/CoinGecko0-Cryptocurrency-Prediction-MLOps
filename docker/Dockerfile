# Multi-stage build for efficient Docker image
FROM python:3.9-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VERSION
ARG VCS_REF

# Add metadata
LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="cryptocurrency-prediction-mlops" \
      org.label-schema.description="MLOps pipeline for cryptocurrency price prediction" \
      org.label-schema.version=$VERSION \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/your-username/CoinGecko0-Cryptocurrency-Prediction-MLOps" \
      org.label-schema.schema-version="1.0"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.9-slim as production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r mlops && useradd -r -g mlops mlops

# Set working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY airflow/ ./airflow/
COPY data/ ./data/
COPY models/ ./models/
COPY predictions/ ./predictions/
COPY tests/ ./tests/
COPY requirements.txt ./

# Create necessary directories
RUN mkdir -p logs reports && \
    chown -R mlops:mlops /app

# Switch to non-root user
USER mlops

# Set Python path
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000 8080

# Default command
CMD ["python", "src/api/app.py"]