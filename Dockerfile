# Production Multi-Stage Dockerfile for TrustFusion Serving Engine & Interactive Dashboard
# Fully compatible with Hugging Face Spaces (Docker SDK), Render, Railway, and Cloud Run

FROM python:3.12-slim

# Prevent Python from writing .pyc and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

# Install minimal OS dependencies needed for CatBoost / XGBoost / compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch and PyTorch Geometric to keep image size small (~800MB vs 6GB+)
COPY requirements.docker.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.docker.txt

# Create non-root user required by Hugging Face Spaces (UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /app

# Copy backend application, pipelines, explainability suite, and serving models
COPY --chown=user:user backend/ ./backend/
COPY --chown=user:user pipeline/ ./pipeline/
COPY --chown=user:user xai/ ./xai/
COPY --chown=user:user data/processed/ ./data/processed/
COPY --chown=user:user dashboard.py ./

# Expose Hugging Face Space default port
EXPOSE 7860

# Healthcheck to ensure FastAPI serving engine is operational
HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Start FastAPI serving engine with Uvicorn (listening on 0.0.0.0 and dynamic $PORT)
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
