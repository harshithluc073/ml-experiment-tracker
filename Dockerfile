# Dockerfile for ML Experiment Tracker
# Multi-stage build for optimized image size

# Stage 1: Build stage
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime stage
FROM python:3.10-slim

WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY src/ ./src/
COPY setup.py .
COPY README.md .

# Install the package
RUN pip install --no-cache-dir -e .

# Create directories for experiments and artifacts
RUN mkdir -p /workspace/experiments /workspace/artifacts

# Set working directory
WORKDIR /workspace

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV EXPERIMENT_TRACKER_DIR=/workspace/experiments

# Expose ports (for MLflow UI, etc.)
EXPOSE 5000

# Default command
CMD ["python"]