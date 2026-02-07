# EPUB Reader with local Kokoro worker
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    espeak-ng \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for user data
RUN mkdir -p static/uploads static/audio static/state static/library/covers static/models

# Prefer ONNX Kokoro backend for deterministic local worker behavior
ENV KOKORO_BACKEND=onnx

# Expose port
EXPOSE 5001

# Run the app
CMD ["python", "app_multithreaded.py"]
