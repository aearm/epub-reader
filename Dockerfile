# EPUB Reader with TTS
FROM python:3.10-slim

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

# Download TTS model at build time (so it's cached)
RUN python -c "from TTS.api import TTS; TTS('tts_models/en/ljspeech/tacotron2-DDC')"

# Copy application code
COPY . .

# Create directories for user data
RUN mkdir -p static/uploads static/audio static/state static/library/covers

# Expose port
EXPOSE 5001

# Run the app
CMD ["python", "app_multithreaded.py"]
