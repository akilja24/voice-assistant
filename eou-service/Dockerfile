FROM python:3.11-slim

# Install system dependencies for audio processing and Python build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    portaudio19-dev \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip and install setuptools first
RUN pip install --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Note: Models will be downloaded on first run to avoid build issues

# Pre-download Silero VAD model during build
RUN python -c "import torch; torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False)" || true

# Copy application code
COPY app/ app/

# Create models directory
RUN mkdir -p /app/models

EXPOSE 8004

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8004"]