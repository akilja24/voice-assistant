FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    libgomp1 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directory for models with proper permissions
RUN mkdir -p /app/models && chmod 755 /app/models

# Download and install Piper binary
RUN cd /tmp && \
    wget https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz && \
    tar -xzvf piper_linux_x86_64.tar.gz && \
    cd piper && \
    chmod +x piper && \
    mv piper /usr/local/bin/ && \
    mv *.so* /usr/local/lib/ && \
    mv espeak-ng-data /usr/share/ && \
    ldconfig && \
    cd /tmp && \
    rm -rf piper piper_linux_x86_64.tar.gz && \
    # Test that piper is accessible
    /usr/local/bin/piper --help

# Copy application code
COPY app/ ./app/

# Create non-root user and ensure proper permissions
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app && \
    chmod -R 755 /app/models

# Pre-download the default voice model as root to avoid permission issues
RUN mkdir -p /app/models/en_US/amy/medium && \
    wget -q -O /app/models/en_US-amy-medium.onnx \
    "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx" && \
    wget -q -O /app/models/en_US-amy-medium.onnx.json \
    "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx.json" && \
    chown -R appuser:appuser /app/models

USER appuser

# Expose port
EXPOSE 8003

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8003"]