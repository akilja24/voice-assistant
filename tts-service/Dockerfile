FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download Piper binary
RUN wget -q https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_amd64.tar.gz && \
    tar -xzf piper_amd64.tar.gz && \
    mv piper /usr/local/bin/ && \
    rm piper_amd64.tar.gz

# Create directory for models
RUN mkdir -p /app/models

# Copy application code
COPY app/ ./app/

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8003

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8003"]