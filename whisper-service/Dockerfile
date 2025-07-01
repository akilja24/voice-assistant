FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
# Set library paths for CUDA and cuDNN
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Verify cuDNN installation
RUN ldconfig -p | grep cudnn || echo "cuDNN libraries not found in ldconfig"
RUN ls -la /usr/lib/x86_64-linux-gnu/libcudnn* || echo "No cuDNN files in /usr/lib"
RUN ls -la /usr/local/cuda*/targets/x86_64-linux/lib/libcudnn* || echo "No cuDNN files in CUDA dir"

# Set working directory
WORKDIR /app

# Create model cache directory
RUN mkdir -p /app/models

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8001

# Run the application
CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]