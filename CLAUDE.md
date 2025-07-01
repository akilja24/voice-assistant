# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# voice-assistant – Real-time AI Voice Assistant Backend

## Overview

This project implements a production-ready voice assistant backend with GPU-accelerated speech recognition, LLM integration, and fast text-to-speech synthesis. The system supports both REST and WebSocket interfaces for real-time audio streaming.

| Component         | Role                            | Runtime                      | Performance                    |
|------------------|----------------------------------|------------------------------|--------------------------------|
| orchestrator      | WebSocket/REST API gateway       | Docker (FastAPI)             | Handles streaming & coordination|
| whisper-service   | Audio → Text (ASR)               | Docker (CUDA 12.4 + cuDNN 9) | RTF: 0.04-0.09 (11-25x realtime)|
| Ollama (external) | Text → Response (LLM)            | Native host service          | Configurable models            |
| tts-service       | Text → Audio (TTS)               | Docker (CPU-optimized)       | ~1.24s for 164 chars          |

## 🔗 Component Repositories

- **faster-whisper (GPU-accelerated ASR)**  
  https://github.com/SYSTRAN/faster-whisper  
  Using CTranslate2 and NVIDIA GPU acceleration

- **Piper TTS (CPU-optimized local speech)**  
  https://github.com/rhasspy/piper  
  Version: 2023.11.14-2, Voice: en_US-amy-medium

- **Ollama (LLM runtime)**  
  https://github.com/ollama/ollama  
  Docs: https://ollama.com

## 🔁 Architecture and Data Flow

### WebSocket Streaming (Primary Interface)
```
[Client] 
   │
   ├─── ws://localhost:8080/ws ───► orchestrator
   │    (stream audio chunks)         ├──► whisper-service (GPU transcribe)
   │                                  ├──► Ollama (generate response)
   │                                  └──► tts-service (synthesize speech)
   │                                           │
   └─────────── audio stream ◄────────────────┘
```

### REST API (Alternative)
```
POST /interact (audio file) ───► orchestrator ───► [same pipeline] ───► audio response
```

## 📂 Project Structure

```
voice-assistant/
├── compose.yaml              # Docker Compose configuration
├── CLAUDE.md                 # This file - project documentation
├── orchestrator/
│   ├── Dockerfile           # Python 3.12 slim base
│   ├── requirements.txt     # FastAPI, httpx, aiofiles, numpy
│   └── app/
│       └── main.py          # WebSocket & REST endpoints
├── whisper-service/
│   ├── Dockerfile           # NVIDIA CUDA 12.4.1 + cuDNN 9
│   ├── requirements.txt     # faster-whisper, nvidia-ml-py
│   └── app/
│       └── main.py          # GPU-accelerated transcription
├── tts-service/
│   ├── Dockerfile           # Python 3.12 + Piper binary
│   ├── requirements.txt     # FastAPI, aiofiles
│   └── app/
│       └── main.py          # Piper TTS integration
└── tests/
    ├── test_websocket_client.py  # WebSocket streaming test
    ├── test_whisper_direct.py     # Direct whisper API test
    ├── test_audio_validation.py   # Audio format validation
    └── audio_samples/
        ├── Clear-Short_16k.wav    # "I want to learn about quantum computing..."
        ├── Clear-Medium_16k.wav   # Weather query sample
        └── Noisy-Short_16k.wav    # Background noise test
```

## 🚀 Quick Start

### Prerequisites
- Docker with NVIDIA Container Toolkit
- NVIDIA GPU with CUDA support (tested on Tesla T4)
- Ollama installed on host (optional, uses fallback if not available)

### Docker Commands

```bash
# Build all services
docker compose build

# Start all services
docker compose up -d

# View logs
docker compose logs -f
docker compose logs orchestrator -f
docker compose logs whisper-service -f
docker compose logs tts-service -f

# Check service health
curl http://localhost:8080/health   # orchestrator (WebSocket server)
curl http://localhost:8001/health   # whisper-service  
curl http://localhost:8003/health   # tts-service

# Stop services
docker compose down

# Clean up (including volumes)
docker compose down -v

# Rebuild specific service
docker compose build whisper-service
docker compose up -d whisper-service
```

## 🌐 WebSocket Interface

### URLs
- **Local**: `ws://localhost:8080/ws`
- **Network**: `ws://<your-ip>:8080/ws`
- **AWS/Public**: `ws://<public-ip>:8080/ws` (configure security group for port 8080)

### Protocol
1. Connect to WebSocket
2. Send audio data as binary chunks (recommended: 64KB chunks)
3. Send JSON message: `{"type": "end_stream"}` when done
4. Receive:
   - JSON: `{"type": "metadata", "audio_info": {...}}`
   - Binary: Audio response chunks
   - JSON: `{"type": "audio_complete"}`

### Example Test
```bash
cd tests
python test_websocket_client.py audio_samples/Clear-Short_16k.wav
```

## ⚙️ Configuration

### Environment Variables

| Variable               | Service         | Default                            | Description                        |
|-----------------------|-----------------|------------------------------------|------------------------------------|
| **Whisper Service**   |                 |                                    |                                    |
| WHISPER_MODEL         | whisper-service | base                               | Model size: tiny/base/small/medium/large |
| DEVICE                | whisper-service | cuda                               | Always cuda (GPU required)         |
| COMPUTE_TYPE          | whisper-service | float16                            | GPU compute precision              |
| VAD_FILTER            | whisper-service | true                               | Voice activity detection           |
| MODEL_CACHE_DIR       | whisper-service | /app/models                        | Model storage path                 |
| **TTS Service**       |                 |                                    |                                    |
| TTS_ENGINE            | tts-service     | piper                              | TTS engine (only piper supported)  |
| TTS_VOICE             | tts-service     | en_US-amy-medium                   | Piper voice model                  |
| PIPER_MODEL_PATH      | tts-service     | /app/models                        | Voice model storage                |
| **Orchestrator**      |                 |                                    |                                    |
| OLLAMA_URL            | orchestrator    | http://host.docker.internal:11434  | Ollama API endpoint                |
| OLLAMA_MODEL          | orchestrator    | llama3                             | LLM model name                     |
| **Shared**            |                 |                                    |                                    |
| SHARED_SECRET         | all services    | (empty)                            | Optional API authentication        |

### Port Mappings
- `8080` → orchestrator (WebSocket/REST API)
- `8001` → whisper-service (internal)
- `8003` → tts-service (internal)

## 📊 Performance Metrics

### Whisper GPU Transcription
- **Hardware**: NVIDIA Tesla T4 (15GB VRAM)
- **Memory Usage**: ~258-600MB GPU memory
- **Real-Time Factor**: 0.04-0.09 (11-25x faster than real-time)
- **Model**: base.en (74M parameters)
- **Optimizations**: VAD enabled, float16 precision

### Piper TTS Synthesis  
- **CPU Performance**: ~1.24s for 164 characters
- **Voice Model**: en_US-amy-medium (60.3MB)
- **Audio Output**: 22050 Hz, 16-bit PCM WAV
- **Optimizations**: Sentence silence, speed control

### Full Pipeline Latency
- Audio upload → Transcription → LLM → TTS → Response: ~3-5 seconds total

## 🧪 Testing

### Test Audio Samples
The `tests/audio_samples/` directory contains:
- **Clear-Short_16k.wav**: "I want to learn more about quantum computing. Can you explain to me the basics?"
- **Clear-Medium_16k.wav**: Weather-related query
- **Noisy-Short_16k.wav**: Same as Clear-Short but with background noise

### Running Tests
```bash
# Test WebSocket streaming
cd tests
python test_websocket_client.py audio_samples/Clear-Short_16k.wav

# Test Whisper service directly
python test_whisper_direct.py

# Test all audio samples
python test_websocket_client.py
```

## 🚀 Production Deployment

### AWS EC2 Requirements
- **Instance Type**: g4dn.xlarge or better (Tesla T4 GPU)
- **Storage**: 100GB+ (for Docker images and models)
- **Security Group**: Open ports 8080 (WebSocket), 22 (SSH)

### GPU Setup
```bash
# Verify GPU
nvidia-smi

# Check CUDA version
nvidia-smi | grep "CUDA Version"

# Verify Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### Disk Space Management
```bash
# Check disk usage
df -h

# Clean up Docker
docker system prune -af --volumes
```

## 🧹 Maintenance

### View GPU Usage
```bash
# Real-time GPU monitoring
nvidia-smi -l 1

# Check whisper GPU usage
docker exec voice-assistant-whisper-service-1 nvidia-smi
```

### Update Models
```bash
# Change Whisper model size
export WHISPER_MODEL=small  # tiny, base, small, medium, large
docker compose up -d whisper-service

# Download additional Piper voices
# Add to tts-service Dockerfile or download at runtime
```

### Troubleshooting
```bash
# Check if services are running
docker compose ps

# Restart a crashed service
docker compose restart whisper-service

# Check for cuDNN issues
docker compose logs whisper-service | grep -i cudnn

# Test without Ollama (uses fallback)
# Just ensures Ollama URL is unreachable
```

## ✅ Completed Features
- [x] WebSocket streaming interface on /ws endpoint
- [x] GPU-accelerated Whisper transcription
- [x] CPU-optimized Piper TTS
- [x] Full audio pipeline integration
- [x] Comprehensive test suite
- [x] Production-ready Docker setup

## 🔄 Future Enhancements
- [ ] Multi-language support (Whisper supports 100+ languages)
- [ ] Alternative TTS voices and engines
- [ ] Audio preprocessing (noise reduction, normalization)
- [ ] Streaming TTS output
- [ ] Metrics and monitoring integration
- [ ] Kubernetes deployment manifests