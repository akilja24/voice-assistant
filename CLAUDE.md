# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# voice-assistant – Real-time AI Voice Assistant Backend

## Overview

This project implements a production-ready voice assistant backend with GPU-accelerated speech recognition, LLM integration, and fast text-to-speech synthesis. The system supports both REST and WebSocket interfaces for real-time audio streaming.

| Component         | Role                            | Runtime                      | Performance                    |
|------------------|----------------------------------|------------------------------|--------------------------------|
| orchestrator      | WebSocket/REST API gateway       | Docker (FastAPI)             | Handles streaming & coordination|
| whisper-service   | Audio → Text (ASR)               | Docker (CUDA 12.4 + cuDNN 9) | RTF: 0.04-0.09 (11-25x realtime)|
| asr-nemo-service  | Audio → Text (ASR) - Alternative | Docker (CUDA 12.1 + PyTorch) | RTF: 0.002-0.005 (200-500x realtime)|
| Ollama (external) | Text → Response (LLM)            | Native host service          | Configurable models            |
| tts-service       | Text → Audio (TTS)               | Docker (CPU-optimized)       | ~1.24s for 164 chars          |
| tts-nemo-service  | Text → Audio (TTS) - Alternative | Docker (CUDA 12.1 + PyTorch) | 1.1-1.7x faster than Piper    |
| eou-service       | End-of-utterance detection       | Docker (CPU)                 | Real-time silence & semantic detection|

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

- **NVIDIA NeMo (Alternative ASR/TTS)**  
  https://github.com/NVIDIA/NeMo  
  Framework for building state-of-the-art conversational AI models  
  Using QuartzNet (ASR) and Tacotron2 + HiFi-GAN (TTS)

## 🔁 Architecture and Data Flow

### WebSocket Streaming (Primary Interface)
```
[Client] 
   │
   ├─── ws://localhost:8080/ws ───► orchestrator
   │    (stream audio chunks)         │
   │                                  ├──► ASR Backend (based on ASR_BACKEND env var):
   │                                  │    ├─► whisper-service (default)
   │                                  │    └─► asr-nemo-service (if ASR_BACKEND=nemo)
   │                                  │
   │                                  ├──► Ollama (generate response)
   │                                  │
   │                                  └──► TTS Backend (based on TTS_BACKEND env var):
   │                                       ├─► tts-service (default)
   │                                       └─► tts-nemo-service (if TTS_BACKEND=nemo)
   │                                              │
   └─────────── audio stream ◄────────────────────┘
```

### WebSocket Auto-Detection Mode (New)
```
[Client] 
   │
   ├─── ws://localhost:8080/ws/auto ───► orchestrator
   │    (stream audio chunks)              ├──► eou-service (detect end-of-utterance)
   │                                       │
   │                                       ├──► ASR Backend (on EOU trigger):
   │                                       │    ├─► whisper-service (default)
   │                                       │    └─► asr-nemo-service (if ASR_BACKEND=nemo)
   │                                       │
   │                                       ├──► Ollama (generate response)
   │                                       │
   │                                       └──► TTS Backend:
   │                                            ├─► tts-service (default)
   │                                            └─► tts-nemo-service (if TTS_BACKEND=nemo)
   │                                                   │
   └─────────── audio stream ◄────────────────────────┘
```

### REST API (Alternative)
```
POST /interact (audio file) ───► orchestrator ───► [same pipeline] ───► audio response
```

## 📂 Project Structure

```
voice-assistant/
├── compose.yaml                    # Docker Compose configuration
├── CLAUDE.md                       # This file - project documentation
├── NEMO_SERVICES.md               # NeMo services documentation
├── NEMO_BENCHMARK_REPORT.md       # NeMo performance benchmarks
├── orchestrator/
│   ├── Dockerfile                 # Python 3.12 slim base
│   ├── requirements.txt           # FastAPI, httpx, aiofiles, numpy
│   └── app/
│       └── main.py                # WebSocket & REST endpoints
├── whisper-service/
│   ├── Dockerfile                 # NVIDIA CUDA 12.4.1 + cuDNN 9
│   ├── requirements.txt           # faster-whisper, nvidia-ml-py
│   └── app/
│       └── main.py                # GPU-accelerated transcription
├── asr-nemo-service/              # NeMo ASR alternative
│   ├── Dockerfile                 # PyTorch 2.1.0 + CUDA 12.1
│   ├── requirements.txt           # nemo_toolkit[asr]==1.22.0
│   └── app/
│       └── main.py                # NeMo ASR with QuartzNet
├── tts-service/
│   ├── Dockerfile                 # Python 3.12 + Piper binary
│   ├── requirements.txt           # FastAPI, aiofiles
│   └── app/
│       └── main.py                # Piper TTS integration
├── tts-nemo-service/              # NeMo TTS alternative
│   ├── Dockerfile                 # PyTorch 2.1.0 + CUDA 12.1
│   ├── requirements.txt           # nemo_toolkit[tts]==1.22.0
│   └── app/
│       └── main.py                # Tacotron2 + HiFi-GAN
├── eou-service/
│   ├── Dockerfile                 # Python 3.12 + Pipecat
│   ├── requirements.txt           # Pipecat, punctuation model
│   └── app/
│       └── main.py                # End-of-utterance detection
├── tests/
│   ├── test_websocket_client.py  # WebSocket streaming test
│   ├── test_websocket_auto.py    # Auto EOU detection test
│   ├── test_whisper_direct.py    # Direct whisper API test
│   ├── test_audio_validation.py  # Audio format validation
│   ├── test_eou_direct.py        # EOU service direct test
│   ├── test_interruption.py      # Interruption handling test
│   ├── test_auto_eou.py          # Auto EOU with interruption
│   ├── test_backend_comparison.py # Whisper vs NeMo benchmark
│   ├── test_backend_comparison_simple.py # Simple benchmark
│   └── audio_samples/
│       ├── Clear-Short_16k.wav   # "I want to learn about quantum computing..."
│       ├── Clear-Medium_16k.wav  # Weather query sample
│       └── Noisy-Short_16k.wav   # Background noise test
├── test_nemo_setup.py             # NeMo library verification
├── test_nemo_import_fix.py        # NeMo import workaround test
├── test_benchmark_comparison.py   # Comprehensive benchmark
└── benchmark_results.json         # Benchmark results output
```

## 🚀 Quick Start

### Prerequisites
- Docker with NVIDIA Container Toolkit
- NVIDIA GPU with CUDA support (tested on Tesla T4)
- Ollama installed on host (optional, uses fallback if not available)

### Backend Selection

The orchestrator can use different backends for ASR and TTS services:

```bash
# Use default backends (Whisper + Piper)
docker compose up -d

# Use NeMo for both ASR and TTS
export ASR_BACKEND=nemo
export TTS_BACKEND=nemo
docker compose up -d

# Mix backends (e.g., NeMo ASR + Piper TTS)
export ASR_BACKEND=nemo
export TTS_BACKEND=piper
docker compose up -d

# Or Whisper ASR + NeMo TTS
export ASR_BACKEND=whisper
export TTS_BACKEND=nemo
docker compose up -d
```

### Docker Commands

```bash
# Build all services
docker compose build

# Start all services with default backends
docker compose up -d

# Start with specific backends
ASR_BACKEND=nemo TTS_BACKEND=nemo docker compose up -d

# View logs
docker compose logs -f
docker compose logs orchestrator -f
docker compose logs whisper-service -f
docker compose logs tts-service -f

# Check service health
curl http://localhost:8080/health   # orchestrator (WebSocket server)
curl http://localhost:8001/health   # whisper-service  
curl http://localhost:8003/health   # tts-service
curl http://localhost:8004/health   # eou-service
curl http://localhost:8005/health   # asr-nemo-service (if running)
curl http://localhost:8006/health   # tts-nemo-service (if running)

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
- **Manual Mode**: `ws://localhost:8080/ws` (requires explicit end_stream message)
- **Auto-Detection Mode**: `ws://localhost:8080/ws/auto` (automatic end-of-utterance detection)
- **Network**: `ws://<your-ip>:8080/ws` or `ws://<your-ip>:8080/ws/auto`
- **AWS/Public**: `ws://<public-ip>:8080/ws` or `ws://<public-ip>:8080/ws/auto` (configure security group for port 8080)

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

### Backend Selection Variables

| Variable               | Service         | Default                            | Description                        |
|-----------------------|------------------|------------------------------------|------------------------------------|
| ASR_BACKEND           | orchestrator    | whisper                            | ASR backend: "whisper" or "nemo"  |
| TTS_BACKEND           | orchestrator    | piper                              | TTS backend: "piper" or "nemo"    |

### Environment Variables
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
| **EOU Service**       |                 |                                    |                                    |
| VAD_MODEL             | eou-service     | silero_vad                         | Voice activity detection model     |
| VAD_SILENCE_THRESHOLD_MS | eou-service  | 400                                | Silence duration to trigger evaluation (ms) |
| SEMANTIC_MODEL        | eou-service     | pipecat_smart_turn                 | Semantic completion model          |
| EOU_PROBABILITY_THRESHOLD | eou-service | 0.7                                | Probability threshold for EOU      |
| MAX_SILENCE_TIMEOUT_MS | eou-service    | 2000                               | Maximum silence before forced EOU  |
| **Shared**            |                 |                                    |                                    |
| SHARED_SECRET         | all services    | (empty)                            | Optional API authentication        |

### Complete Environment Variables Reference

#### Orchestrator Service
| Variable               | Default                            | Description                        |
|-----------------------|------------------------------------|------------------------------------|  
| ASR_BACKEND           | whisper                            | ASR backend selection              |
| TTS_BACKEND           | piper                              | TTS backend selection              |
| WHISPER_SERVICE_URL   | http://whisper-service:8001        | Whisper service endpoint           |
| TTS_SERVICE_URL       | http://tts-service:8003            | Piper TTS service endpoint         |
| EOU_SERVICE_URL       | http://eou-service:8004            | EOU detection service endpoint     |
| ASR_NEMO_SERVICE_URL  | http://asr-nemo-service:8005       | NeMo ASR service endpoint          |
| TTS_NEMO_SERVICE_URL  | http://tts-nemo-service:8006       | NeMo TTS service endpoint          |
| OLLAMA_URL            | http://host.docker.internal:11434  | Ollama LLM API endpoint            |
| OLLAMA_MODEL          | llama3                             | LLM model name                     |
| OLLAMA_SYSTEM_PROMPT  | (see default below)                | System prompt for responses        |
| OLLAMA_MAX_TOKENS     | 300                                | Maximum response tokens            |
| SHARED_SECRET         | (empty)                            | API authentication token           |

**Default OLLAMA_SYSTEM_PROMPT**: "You are a helpful voice assistant. Provide clear, concise responses that are natural for spoken conversation."

#### Whisper Service (ASR)
| Variable                    | Default      | Description                               |
|----------------------------|--------------|-------------------------------------------|
| WHISPER_MODEL              | base         | Model: tiny/base/small/medium/large       |
| DEVICE                     | cuda         | Always cuda (GPU required)                |
| DEVICE_INDEX               | 0            | GPU device index                          |
| COMPUTE_TYPE               | float16      | GPU compute precision                     |
| BEAM_SIZE                  | 5            | Beam search width                         |
| BEST_OF                    | 5            | Number of candidates to consider          |
| TEMPERATURE                | 0.0          | Sampling temperature (0=greedy)           |
| VAD_FILTER                 | true         | Voice activity detection                  |
| VAD_THRESHOLD              | 0.5          | VAD sensitivity (0-1)                     |
| MIN_SILENCE_DURATION_MS    | 500          | Minimum silence for segmentation          |
| COMPRESSION_RATIO_THRESHOLD| 2.4          | Max compression ratio allowed             |
| LOG_PROB_THRESHOLD         | -1.0         | Average log probability threshold         |
| NO_SPEECH_THRESHOLD        | 0.6          | No speech probability threshold           |
| MODEL_CACHE_DIR            | /app/models  | Model storage path                        |
| SHARED_SECRET              | (empty)      | API authentication                        |

#### Piper TTS Service
| Variable               | Default                 | Description                        |
|-----------------------|-------------------------|------------------------------------|  
| TTS_ENGINE            | piper                   | TTS engine (only piper supported)  |
| TTS_VOICE             | en_US-amy-medium        | Piper voice model                  |
| PIPER_MODEL_PATH      | /app/models             | Voice model storage                |
| SHARED_SECRET         | (empty)                 | API authentication                 |

#### NeMo ASR Service
| Variable               | Default                    | Description                        |
|-----------------------|----------------------------|------------------------------------|  
| NEMO_MODEL            | stt_en_quartznet15x5       | NeMo ASR model                     |
| DEVICE                | cuda                       | Device (cuda/cpu)                  |
| MODEL_CACHE_DIR       | /app/models                | Model storage path                 |
| SAMPLE_RATE           | 16000                      | Audio sample rate                  |
| BATCH_SIZE            | 1                          | Batch size for inference           |
| MAX_RETRIES           | 3                          | Model download retries             |
| TORCH_HOME            | /app/models/torch          | PyTorch cache directory            |
| HF_HOME               | /app/models/huggingface    | HuggingFace cache directory        |
| SHARED_SECRET         | (empty)                    | API authentication                 |

#### NeMo TTS Service  
| Variable               | Default                    | Description                        |
|-----------------------|----------------------------|------------------------------------|  
| TACOTRON_MODEL        | tts_en_tacotron2           | Tacotron2 model                    |
| VOCODER_MODEL         | tts_en_hifigan             | Vocoder model                      |
| DEVICE                | cuda                       | Device (cuda/cpu)                  |
| MODEL_CACHE_DIR       | /app/models                | Model storage path                 |
| SAMPLE_RATE           | 22050                      | Audio sample rate                  |
| DEFAULT_SPEAKER       | 0                          | Default speaker ID                 |
| MAX_RETRIES           | 3                          | Model download retries             |
| TORCH_HOME            | /app/models/torch          | PyTorch cache directory            |
| HF_HOME               | /app/models/huggingface    | HuggingFace cache directory        |
| SHARED_SECRET         | (empty)                    | API authentication                 |

#### EOU Service
| Variable                    | Default                       | Description                        |
|----------------------------|-------------------------------|------------------------------------|  
| VAD_MODEL                  | silero_vad                    | Voice activity detection model     |
| VAD_SILENCE_THRESHOLD_MS   | 400                           | Silence before EOU evaluation (ms) |
| VAD_THRESHOLD              | 0.5                           | VAD sensitivity (0-1)              |
| SEMANTIC_MODEL             | pipecat_smart_turn            | Semantic completion model          |
| PUNCTUATION_MODEL          | deepmultilingualpunctuation   | Punctuation detection model        |
| EOU_PROBABILITY_THRESHOLD  | 0.7                           | EOU probability threshold          |
| MAX_SILENCE_TIMEOUT_MS     | 2000                          | Max silence before forced EOU      |
| SAMPLE_RATE                | 16000                         | Audio sample rate                  |
| SHARED_SECRET              | (empty)                       | API authentication                 |

### Port Mappings
- `8080` → orchestrator (WebSocket/REST API)
- `8001` → whisper-service (internal)
- `8003` → tts-service (internal)
- `8004` → eou-service (internal)
- `8005` → asr-nemo-service (internal)
- `8006` → tts-nemo-service (internal)

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

### NeMo ASR Performance
- **Hardware**: NVIDIA Tesla T4 (15GB VRAM)
- **Memory Usage**: ~80MB GPU memory (significantly lower than Whisper)
- **Real-Time Factor**: 0.002-0.005 (200-500x faster than real-time)
- **Model**: QuartzNet15x5 (18.9M parameters)
- **Performance**: 5-10x faster than Whisper

### NeMo TTS Performance
- **Hardware**: NVIDIA Tesla T4 (15GB VRAM)
- **Memory Usage**: ~480MB GPU memory
- **Performance**: 1.1-1.7x faster than Piper
- **Models**: Tacotron2 + HiFi-GAN
- **Audio Output**: 22050 Hz, 16-bit PCM WAV
- **Quality**: Higher quality than Piper with more natural prosody

## 🧪 Testing

### Test Audio Samples
The `tests/audio_samples/` directory contains:
- **Clear-Short_16k.wav**: "I want to learn more about quantum computing. Can you explain to me the basics?"
- **Clear-Medium_16k.wav**: Weather-related query
- **Noisy-Short_16k.wav**: Same as Clear-Short but with background noise

### Test Files Overview

#### Core WebSocket Tests
- **test_websocket_client.py** - Tests manual WebSocket streaming (`/ws` endpoint)
- **test_websocket_auto.py** - Tests automatic EOU detection (`/ws/auto` endpoint)
- **test_interruption.py** - Tests interruption handling during audio playback
- **test_auto_eou.py** - Tests auto-EOU with interruption scenarios

#### Service-Specific Tests
- **test_whisper_direct.py** - Direct Whisper ASR service testing
- **test_eou_direct.py** - Direct EOU service testing (REST and WebSocket)
- **test_audio_validation.py** - Audio file format validation

#### Performance Benchmarks
- **test_backend_comparison.py** - Compares Whisper/Piper vs NeMo backends
- **test_backend_comparison_simple.py** - Simple benchmark for current services
- **test_benchmark_comparison.py** - Comprehensive multi-run benchmark (root dir)

#### NeMo Setup Tests
- **test_nemo_setup.py** - Verifies NeMo library installation
- **test_nemo_import_fix.py** - Tests NeMo import workaround

### Running Tests

#### Basic Testing
```bash
# Test WebSocket streaming with all samples
cd tests
python test_websocket_client.py

# Test with specific audio file
python test_websocket_client.py audio_samples/Clear-Short_16k.wav

# Test auto-detection mode
python test_websocket_auto.py
python test_websocket_auto.py audio_samples/Clear-Short_16k.wav

# Test service directly
python test_whisper_direct.py
python test_eou_direct.py

# Test interruption handling
python test_interruption.py
python test_auto_eou.py

# Validate audio files
python test_audio_validation.py
```

#### Performance Testing
```bash
# Simple benchmark (current services)
cd tests
python test_backend_comparison_simple.py

# Full benchmark (requires NeMo services)
python test_backend_comparison.py

# Comprehensive benchmark with statistics
cd /home/ubuntu/voice-assistant
python test_benchmark_comparison.py
```

#### NeMo Testing
```bash
# Test NeMo setup
python test_nemo_setup.py

# Test with actual model loading
TEST_MODEL_LOAD=1 python test_nemo_setup.py

# Test import fix
python test_nemo_import_fix.py
```

#### Direct Service Testing
```bash
# Test ASR services directly
curl -X POST http://localhost:8001/transcribe \
  -F "audio=@tests/audio_samples/Clear-Short_16k.wav"

curl -X POST http://localhost:8005/transcribe \
  -F "audio=@tests/audio_samples/Clear-Short_16k.wav"

# Test TTS services directly  
curl -X POST http://localhost:8003/speak \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello from Piper!"}' \
  --output piper_speech.wav

curl -X POST http://localhost:8006/speak \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello from NeMo!"}' \
  --output nemo_speech.wav
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

## 🎙️ LLM Response Configuration

### Controlling Response Length

The orchestrator controls LLM response length through the Ollama API parameters:

```python
# In orchestrator/app/main.py
"options": {
    "num_predict": 300,  # Maximum tokens (adjust for shorter/longer responses)
    "temperature": 0.7,
    "top_p": 0.9
}
```

To make responses shorter (max 3 sentences), you can:

1. **Modify the system prompt** (recommended):
```python
system_prompt = "You are a helpful voice assistant. Keep responses very brief - maximum 3 sentences. Be concise and direct."
```

2. **Reduce num_predict tokens**:
```python
"num_predict": 100,  # Shorter responses
```

3. **Set via environment variable** (add to orchestrator):
```bash
export OLLAMA_MAX_TOKENS=100
export OLLAMA_SYSTEM_PROMPT="You are a helpful voice assistant. Keep responses very brief - maximum 3 sentences."
```

### Example Implementation

To implement configurable response length:

1. Add environment variables to `orchestrator/app/main.py`:
```python
OLLAMA_MAX_TOKENS = int(os.getenv("OLLAMA_MAX_TOKENS", "300"))
OLLAMA_SYSTEM_PROMPT = os.getenv("OLLAMA_SYSTEM_PROMPT", 
    "You are a helpful voice assistant. Provide clear, concise responses that are natural for spoken conversation.")
```

2. Update the Ollama API call:
```python
system_prompt = OLLAMA_SYSTEM_PROMPT
# ...
"options": {
    "num_predict": OLLAMA_MAX_TOKENS,
    "temperature": 0.7,
    "top_p": 0.9
}
```

3. Configure via Docker Compose:
```yaml
orchestrator:
  environment:
    - OLLAMA_MAX_TOKENS=100
    - OLLAMA_SYSTEM_PROMPT=You are a helpful voice assistant. Keep responses very brief - maximum 3 sentences.
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
- [x] Automatic end-of-utterance detection with Pipecat integration

## 🔄 Future Enhancements
- [ ] Multi-language support (Whisper supports 100+ languages)
- [ ] Alternative TTS voices and engines
- [ ] Audio preprocessing (noise reduction, normalization)
- [ ] Streaming TTS output
- [ ] Metrics and monitoring integration
- [ ] Kubernetes deployment manifests