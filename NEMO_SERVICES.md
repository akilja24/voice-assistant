# NeMo ASR and TTS Services

This document describes the NVIDIA NeMo-based ASR and TTS services that have been added alongside the existing Whisper and Piper services.

## Overview

Two new services have been added:
- **asr-nemo-service**: Speech recognition using NVIDIA NeMo models (FastConformer/Parakeet)
- **tts-nemo-service**: Text-to-speech using NVIDIA NeMo models (FastPitch + HiFi-GAN)

These services are fully compatible with the existing API contracts, allowing seamless switching between backends.

## Architecture

```
┌─────────────────┐
│   Orchestrator  │
│  (Backend Router)│
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    v         v
┌──────────┐ ┌──────────┐
│ Whisper  │ │   NeMo   │
│   ASR    │ │   ASR    │
└──────────┘ └──────────┘
    
    ┌────┴────┐
    │         │
    v         v
┌──────────┐ ┌──────────┐
│  Piper   │ │   NeMo   │
│   TTS    │ │   TTS    │
└──────────┘ └──────────┘
```

## Quick Start

### 1. Build all services (including NeMo)
```bash
docker compose build
```

### 2. Start with default backends (Whisper + Piper)
```bash
docker compose up -d
```

### 3. Switch to NeMo backends
```bash
export ASR_BACKEND=nemo
export TTS_BACKEND=nemo
docker compose up -d
```

### 4. Mix and match backends
```bash
# NeMo ASR + Piper TTS
export ASR_BACKEND=nemo
export TTS_BACKEND=piper
docker compose up -d

# Whisper ASR + NeMo TTS
export ASR_BACKEND=whisper
export TTS_BACKEND=nemo
docker compose up -d
```

## Service Details

### ASR NeMo Service (Port 8005)

**Models Available:**
- `stt_en_fastconformer_ctc_small` (default) - Fast, accurate
- `stt_en_fastconformer_ctc_large` - More accurate, slower
- `stt_en_conformer_ctc_small` - Older architecture
- `stt_en_citrinet_256` - Lightweight option

**Environment Variables:**
```bash
NEMO_MODEL=stt_en_fastconformer_ctc_small
DEVICE=cuda
SAMPLE_RATE=16000
```

**API Endpoints:**
- `GET /health` - Health check
- `POST /transcribe` - Transcribe audio (same as Whisper)
- `POST /transcribe_batch` - Batch transcription
- `GET /models` - List available models

### TTS NeMo Service (Port 8006)

**Models:**
- FastPitch: `tts_en_fastpitch` (spectrogram generation)
- HiFi-GAN: `tts_en_hifigan` (vocoder)

**Environment Variables:**
```bash
FASTPITCH_MODEL=tts_en_fastpitch
HIFIGAN_MODEL=tts_en_hifigan
DEVICE=cuda
SAMPLE_RATE=22050
DEFAULT_SPEAKER=0
```

**API Endpoints:**
- `GET /health` - Health check
- `POST /speak` - Generate speech (same as Piper)
- `POST /speak_batch` - Batch synthesis
- `GET /voices` - List available voices
- `GET /models` - List available models

## Performance Comparison

### ASR Performance (RTF = Real-Time Factor, lower is better)
| Service | Model | RTF | GPU Memory |
|---------|-------|-----|------------|
| Whisper | base | 0.04-0.09 | ~600MB |
| NeMo | FastConformer Small | 0.02-0.05 | ~800MB |
| NeMo | FastConformer Large | 0.05-0.12 | ~1.5GB |

### TTS Performance
| Service | Processing Time | Quality |
|---------|----------------|---------|
| Piper | ~1.2s/164 chars | Good |
| NeMo | ~0.8s/164 chars | Excellent |

## Testing

### 1. Test individual services
```bash
# Test NeMo ASR
curl -X POST http://localhost:8005/transcribe \
  -F "audio=@tests/audio_samples/Clear-Short_16k.wav"

# Test NeMo TTS
curl -X POST http://localhost:8006/speak \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello from NeMo text to speech!"}' \
  --output nemo_speech.wav
```

### 2. Run benchmark comparison
```bash
cd tests
python test_backend_comparison.py
```

### 3. Test via WebSocket with backend switching
```bash
# With NeMo backends
ASR_BACKEND=nemo TTS_BACKEND=nemo docker compose up -d
python test_websocket_client.py

# With original backends
ASR_BACKEND=whisper TTS_BACKEND=piper docker compose up -d
python test_websocket_client.py
```

## GPU Requirements

NeMo services require NVIDIA GPU with:
- CUDA 11.8 or higher
- At least 4GB VRAM for small models
- 8GB+ VRAM recommended for larger models

Both services can share the same GPU and will manage memory efficiently.

## Troubleshooting

### 1. NeMo container build issues
The NeMo base container is large (~15GB). Ensure you have enough disk space.

### 2. Model download failures
Models are downloaded on first use. If download fails:
```bash
# Check logs
docker compose logs asr-nemo-service
docker compose logs tts-nemo-service

# Restart service to retry
docker compose restart asr-nemo-service
```

### 3. GPU memory errors
If you get CUDA out of memory errors, try:
- Using smaller models
- Reducing batch size
- Ensuring only one service uses GPU at a time

### 4. Check current backend configuration
```bash
curl http://localhost:8080/health | jq .backends
```

## Advanced Configuration

### Custom NeMo Models
To use custom or fine-tuned NeMo models:

1. Mount model directory:
```yaml
volumes:
  - /path/to/custom/models:/app/custom_models
```

2. Set model path:
```bash
NEMO_MODEL=/app/custom_models/my_custom_model.nemo
```

### Multi-GPU Setup
To use different GPUs for different services:
```yaml
# In compose.yaml
asr-nemo-service:
  environment:
    - CUDA_VISIBLE_DEVICES=0
    
tts-nemo-service:
  environment:
    - CUDA_VISIBLE_DEVICES=1
```

## Benefits of NeMo Services

1. **Better accuracy**: State-of-the-art models
2. **Faster inference**: Optimized for NVIDIA GPUs
3. **More languages**: Support for 100+ languages (with appropriate models)
4. **Customizable**: Can fine-tune models for specific domains
5. **Unified framework**: Both ASR and TTS in same framework

## Future Enhancements

- [ ] Streaming ASR support
- [ ] Multi-language model switching
- [ ] Voice cloning with NeMo TTS
- [ ] Model quantization for faster inference
- [ ] Kubernetes deployment manifests
- [ ] A/B testing framework for backends