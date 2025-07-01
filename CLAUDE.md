# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# voice-assistant – Modular AI Voice Backend

## Overview

This project implements a modular voice assistant backend with four services:

| Component         | Role                            | Runtime                      |
|------------------|----------------------------------|------------------------------|
| orchestrator      | Main API gateway / coordinator   | Docker (FastAPI)             |
| whisper-service   | Audio → Text (ASR)               | Docker (CUDA 12.2)           |
| Ollama (external) | Text → Response (LLM)            | Native host service          |
| tts-service       | Text → Audio (TTS)               | Docker (CPU only)            |

All services are REST-based. The orchestrator is the single public entry point that manages all downstream interactions.

## 🔗 Component Repositories

- **faster-whisper (ASR)**  
  https://github.com/guillaumekln/faster-whisper

- **Piper TTS (local speech)**  
  https://github.com/rhasspy/piper

- **Ollama (LLM runtime)**  
  https://github.com/ollama/ollama  
  Docs: https://ollama.com

## 🔁 Architecture and Data Flow

```
[Client]
   │
   ▼
POST /interact (audio) ───► orchestrator
                             ├──► whisper-service (transcribe)
                             ├──► Ollama (generate)
                             └──► tts-service (speak)
                                           │
                                       return audio
```

### Interaction Flow

1. Client uploads an audio file to `POST /interact` on `orchestrator`.
2. Orchestrator:
   - Sends audio to `whisper-service`
   - Sends transcription to `Ollama` (running at `http://host.docker.internal:11434`)
   - Sends generated text to `tts-service`
   - Returns synthesized audio to the client

## 📂 Project Structure

```
voice-assistant/
├── compose.yaml
├── CLAUDE.md
├── .gitignore
├── .pre-commit-config.yaml
├── .github/
│   └── workflows/ci.yml
├── orchestrator/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/main.py
├── whisper-service/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/main.py
├── tts-service/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/main.py
```

## ⚙️ Environment Variables

| Variable         | Used by         | Description                              |
|------------------|------------------|------------------------------------------|
| WHISPER_MODEL     | whisper-service  | e.g. `tiny.en`                            |
| TTS_VOICE         | tts-service      | e.g. `en_US-amy-medium`                   |
| SHARED_SECRET     | all              | Optional API key for internal services   |
| OLLAMA_MODEL      | orchestrator     | e.g. `llama3`, `mistral`, `phi3`, etc.    |
| OLLAMA_URL        | orchestrator     | e.g. `http://host.docker.internal:11434` |

## ⚙️ Docker Compose Notes

- Each service is containerized except Ollama, which must be installed natively.


## 🧪 Local Dev & Testing

```bash
docker compose build
docker compose up -d
```

### Health Checks

```bash
curl http://localhost:8000/health       # orchestrator
curl http://localhost:8001/health       # whisper
curl http://localhost:8003/health       # tts
```


## 🧼 Code Quality & Tooling

- **Format:** Black, isort
- **Commits:** Conventional Commits (`feat:`, `fix:`, `chore:`, etc.)
- **CI/CD:** GitHub Actions via `.github/workflows/ci.yml`
- **Pre-commit hooks:** see `.pre-commit-config.yaml`

## ✅ Future Tasks

- [ ] Add streaming mode via WebSockets to orchestrator
- [ ] Optional fallback to OpenAI for LLM or TTS if local models fail
- [ ] Add CLI or minimal front-end interface