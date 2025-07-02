from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Fix for huggingface_hub ModelFilter import issue
try:
    import huggingface_hub
    if not hasattr(huggingface_hub, 'ModelFilter'):
        class ModelFilter:
            def __init__(self, *args, **kwargs):
                pass
        huggingface_hub.ModelFilter = ModelFilter
except:
    pass

import nemo.collections.tts as nemo_tts
from nemo.collections.tts.models import Tacotron2Model, HifiGanModel
import torch
import logging
import os
from typing import Optional
import tempfile
import time
import soundfile as sf
import numpy as np
import io
import asyncio
import aiofiles

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
TACOTRON_MODEL = os.getenv("TACOTRON_MODEL", "tts_en_tacotron2")
VOCODER_MODEL = os.getenv("VOCODER_MODEL", "tts_en_hifigan")
DEVICE = os.getenv("DEVICE", "cuda")
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/app/models")
SHARED_SECRET = os.getenv("SHARED_SECRET", "")
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "22050"))
DEFAULT_SPEAKER = int(os.getenv("DEFAULT_SPEAKER", "0"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

app = FastAPI(title="TTS NeMo Service")

# Global model variables
spec_generator = None
vocoder = None
device = None


class TextToSpeechRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    speed: Optional[float] = 1.0
    speaker_id: Optional[int] = None


def verify_auth(authorization: Optional[str] = Header(None)):
    """Verify shared secret if configured"""
    if SHARED_SECRET and authorization != f"Bearer {SHARED_SECRET}":
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.on_event("startup")
async def startup_event():
    """Load NeMo TTS models on startup"""
    global spec_generator, vocoder, device
    
    try:
        # Set up device
        if DEVICE == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU device")
        
        # Set model cache directory
        os.environ['NEMO_CACHE_DIR'] = MODEL_CACHE_DIR
        
        # Load Tacotron2 model (spectrogram generator)
        logger.info(f"Loading Tacotron2 model: {TACOTRON_MODEL}")
        start_time = time.time()
        
        # Try to load Tacotron2 with retries
        model_loaded = False
        tacotron_models = [
            TACOTRON_MODEL,
            "tts_en_tacotron2",
            "nvidia/tts_en_tacotron2"
        ]
        
        for model_name in tacotron_models:
            for attempt in range(MAX_RETRIES):
                try:
                    logger.info(f"Attempting to load {model_name} (attempt {attempt+1}/{MAX_RETRIES})")
                    spec_generator = Tacotron2Model.from_pretrained(model_name)
                    logger.info(f"Successfully loaded Tacotron2: {model_name}")
                    model_loaded = True
                    break
                except Exception as e:
                    logger.warning(f"Failed to load {model_name}: {e}")
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(5 * (attempt + 1))
            
            if model_loaded:
                break
        
        if not model_loaded:
            raise RuntimeError("Failed to load Tacotron2 model after all attempts")
        
        spec_generator = spec_generator.to(device)
        spec_generator.eval()
        
        tacotron_time = time.time() - start_time
        logger.info(f"Tacotron2 loaded in {tacotron_time:.2f} seconds")
        
        # Load HiFi-GAN vocoder
        logger.info(f"Loading HiFi-GAN vocoder: {VOCODER_MODEL}")
        start_time = time.time()
        
        # Try to load vocoder with retries
        model_loaded = False
        vocoder_models = [
            VOCODER_MODEL,
            "tts_en_hifigan",
            "nvidia/tts_en_hifigan"
        ]
        
        for model_name in vocoder_models:
            for attempt in range(MAX_RETRIES):
                try:
                    logger.info(f"Attempting to load {model_name} (attempt {attempt+1}/{MAX_RETRIES})")
                    vocoder = HifiGanModel.from_pretrained(model_name)
                    logger.info(f"Successfully loaded HiFi-GAN: {model_name}")
                    model_loaded = True
                    break
                except Exception as e:
                    logger.warning(f"Failed to load {model_name}: {e}")
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(5 * (attempt + 1))
            
            if model_loaded:
                break
        
        if not model_loaded:
            raise RuntimeError("Failed to load HiFi-GAN vocoder after all attempts")
        
        vocoder = vocoder.to(device)
        vocoder.eval()
        
        hifigan_time = time.time() - start_time
        logger.info(f"HiFi-GAN loaded in {hifigan_time:.2f} seconds")
        
        # Warm up the models
        logger.info("Warming up TTS models...")
        with torch.no_grad():
            parsed = spec_generator.parse("Hello world")
            spectrogram = spec_generator.generate_spectrogram(tokens=parsed)
            _ = vocoder.convert_spectrogram_to_audio(spec=spectrogram)
        logger.info("Model warmup complete")
        
    except Exception as e:
        logger.error(f"Failed to load NeMo TTS models: {e}")
        raise RuntimeError(f"Model initialization failed: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global spec_generator, vocoder
    if spec_generator:
        del spec_generator
    if vocoder:
        del vocoder
    torch.cuda.empty_cache()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health_info = {
        "status": "healthy" if (spec_generator and vocoder) else "unhealthy",
        "models": {
            "tacotron2": {
                "requested": TACOTRON_MODEL,
                "loaded": type(spec_generator).__name__ if spec_generator else None
            },
            "vocoder": {
                "requested": VOCODER_MODEL,
                "loaded": type(vocoder).__name__ if vocoder else None
            }
        },
        "device": str(device) if device else "not initialized",
        "cuda_available": torch.cuda.is_available(),
        "sample_rate": SAMPLE_RATE
    }
    
    if torch.cuda.is_available() and device and device.type == "cuda":
        health_info["gpu_info"] = {
            "name": torch.cuda.get_device_name(0),
            "memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**3:.2f} GB",
            "memory_reserved": f"{torch.cuda.memory_reserved(0) / 1024**3:.2f} GB"
        }
    
    return health_info


async def generate_speech_nemo(text: str, speed: float = 1.0, speaker_id: Optional[int] = None) -> bytes:
    """Generate speech using NeMo Tacotron2 + HiFi-GAN"""
    try:
        # Parse and generate speech
        with torch.no_grad():
            # Parse text to tokens
            parsed = spec_generator.parse(text)
            
            # Generate mel-spectrogram from tokens
            spectrogram = spec_generator.generate_spectrogram(tokens=parsed)
            
            # Convert spectrogram to audio using vocoder
            audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)
        
        # Convert to numpy array
        if isinstance(audio, torch.Tensor):
            audio_np = audio.squeeze().cpu().numpy()
        else:
            audio_np = audio
        
        # Ensure audio is in the correct range
        if audio_np.max() > 1.0 or audio_np.min() < -1.0:
            audio_np = np.clip(audio_np, -1.0, 1.0)
        
        # Convert to 16-bit PCM
        audio_int16 = (audio_np * 32767).astype(np.int16)
        
        # Save to WAV format
        buffer = io.BytesIO()
        sf.write(buffer, audio_int16, SAMPLE_RATE, format='WAV', subtype='PCM_16')
        buffer.seek(0)
        
        return buffer.read()
        
    except Exception as e:
        logger.error(f"Speech generation failed: {e}")
        raise


@app.post("/speak")
async def text_to_speech(
    request: TextToSpeechRequest,
    authorization: Optional[str] = Header(None)
):
    """
    Convert text to speech using NeMo TTS
    API compatible with tts-service
    """
    verify_auth(authorization)
    
    if not spec_generator or not vocoder:
        raise HTTPException(status_code=503, detail="Models not available")
    
    try:
        logger.info(f"Generating speech for text: {request.text[:100]}...")
        
        start_time = time.time()
        
        # Generate speech
        audio_data = await generate_speech_nemo(
            text=request.text,
            speed=request.speed,
            speaker_id=request.speaker_id or DEFAULT_SPEAKER
        )
        
        generation_time = time.time() - start_time
        logger.info(f"Speech generated in {generation_time:.2f}s for {len(request.text)} characters")
        logger.info(f"Generated audio size: {len(audio_data) / 1024:.1f} KB")
        
        # Return audio stream
        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=speech.wav",
                "X-Processing-Time": str(generation_time),
                "X-Model": f"{TACOTRON_MODEL}+{VOCODER_MODEL}"
            }
        )
        
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/speak_batch")
async def text_to_speech_batch(
    texts: list[str],
    speed: float = 1.0,
    speaker_id: Optional[int] = None,
    authorization: Optional[str] = Header(None)
):
    """Generate speech for multiple texts"""
    verify_auth(authorization)
    
    if not spec_generator or not vocoder:
        raise HTTPException(status_code=503, detail="Models not available")
    
    results = []
    start_time = time.time()
    
    for text in texts:
        try:
            audio_data = await generate_speech_nemo(
                text=text,
                speed=speed,
                speaker_id=speaker_id or DEFAULT_SPEAKER
            )
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_data)
                results.append({
                    "text": text,
                    "audio_path": tmp.name,
                    "size_kb": len(audio_data) / 1024
                })
        except Exception as e:
            results.append({
                "text": text,
                "error": str(e)
            })
    
    total_time = time.time() - start_time
    
    return {
        "results": results,
        "batch_size": len(texts),
        "total_time": total_time,
        "avg_time_per_text": total_time / len(texts)
    }


@app.get("/voices")
async def list_voices():
    """List available voices/speakers"""
    voices_info = {
        "default_voice": "en_US",
        "default_speaker_id": DEFAULT_SPEAKER,
        "multi_speaker": False
    }
    
    # Check if model supports multiple speakers
    if spec_generator and hasattr(spec_generator, 'speaker_emb'):
        if hasattr(spec_generator.speaker_emb, 'num_embeddings'):
            num_speakers = spec_generator.speaker_emb.num_embeddings
            voices_info["multi_speaker"] = True
            voices_info["num_speakers"] = num_speakers
            voices_info["speaker_ids"] = list(range(num_speakers))
    
    return voices_info


@app.get("/models")
async def list_models():
    """List available models and current configuration"""
    return {
        "current_models": {
            "tacotron2": {
                "requested": TACOTRON_MODEL,
                "loaded": type(spec_generator).__name__ if spec_generator else None
            },
            "vocoder": {
                "requested": VOCODER_MODEL,
                "loaded": type(vocoder).__name__ if vocoder else None
            }
        },
        "available_models": {
            "tacotron2": [
                "tts_en_tacotron2",
                "nvidia/tts_en_tacotron2"
            ],
            "vocoders": [
                "tts_en_hifigan",
                "nvidia/tts_en_hifigan",
                "tts_en_waveglow"
            ]
        },
        "device": str(device) if device else "not initialized",
        "sample_rate": SAMPLE_RATE
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)