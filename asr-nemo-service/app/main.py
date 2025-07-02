from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.responses import JSONResponse

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

import nemo.collections.asr as nemo_asr
import torch
import logging
import os
from typing import Optional, Dict, Any, List
import tempfile
import time
import soundfile as sf
import numpy as np
import librosa

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
NEMO_MODEL = os.getenv("NEMO_MODEL", "stt_en_quartznet15x5")
DEVICE = os.getenv("DEVICE", "cuda")
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/app/models")
SHARED_SECRET = os.getenv("SHARED_SECRET", "")
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

app = FastAPI(title="ASR NeMo Service")

# Global model variable
asr_model = None
device = None


def verify_auth(authorization: Optional[str] = Header(None)):
    """Verify shared secret if configured"""
    if SHARED_SECRET and authorization != f"Bearer {SHARED_SECRET}":
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.on_event("startup")
async def startup_event():
    """Load NeMo ASR model on startup"""
    global asr_model, device
    
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
        
        logger.info(f"Loading NeMo ASR model: {NEMO_MODEL}")
        start_time = time.time()
        
        # Try to load the model with retries and fallbacks
        model_loaded = False
        fallback_models = [
            NEMO_MODEL,
            "stt_en_quartznet15x5",
            "QuartzNet15x5Base-En",
            "stt_en_citrinet_256",
            "nvidia/stt_en_citrinet_256"
        ]
        
        for model_name in fallback_models:
            for attempt in range(MAX_RETRIES):
                try:
                    logger.info(f"Attempting to load {model_name} (attempt {attempt+1}/{MAX_RETRIES})")
                    asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name)
                    logger.info(f"Successfully loaded model: {model_name}")
                    model_loaded = True
                    break
                except Exception as e:
                    logger.warning(f"Failed to load {model_name}: {e}")
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(5 * (attempt + 1))  # Exponential backoff
            
            if model_loaded:
                break
        
        if not model_loaded:
            raise RuntimeError("Failed to load any ASR model after all attempts")
        
        # Move model to device
        asr_model = asr_model.to(device)
        asr_model.eval()
        
        load_time = time.time() - start_time
        logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
        
        # Log model info
        logger.info(f"Model type: {type(asr_model).__name__}")
        
        # Warm up the model with a dummy input
        logger.info("Warming up the model...")
        dummy_audio = np.zeros(SAMPLE_RATE, dtype=np.float32)  # 1 second of silence
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            sf.write(tmp.name, dummy_audio, SAMPLE_RATE)
            _ = asr_model.transcribe([tmp.name])
        logger.info("Model warmup complete")
        
    except Exception as e:
        logger.error(f"Failed to load NeMo ASR model: {e}")
        raise RuntimeError(f"Model initialization failed: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global asr_model
    if asr_model:
        del asr_model
        torch.cuda.empty_cache()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health_info = {
        "status": "healthy" if asr_model else "unhealthy",
        "model": {
            "requested": NEMO_MODEL,
            "loaded": type(asr_model).__name__ if asr_model else None
        },
        "device": str(device) if device else "not initialized",
        "cuda_available": torch.cuda.is_available()
    }
    
    if torch.cuda.is_available() and device and device.type == "cuda":
        health_info["gpu_info"] = {
            "name": torch.cuda.get_device_name(0),
            "memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**3:.2f} GB",
            "memory_reserved": f"{torch.cuda.memory_reserved(0) / 1024**3:.2f} GB"
        }
    
    return health_info


@app.post("/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    authorization: Optional[str] = Header(None)
):
    """
    Transcribe audio file to text using NeMo ASR
    API compatible with whisper-service
    """
    verify_auth(authorization)
    
    if not asr_model:
        raise HTTPException(status_code=503, detail="Model not available")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            content = await audio.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Log file info
        file_size = len(content) / 1024  # KB
        logger.info(f"Transcribing audio file: {audio.filename} ({file_size:.1f} KB)")
        
        # Load and preprocess audio
        try:
            # Load audio file
            audio_data, sr = sf.read(tmp_file_path)
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Resample if necessary
            if sr != SAMPLE_RATE:
                logger.info(f"Resampling from {sr} Hz to {SAMPLE_RATE} Hz")
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=SAMPLE_RATE)
            
            # Save preprocessed audio
            processed_path = tmp_file_path + "_processed.wav"
            sf.write(processed_path, audio_data, SAMPLE_RATE)
            
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid audio file: {e}")
        
        # Start transcription
        start_time = time.time()
        
        # Transcribe with NeMo
        with torch.no_grad():
            transcriptions = asr_model.transcribe([processed_path])
        
        # Get the transcription text
        if isinstance(transcriptions, list) and len(transcriptions) > 0:
            if isinstance(transcriptions[0], tuple):
                # Some models return (text, additional_info)
                text = transcriptions[0][0] if transcriptions[0] else ""
            else:
                text = transcriptions[0]
        else:
            text = str(transcriptions)
        
        transcription_time = time.time() - start_time
        
        # Calculate audio duration
        audio_duration = len(audio_data) / SAMPLE_RATE
        rtf = transcription_time / audio_duration if audio_duration > 0 else 0
        
        logger.info(f"Transcription completed in {transcription_time:.2f}s (RTF: {rtf:.2f})")
        logger.info(f"Transcription: {text[:100]}...")
        
        # Return response compatible with whisper-service
        response = {
            "text": text,
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": audio_duration,
                    "text": text,
                    "confidence": 1.0  # NeMo doesn't provide confidence by default
                }
            ],
            "language": "en",
            "duration": audio_duration,
            "processing_time": transcription_time,
            "rtf": rtf,
            "model": NEMO_MODEL
        }
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary files
        for path in [tmp_file_path, processed_path] if 'processed_path' in locals() else [tmp_file_path]:
            if os.path.exists(path):
                try:
                    os.unlink(path)
                except:
                    pass


@app.post("/transcribe_batch")
async def transcribe_batch(
    files: List[UploadFile] = File(...),
    authorization: Optional[str] = Header(None)
):
    """
    Transcribe multiple audio files in batch
    """
    verify_auth(authorization)
    
    if not asr_model:
        raise HTTPException(status_code=503, detail="Model not available")
    
    temp_files = []
    try:
        # Save all uploaded files
        for file in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                temp_files.append(tmp_file.name)
        
        # Transcribe in batch
        start_time = time.time()
        with torch.no_grad():
            transcriptions = asr_model.transcribe(temp_files)
        batch_time = time.time() - start_time
        
        # Format results
        results = []
        for i, (file, transcription) in enumerate(zip(files, transcriptions)):
            text = transcription[0] if isinstance(transcription, tuple) else transcription
            results.append({
                "filename": file.filename,
                "text": text
            })
        
        return {
            "results": results,
            "batch_size": len(files),
            "processing_time": batch_time
        }
        
    finally:
        # Clean up
        for path in temp_files:
            if os.path.exists(path):
                try:
                    os.unlink(path)
                except:
                    pass


@app.get("/models")
async def list_models():
    """List available models and current configuration"""
    return {
        "current_model": {
            "requested": NEMO_MODEL,
            "loaded": type(asr_model).__name__ if asr_model else None
        },
        "available_models": [
            "stt_en_quartznet15x5",
            "QuartzNet15x5Base-En",
            "stt_en_citrinet_256",
            "stt_en_citrinet_512",
            "stt_en_citrinet_1024",
            "stt_en_conformer_ctc_small",
            "stt_en_conformer_ctc_large",
            "stt_en_fastconformer_ctc_small",
            "stt_en_fastconformer_ctc_large"
        ],
        "device": str(device) if device else "not initialized",
        "sample_rate": SAMPLE_RATE
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)