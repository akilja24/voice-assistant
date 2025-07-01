from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from faster_whisper import WhisperModel
import logging
import os
from typing import Optional, Dict, Any
import tempfile
import time
from pathlib import Path
try:
    import pynvml as nvml
except ImportError:
    nvml = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables with defaults
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
DEVICE = os.getenv("DEVICE", "cuda")
DEVICE_INDEX = int(os.getenv("DEVICE_INDEX", "0"))
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16")
BEAM_SIZE = int(os.getenv("BEAM_SIZE", "5"))
BEST_OF = int(os.getenv("BEST_OF", "5"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
VAD_FILTER = os.getenv("VAD_FILTER", "true").lower() == "true"
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.5"))
MIN_SILENCE_DURATION_MS = int(os.getenv("MIN_SILENCE_DURATION_MS", "500"))
COMPRESSION_RATIO_THRESHOLD = float(os.getenv("COMPRESSION_RATIO_THRESHOLD", "2.4"))
LOG_PROB_THRESHOLD = float(os.getenv("LOG_PROB_THRESHOLD", "-1.0"))
NO_SPEECH_THRESHOLD = float(os.getenv("NO_SPEECH_THRESHOLD", "0.6"))
SHARED_SECRET = os.getenv("SHARED_SECRET", "")
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/app/models")

app = FastAPI(title="Whisper Service")

# Initialize model on startup
model = None
gpu_available = False


def get_gpu_info() -> Dict[str, Any]:
    """Get GPU information if available"""
    if not nvml:
        return {}
    try:
        nvml.nvmlInit()
        device_count = nvml.nvmlDeviceGetCount()
        if device_count > 0 and DEVICE_INDEX < device_count:
            handle = nvml.nvmlDeviceGetHandleByIndex(DEVICE_INDEX)
            info = {
                "name": nvml.nvmlDeviceGetName(handle) if isinstance(nvml.nvmlDeviceGetName(handle), str) else nvml.nvmlDeviceGetName(handle).decode('utf-8'),
                "memory_total": nvml.nvmlDeviceGetMemoryInfo(handle).total // 1024**2,  # MB
                "memory_used": nvml.nvmlDeviceGetMemoryInfo(handle).used // 1024**2,  # MB
                "temperature": nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU),
                "utilization": nvml.nvmlDeviceGetUtilizationRates(handle).gpu
            }
            return info
    except Exception as e:
        logger.warning(f"Could not get GPU info: {e}")
    return {}


@app.on_event("startup")
async def startup_event():
    global model, gpu_available
    
    # Check GPU availability
    gpu_info = get_gpu_info()
    if gpu_info:
        gpu_available = True
        logger.info(f"GPU detected: {gpu_info.get('name', 'Unknown')}")
        logger.info(f"GPU Memory: {gpu_info.get('memory_used', 0)}/{gpu_info.get('memory_total', 0)} MB")
    else:
        logger.error("No GPU detected. This service requires GPU.")
        raise RuntimeError("GPU is required but not available")
        
    # Create model cache directory
    Path(MODEL_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading Whisper model: {WHISPER_MODEL}")
    logger.info(f"Device: {DEVICE}, Compute Type: {COMPUTE_TYPE}")
    
    try:
        start_time = time.time()
        model = WhisperModel(
            WHISPER_MODEL,
            device="cuda",  # Always use CUDA
            device_index=DEVICE_INDEX,
            compute_type=COMPUTE_TYPE,  # float16 for GPU
            cpu_threads=1,  # Minimal CPU threads for GPU mode
            download_root=MODEL_CACHE_DIR
        )
        load_time = time.time() - start_time
        logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
        
        # Log model info
        logger.info(f"Model loaded: {WHISPER_MODEL}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def verify_auth(authorization: Optional[str] = Header(None)):
    """Verify shared secret if configured"""
    if SHARED_SECRET and authorization != f"Bearer {SHARED_SECRET}":
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    health_info = {
        "status": "healthy",
        "model": WHISPER_MODEL,
        "device": "cuda",
        "compute_type": COMPUTE_TYPE
    }
    
    # Add GPU info if available
    if gpu_available:
        gpu_info = get_gpu_info()
        if gpu_info:
            health_info["gpu"] = gpu_info
            
    return health_info


@app.post("/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    authorization: Optional[str] = Header(None)
):
    """
    Transcribe audio file to text using Whisper
    """
    verify_auth(authorization)
    
    if not model:
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
        
        # Start transcription
        start_time = time.time()
        
        segments, info = model.transcribe(
            tmp_file_path,
            beam_size=BEAM_SIZE,
            best_of=BEST_OF,
            temperature=TEMPERATURE,
            language="en",  # Force English for better performance
            vad_filter=VAD_FILTER,
            vad_parameters=dict(
                min_silence_duration_ms=MIN_SILENCE_DURATION_MS,
                threshold=VAD_THRESHOLD
            ),
            compression_ratio_threshold=COMPRESSION_RATIO_THRESHOLD,
            log_prob_threshold=LOG_PROB_THRESHOLD,
            no_speech_threshold=NO_SPEECH_THRESHOLD,
            word_timestamps=False,  # Disable for faster processing
            prepend_punctuations="\"'\"([{-",
            append_punctuations="\"'.。,!?:)]}\""
        )
        
        # Collect transcription
        transcription_segments = []
        for segment in segments:
            transcription_segments.append(segment.text.strip())
            logger.debug(f"Segment [{segment.start:.2f}s -> {segment.end:.2f}s]: {segment.text.strip()}")
        
        transcription = " ".join(transcription_segments)
        
        # Calculate metrics
        transcription_time = time.time() - start_time
        audio_duration = info.duration
        rtf = transcription_time / audio_duration if audio_duration > 0 else 0
        
        # Clean up
        os.unlink(tmp_file_path)
        
        logger.info(f"Transcription complete in {transcription_time:.2f}s (RTF: {rtf:.2f})")
        logger.info(f"Text: {transcription[:100]}...")
        
        # Get current GPU stats if available
        gpu_stats = {}
        if gpu_available:
            gpu_info = get_gpu_info()
            if gpu_info:
                gpu_stats = {
                    "memory_used_mb": gpu_info.get("memory_used", 0),
                    "gpu_utilization": gpu_info.get("utilization", 0)
                }
        
        return {
            "text": transcription,
            "language": info.language,
            "duration": audio_duration,
            "processing_time": transcription_time,
            "rtf": rtf,  # Real-time factor (< 1 is faster than real-time)
            "gpu_stats": gpu_stats
        }
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        # Clean up on error
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def list_models():
    """List available models in cache"""
    models = []
    if os.path.exists(MODEL_CACHE_DIR):
        for item in os.listdir(MODEL_CACHE_DIR):
            if os.path.isdir(os.path.join(MODEL_CACHE_DIR, item)):
                models.append(item)
    
    return {
        "current_model": WHISPER_MODEL,
        "available_models": models,
        "supported_models": [
            "tiny", "tiny.en", "base", "base.en", 
            "small", "small.en", "medium", "medium.en",
            "large-v1", "large-v2", "large-v3"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)