from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from faster_whisper import WhisperModel
import numpy as np
import io
import logging
import os
from typing import Optional
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny.en")
DEVICE = os.getenv("DEVICE", "cuda")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16")
SHARED_SECRET = os.getenv("SHARED_SECRET", "")

app = FastAPI(title="Whisper Service")

# Initialize model on startup
model = None


@app.on_event("startup")
async def startup_event():
    global model
    logger.info(f"Loading Whisper model: {WHISPER_MODEL}")
    try:
        model = WhisperModel(
            WHISPER_MODEL,
            device=DEVICE,
            compute_type=COMPUTE_TYPE
        )
        logger.info("Model loaded successfully")
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
    return {"status": "healthy", "model": WHISPER_MODEL}


@app.post("/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    authorization: Optional[str] = Header(None)
):
    """
    Transcribe audio file to text using Whisper
    """
    verify_auth(authorization)
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            content = await audio.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Transcribe audio
        logger.info(f"Transcribing audio file: {audio.filename}")
        segments, info = model.transcribe(
            tmp_file_path,
            beam_size=5,
            language="en",
            vad_filter=True,  # Enable VAD
            vad_parameters=dict(
                min_silence_duration_ms=500,
                threshold=0.5
            )
        )
        
        # Collect transcription
        transcription = " ".join([segment.text.strip() for segment in segments])
        
        # Clean up
        os.unlink(tmp_file_path)
        
        logger.info(f"Transcription complete: {transcription[:100]}...")
        
        return {
            "text": transcription,
            "language": info.language,
            "duration": info.duration
        }
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        # Clean up on error
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe_stream")
async def transcribe_stream(
    audio_chunk: bytes,
    authorization: Optional[str] = Header(None)
):
    """
    Transcribe audio stream chunk (for future WebSocket implementation)
    """
    verify_auth(authorization)
    
    # This is a placeholder for streaming implementation
    # Real implementation would maintain state across chunks
    return {
        "partial": True,
        "text": "Streaming transcription not yet implemented"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)