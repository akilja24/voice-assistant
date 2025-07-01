from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import subprocess
import tempfile
import os
import logging
from typing import Optional
import httpx
import asyncio
import aiofiles
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
TTS_ENGINE = os.getenv("TTS_ENGINE", "piper")
TTS_VOICE = os.getenv("TTS_VOICE", "en_US-amy-medium")
SHARED_SECRET = os.getenv("SHARED_SECRET", "")
PIPER_MODEL_PATH = os.getenv("PIPER_MODEL_PATH", "/app/models")

app = FastAPI(title="TTS Service")


class TextToSpeechRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    speed: Optional[float] = 1.0


def verify_auth(authorization: Optional[str] = Header(None)):
    """Verify shared secret if configured"""
    if SHARED_SECRET and authorization != f"Bearer {SHARED_SECRET}":
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.on_event("startup")
async def startup_event():
    """Verify Piper is installed and model is available"""
    if TTS_ENGINE == "piper":
        # Check if piper binary is available
        try:
            result = subprocess.run(["piper", "--version"], capture_output=True, text=True)
            logger.info(f"Piper version: {result.stdout.strip()}")
        except FileNotFoundError:
            logger.error("Piper binary not found in PATH")
            raise RuntimeError("Piper TTS binary not installed")
        
        # Check if default model exists
        model_file = f"{PIPER_MODEL_PATH}/{TTS_VOICE}.onnx"
        if os.path.exists(model_file):
            logger.info(f"Default voice model available: {TTS_VOICE}")
            # Get model file size for debugging
            size_mb = os.path.getsize(model_file) / (1024 * 1024)
            logger.info(f"Model size: {size_mb:.1f} MB")
        else:
            logger.warning(f"Default voice model not found: {model_file}")
            logger.info(f"Available models: {os.listdir(PIPER_MODEL_PATH) if os.path.exists(PIPER_MODEL_PATH) else []}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "engine": TTS_ENGINE}


async def generate_speech_piper(text: str, voice: str, speed: float = 1.0) -> bytes:
    """Generate speech using Piper TTS with optimized settings"""
    model_path = f"{PIPER_MODEL_PATH}/{voice}.onnx"
    
    if not os.path.exists(model_path):
        # Try alternative path structure
        alt_model_path = f"{PIPER_MODEL_PATH}/{voice.replace('_', '/')}/{voice}.onnx"
        if os.path.exists(alt_model_path):
            model_path = alt_model_path
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Voice model not found: {voice}"
            )
    
    # Create temporary file for output
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_file_path = tmp_file.name
    
    try:
        # Run Piper with optimized settings
        cmd = [
            "piper",
            "--model", model_path,
            "--output_file", tmp_file_path,
            "--length_scale", str(1.0 / speed),  # Adjust speed
            "--sentence_silence", "0.2"  # Add natural pauses
        ]
        
        # Log the command for debugging
        logger.debug(f"Running Piper command: {' '.join(cmd)}")
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Start timing
        start_time = asyncio.get_event_loop().time()
        
        stdout, stderr = await process.communicate(input=text.encode('utf-8'))
        
        # Log generation time
        generation_time = asyncio.get_event_loop().time() - start_time
        logger.info(f"Speech generated in {generation_time:.2f}s for {len(text)} characters")
        
        if process.returncode != 0:
            logger.error(f"Piper error: {stderr.decode()}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate speech: {stderr.decode()}"
            )
        
        # Read the generated audio
        async with aiofiles.open(tmp_file_path, 'rb') as f:
            audio_data = await f.read()
        
        logger.info(f"Generated audio size: {len(audio_data) / 1024:.1f} KB")
        return audio_data
        
    except Exception as e:
        logger.error(f"Error in generate_speech_piper: {str(e)}")
        raise
    finally:
        # Clean up
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)


async def generate_speech_polly(text: str, voice: str) -> bytes:
    """Generate speech using AWS Polly (placeholder)"""
    # This would require AWS credentials and boto3 setup
    raise HTTPException(
        status_code=501,
        detail="AWS Polly integration not implemented"
    )


@app.post("/speak")
async def text_to_speech(
    request: TextToSpeechRequest,
    authorization: Optional[str] = Header(None)
):
    """
    Convert text to speech
    """
    verify_auth(authorization)
    
    try:
        voice = request.voice or TTS_VOICE
        
        logger.info(f"Generating speech for text: {request.text[:100]}...")
        
        if TTS_ENGINE == "piper":
            audio_data = await generate_speech_piper(request.text, voice, request.speed)
        elif TTS_ENGINE == "polly":
            audio_data = await generate_speech_polly(request.text, voice)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown TTS engine: {TTS_ENGINE}"
            )
        
        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=speech.wav"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/voices")
async def list_voices(authorization: Optional[str] = Header(None)):
    """List available voices"""
    verify_auth(authorization)
    
    if TTS_ENGINE == "piper":
        # List downloaded models
        voices = []
        if os.path.exists(PIPER_MODEL_PATH):
            for file in os.listdir(PIPER_MODEL_PATH):
                if file.endswith(".onnx"):
                    voices.append(file.replace(".onnx", ""))
        
        return {
            "engine": TTS_ENGINE,
            "voices": voices,
            "default": TTS_VOICE
        }
    else:
        return {
            "engine": TTS_ENGINE,
            "voices": [],
            "default": TTS_VOICE
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)