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
    """Download default Piper model on startup if not exists"""
    if TTS_ENGINE == "piper":
        model_file = f"{PIPER_MODEL_PATH}/{TTS_VOICE}.onnx"
        if not os.path.exists(model_file):
            logger.info(f"Downloading Piper model: {TTS_VOICE}")
            try:
                # Download model using wget
                model_url = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/{TTS_VOICE.replace('-', '/')}/{TTS_VOICE}.onnx"
                config_url = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/{TTS_VOICE.replace('-', '/')}/{TTS_VOICE}.onnx.json"
                
                os.makedirs(PIPER_MODEL_PATH, exist_ok=True)
                
                subprocess.run([
                    "wget", "-q", "-O", model_file, model_url
                ], check=True)
                
                subprocess.run([
                    "wget", "-q", "-O", f"{model_file}.json", config_url
                ], check=True)
                
                logger.info(f"Model downloaded successfully: {TTS_VOICE}")
            except Exception as e:
                logger.error(f"Failed to download model: {e}")
                # Continue anyway, will fail on first request


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "engine": TTS_ENGINE}


async def generate_speech_piper(text: str, voice: str) -> bytes:
    """Generate speech using Piper TTS"""
    model_path = f"{PIPER_MODEL_PATH}/{voice}.onnx"
    
    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=400,
            detail=f"Voice model not found: {voice}"
        )
    
    # Create temporary file for output
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_file_path = tmp_file.name
    
    try:
        # Run Piper
        cmd = [
            "piper",
            "--model", model_path,
            "--output_file", tmp_file_path
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate(input=text.encode())
        
        if process.returncode != 0:
            logger.error(f"Piper error: {stderr.decode()}")
            raise HTTPException(
                status_code=500,
                detail="Failed to generate speech"
            )
        
        # Read the generated audio
        async with aiofiles.open(tmp_file_path, 'rb') as f:
            audio_data = await f.read()
        
        return audio_data
        
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
            audio_data = await generate_speech_piper(request.text, voice)
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