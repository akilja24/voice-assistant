from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
import httpx
import json
import logging
from typing import Optional, List
import os
import asyncio
from contextlib import asynccontextmanager
import aiofiles
import tempfile
import struct
import wave
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
WHISPER_SERVICE_URL = os.getenv("WHISPER_SERVICE_URL", "http://whisper-service:8001")
TTS_SERVICE_URL = os.getenv("TTS_SERVICE_URL", "http://tts-service:8003")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
SHARED_SECRET = os.getenv("SHARED_SECRET", "")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.http_client = httpx.AsyncClient(timeout=120.0)
    yield
    # Shutdown
    await app.state.http_client.aclose()


app = FastAPI(title="Voice Assistant Orchestrator", lifespan=lifespan)


async def save_pcm_to_wav(pcm_data: bytes, sample_rate: int = 16000, channels: int = 1, sample_width: int = 2) -> str:
    """
    Save raw PCM data to a WAV file
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir="/tmp") as tmp_file:
        wav_path = tmp_file.name
    
    async with aiofiles.open(wav_path, 'wb') as wav_file:
        # Create WAV file with proper headers
        with wave.open(wav_path, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_data)
    
    return wav_path


async def process_audio_pipeline(audio_path: str, http_client: httpx.AsyncClient) -> bytes:
    """
    Process audio through the full pipeline: Whisper -> Ollama -> TTS
    Returns the final audio bytes
    """
    try:
        # Step 1: Transcribe audio
        logger.info("Transcribing audio...")
        async with aiofiles.open(audio_path, 'rb') as f:
            audio_data = await f.read()
        
        files = {"audio": ("audio.wav", audio_data, "audio/wav")}
        headers = {}
        if SHARED_SECRET:
            headers["Authorization"] = f"Bearer {SHARED_SECRET}"
        
        try:
            response = await http_client.post(
                f"{WHISPER_SERVICE_URL}/transcribe",
                files=files,
                headers=headers,
                timeout=60.0  # 60 second timeout for transcription
            )
            response.raise_for_status()
            result = response.json()
            transcription = result["text"]
            logger.info(f"Transcription: {transcription[:100]}...")
            
        except httpx.TimeoutException:
            logger.error("Whisper service timeout")
            transcription = "Sorry, the transcription service timed out."
        except httpx.HTTPError as e:
            logger.error(f"Whisper service error: {e}")
            transcription = "Sorry, I couldn't transcribe the audio."
        
        # Step 2: Generate response with Ollama
        logger.info("Generating LLM response...")
        try:
            # Strong instruction for very short responses
            system_prompt = "You are a voice assistant. You MUST respond in ONLY 1-2 short sentences. Maximum 30 words total. Be extremely concise and direct."
            prompt = f"{system_prompt}\n\nUser: {transcription}\n\nAssistant (remember: 1-2 sentences, max 30 words):"
            
            response = await http_client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL, 
                    "prompt": prompt, 
                    "stream": False,
                    "options": {
                        "num_predict": 50,  # Limit tokens to force short response
                        "temperature": 0.7,
                        "top_p": 0.9
                    }
                },
                timeout=30.0
            )
            response.raise_for_status()
            llm_text = response.json()["response"]
            
            # Truncate if still too long (backup measure)
            if len(llm_text) > 150:
                llm_text = llm_text[:150].rsplit('.', 1)[0] + '.'
            
            logger.info(f"LLM response ({len(llm_text)} chars): {llm_text}")
        except Exception as e:
            logger.warning(f"Ollama error (using fallback): {e}")
            # Fallback response if Ollama is not available
            llm_text = f"I heard you say: '{transcription}'. I'm here to help you with any questions or tasks you might have."
        
        # Step 3: Convert to speech
        logger.info("Converting to speech...")
        try:
            response = await http_client.post(
                f"{TTS_SERVICE_URL}/speak",
                json={"text": llm_text},
                headers=headers if SHARED_SECRET else {},
                timeout=30.0
            )
            response.raise_for_status()
            audio_bytes = response.content
            logger.info(f"TTS generated {len(audio_bytes)} bytes of audio")
        except Exception as e:
            logger.error(f"TTS service error: {e}")
            # Generate error beep as fallback
            import numpy as np
            sample_rate = 22050
            duration = 0.5
            frequency = 440.0
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            audio = np.sin(frequency * 2 * np.pi * t)
            audio = (audio * 32767).astype(np.int16)
            audio_bytes = audio.tobytes()
            
            # Wrap in WAV format
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                with wave.open(tmp.name, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    wf.writeframes(audio_bytes)
                
                async with aiofiles.open(tmp.name, 'rb') as f:
                    audio_bytes = await f.read()
                
                os.unlink(tmp.name)
        
        return audio_bytes
        
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.websocket("/ws/interact")
async def websocket_interact(websocket: WebSocket):
    """
    WebSocket endpoint for real-time audio streaming interaction
    """
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    audio_chunks: List[bytes] = []
    
    try:
        while True:
            # Receive message (can be binary audio or JSON control)
            message = await websocket.receive()
            
            if "bytes" in message:
                # Binary audio chunk received
                audio_chunk = message["bytes"]
                audio_chunks.append(audio_chunk)
                logger.info(f"Received audio chunk: {len(audio_chunk)} bytes")
                
            elif "text" in message:
                # JSON control message
                try:
                    control_msg = json.loads(message["text"])
                    
                    if control_msg.get("type") == "end_stream":
                        logger.info("End of audio stream received")
                        
                        # Combine all audio chunks
                        if not audio_chunks:
                            await websocket.send_json({
                                "type": "error",
                                "message": "No audio data received"
                            })
                            continue
                        
                        pcm_data = b''.join(audio_chunks)
                        logger.info(f"Total audio size: {len(pcm_data)} bytes")
                        
                        # Save PCM to WAV
                        wav_path = await save_pcm_to_wav(pcm_data)
                        logger.info(f"Saved audio to: {wav_path}")
                        
                        try:
                            # Process through pipeline
                            response_audio = await process_audio_pipeline(wav_path, app.state.http_client)
                            
                            # Send metadata
                            await websocket.send_json({
                                "type": "metadata",
                                "audio_info": {
                                    "format": "wav",
                                    "sample_rate": 22050,
                                    "bitrate": 128
                                }
                            })
                            
                            # Stream audio in chunks (4KB chunks)
                            chunk_size = 4096
                            for i in range(0, len(response_audio), chunk_size):
                                chunk = response_audio[i:i + chunk_size]
                                await websocket.send_bytes(chunk)
                                await asyncio.sleep(0.01)  # Small delay to prevent overwhelming
                            
                            # Send completion message
                            await websocket.send_json({
                                "type": "audio_complete"
                            })
                            
                            logger.info("Audio response sent successfully")
                            
                        finally:
                            # Clean up temporary file
                            if os.path.exists(wav_path):
                                os.unlink(wav_path)
                        
                        # Reset for next interaction
                        audio_chunks.clear()
                        
                except json.JSONDecodeError:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Invalid JSON control message"
                    })
                    
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        # Check if it's a disconnect-related error
        if "disconnect" in str(e).lower():
            logger.info("WebSocket closed by client")
        else:
            logger.error(f"WebSocket error: {e}")
            try:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
            except:
                pass


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for audio streaming
    """
    await websocket.accept()
    logger.info(f"WebSocket connection established on /ws from {websocket.client}")
    
    audio_chunks: List[bytes] = []
    
    try:
        while True:
            # Receive message (can be binary audio or JSON control)
            message = await websocket.receive()
            
            if "bytes" in message:
                # Binary audio chunk received
                audio_chunk = message["bytes"]
                audio_chunks.append(audio_chunk)
                logger.info(f"Received audio chunk: {len(audio_chunk)} bytes (total chunks: {len(audio_chunks)})")
                
            elif "text" in message:
                # JSON control message
                try:
                    control_msg = json.loads(message["text"])
                    logger.info(f"Received control message: {control_msg}")
                    
                    if control_msg.get("type") == "end_stream":
                        logger.info("End of audio stream received, processing...")
                        
                        # Combine all audio chunks
                        if not audio_chunks:
                            await websocket.send_json({
                                "type": "error",
                                "message": "No audio data received"
                            })
                            continue
                        
                        pcm_data = b''.join(audio_chunks)
                        logger.info(f"Total audio size: {len(pcm_data)} bytes")
                        
                        # Save PCM to WAV
                        wav_path = await save_pcm_to_wav(pcm_data)
                        logger.info(f"Saved audio to: {wav_path}")
                        
                        try:
                            # Process through pipeline
                            response_audio = await process_audio_pipeline(wav_path, app.state.http_client)
                            
                            # Send metadata
                            await websocket.send_json({
                                "type": "metadata",
                                "audio_info": {
                                    "format": "wav",
                                    "sample_rate": 22050,
                                    "bitrate": 128
                                }
                            })
                            
                            # Stream audio in chunks (4KB chunks)
                            chunk_size = 4096
                            for i in range(0, len(response_audio), chunk_size):
                                chunk = response_audio[i:i + chunk_size]
                                await websocket.send_bytes(chunk)
                                await asyncio.sleep(0.01)  # Small delay to prevent overwhelming
                            
                            # Send completion message
                            await websocket.send_json({
                                "type": "audio_complete"
                            })
                            
                            logger.info("Audio response sent successfully")
                            
                        finally:
                            # Clean up temporary file
                            if os.path.exists(wav_path):
                                os.unlink(wav_path)
                        
                        # Reset for next interaction
                        audio_chunks.clear()
                        
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON received: {message['text']}")
                    await websocket.send_json({
                        "type": "error",
                        "message": "Invalid JSON control message"
                    })
                    
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        # Check if it's a disconnect-related error
        if "disconnect" in str(e).lower():
            logger.info("WebSocket closed by client")
        else:
            logger.error(f"WebSocket error: {e}")
            try:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
            except:
                pass


@app.post("/interact")
async def interact(audio_file: UploadFile = File(...)):
    """
    REST endpoint for single audio file interaction
    """
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            content = await audio_file.read()
            tmp_file.write(content)
            wav_path = tmp_file.name
        
        try:
            # Process through pipeline
            response_audio = await process_audio_pipeline(wav_path, app.state.http_client)
            
            # Return audio response
            return StreamingResponse(
                asyncio.io.BytesIO(response_audio),
                media_type="audio/wav",
                headers={"Content-Disposition": "attachment; filename=response.wav"}
            )
            
        finally:
            # Clean up
            if os.path.exists(wav_path):
                os.unlink(wav_path)
                
    except Exception as e:
        logger.error(f"Error in interact endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)