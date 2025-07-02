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
import websockets

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
WHISPER_SERVICE_URL = os.getenv("WHISPER_SERVICE_URL", "http://whisper-service:8001")
TTS_SERVICE_URL = os.getenv("TTS_SERVICE_URL", "http://tts-service:8003")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
SHARED_SECRET = os.getenv("SHARED_SECRET", "")
EOU_SERVICE_URL = os.getenv("EOU_SERVICE_URL", "http://eou-service:8004")

# Backend selection
ASR_BACKEND = os.getenv("ASR_BACKEND", "whisper")  # "whisper" or "nemo"
TTS_BACKEND = os.getenv("TTS_BACKEND", "piper")    # "piper" or "nemo"

# NeMo service URLs
ASR_NEMO_SERVICE_URL = os.getenv("ASR_NEMO_SERVICE_URL", "http://asr-nemo-service:8005")
TTS_NEMO_SERVICE_URL = os.getenv("TTS_NEMO_SERVICE_URL", "http://tts-nemo-service:8006")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.http_client = httpx.AsyncClient(timeout=120.0)
    yield
    # Shutdown
    await app.state.http_client.aclose()


app = FastAPI(title="Voice Assistant Orchestrator", lifespan=lifespan)

# Store active audio streaming tasks for interruption
active_audio_streams: dict = {}


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
        logger.info(f"Transcribing audio using {ASR_BACKEND} backend...")
        async with aiofiles.open(audio_path, 'rb') as f:
            audio_data = await f.read()
        
        files = {"audio": ("audio.wav", audio_data, "audio/wav")}
        headers = {}
        if SHARED_SECRET:
            headers["Authorization"] = f"Bearer {SHARED_SECRET}"
        
        # Select ASR service URL based on backend
        asr_url = ASR_NEMO_SERVICE_URL if ASR_BACKEND == "nemo" else WHISPER_SERVICE_URL
        
        try:
            response = await http_client.post(
                f"{asr_url}/transcribe",
                files=files,
                headers=headers,
                timeout=60.0  # 60 second timeout for transcription
            )
            response.raise_for_status()
            result = response.json()
            transcription = result["text"]
            logger.info(f"Transcription ({ASR_BACKEND}): {transcription[:100]}...")
            
        except httpx.TimeoutException:
            logger.error(f"{ASR_BACKEND} service timeout")
            transcription = "Sorry, the transcription service timed out."
        except httpx.HTTPError as e:
            logger.error(f"{ASR_BACKEND} service error: {e}")
            transcription = "Sorry, I couldn't transcribe the audio."
        
        # Step 2: Generate response with Ollama
        logger.info("Generating LLM response...")
        try:
            # Natural conversational prompt without strict limits
            system_prompt = "You are a helpful voice assistant. Provide clear, concise responses that are natural for spoken conversation."
            prompt = f"{system_prompt}\n\nUser: {transcription}\n\nAssistant:"
            
            response = await http_client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL, 
                    "prompt": prompt, 
                    "stream": False,
                    "options": {
                        "num_predict": 300,  # Allow reasonable response length
                        "temperature": 0.7,
                        "top_p": 0.9
                    }
                },
                timeout=30.0
            )
            response.raise_for_status()
            llm_text = response.json()["response"]
            
            logger.info(f"LLM response ({len(llm_text)} chars): {llm_text}")
        except Exception as e:
            logger.warning(f"Ollama error (using fallback): {e}")
            # Fallback response if Ollama is not available
            llm_text = f"I heard you say: '{transcription}'. I'm here to help you with any questions or tasks you might have."
        
        # Step 3: Convert to speech
        logger.info(f"Converting to speech using {TTS_BACKEND} backend...")
        
        # Select TTS service URL based on backend
        tts_url = TTS_NEMO_SERVICE_URL if TTS_BACKEND == "nemo" else TTS_SERVICE_URL
        
        try:
            response = await http_client.post(
                f"{tts_url}/speak",
                json={"text": llm_text},
                headers=headers if SHARED_SECRET else {},
                timeout=30.0
            )
            response.raise_for_status()
            audio_bytes = response.content
            logger.info(f"TTS ({TTS_BACKEND}) generated {len(audio_bytes)} bytes of audio")
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
    return {
        "status": "healthy",
        "backends": {
            "asr": ASR_BACKEND,
            "tts": TTS_BACKEND
        },
        "services": {
            "asr": {
                "whisper": WHISPER_SERVICE_URL,
                "nemo": ASR_NEMO_SERVICE_URL
            },
            "tts": {
                "piper": TTS_SERVICE_URL,
                "nemo": TTS_NEMO_SERVICE_URL
            },
            "llm": OLLAMA_URL,
            "eou": EOU_SERVICE_URL
        }
    }


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
                            
                            # Create interruptible audio streaming task
                            async def stream_audio():
                                try:
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
                                except asyncio.CancelledError:
                                    logger.info(f"Audio streaming cancelled for {stream_id}")
                                    raise
                            
                            # Start streaming task and track it
                            current_stream_task = asyncio.create_task(stream_audio())
                            active_audio_streams[stream_id] = current_stream_task
                            
                            try:
                                await current_stream_task
                            except asyncio.CancelledError:
                                pass
                            finally:
                                # Clean up
                                if stream_id in active_audio_streams:
                                    del active_audio_streams[stream_id]
                            
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
    finally:
        # Clean up any active streams
        if stream_id in active_audio_streams:
            active_audio_streams[stream_id].cancel()
            del active_audio_streams[stream_id]


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for audio streaming
    """
    await websocket.accept()
    logger.info(f"WebSocket connection established on /ws from {websocket.client}")
    
    audio_chunks: List[bytes] = []
    stream_id = id(websocket)
    current_stream_task = None
    
    try:
        while True:
            # Receive message (can be binary audio or JSON control)
            message = await websocket.receive()
            
            if "bytes" in message:
                # Binary audio chunk received
                audio_chunk = message["bytes"]
                
                # Cancel any active audio streaming for this connection
                if stream_id in active_audio_streams and active_audio_streams[stream_id]:
                    logger.info(f"Interrupting active audio stream for {stream_id}")
                    active_audio_streams[stream_id].cancel()
                    await websocket.send_json({
                        "type": "playback_interrupted",
                        "message": "Previous response interrupted"
                    })
                
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
                            
                            # Create interruptible audio streaming task
                            async def stream_audio():
                                try:
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
                                except asyncio.CancelledError:
                                    logger.info(f"Audio streaming cancelled for {stream_id}")
                                    raise
                            
                            # Start streaming task and track it
                            current_stream_task = asyncio.create_task(stream_audio())
                            active_audio_streams[stream_id] = current_stream_task
                            
                            try:
                                await current_stream_task
                            except asyncio.CancelledError:
                                pass
                            finally:
                                # Clean up
                                if stream_id in active_audio_streams:
                                    del active_audio_streams[stream_id]
                            
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
    finally:
        # Clean up any active streams
        if stream_id in active_audio_streams:
            active_audio_streams[stream_id].cancel()
            del active_audio_streams[stream_id]


@app.websocket("/ws/auto")
async def websocket_auto_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint with automatic end-of-utterance detection
    """
    await websocket.accept()
    logger.info(f"WebSocket auto-detection connection established from {websocket.client}")
    
    audio_chunks: List[bytes] = []
    eou_websocket = None
    stream_id = f"stream_{id(websocket)}"
    processing_lock = asyncio.Lock()
    is_processing = False
    stream_ws_id = id(websocket)
    
    try:
        # Connect to EOU service
        import websockets
        eou_ws_url = f"{EOU_SERVICE_URL.replace('http://', 'ws://')}/ws/stream/{stream_id}"
        eou_websocket = await websockets.connect(eou_ws_url)
        logger.info(f"Connected to EOU service for stream {stream_id}")
        
        # Create task to handle EOU responses
        async def handle_eou_responses():
            nonlocal is_processing
            try:
                async for message in eou_websocket:
                    data = json.loads(message)
                    logger.info(f"EOU response: {data}")
                    
                    if data.get("type") == "eou_status" and data.get("is_end_of_utterance"):
                        async with processing_lock:
                            if not is_processing and audio_chunks:
                                is_processing = True
                                logger.info("End of utterance detected, processing audio...")
                                
                                # Send transcription update to client
                                await websocket.send_json({
                                    "type": "eou_detected",
                                    "probability": data.get("probability", 0),
                                    "punctuated_text": data.get("punctuated_text", ""),
                                    "reason": data.get("reason", "")
                                })
                                
                                # Process accumulated audio
                                pcm_data = b''.join(audio_chunks)
                                wav_path = await save_pcm_to_wav(pcm_data)
                                
                                try:
                                    # Process through pipeline
                                    response_audio = await process_audio_pipeline(wav_path, app.state.http_client)
                                    
                                    # Send response to client
                                    await websocket.send_json({
                                        "type": "metadata",
                                        "audio_info": {
                                            "format": "wav",
                                            "sample_rate": 22050,
                                            "bitrate": 128
                                        }
                                    })
                                    
                                    # Create interruptible audio streaming task
                                    async def stream_audio():
                                        try:
                                            # Stream audio response
                                            chunk_size = 4096
                                            for i in range(0, len(response_audio), chunk_size):
                                                chunk = response_audio[i:i + chunk_size]
                                                await websocket.send_bytes(chunk)
                                                await asyncio.sleep(0.01)
                                            
                                            await websocket.send_json({"type": "audio_complete"})
                                        except asyncio.CancelledError:
                                            logger.info(f"Audio streaming cancelled for {stream_ws_id}")
                                            raise
                                    
                                    # Start streaming task and track it
                                    stream_task = asyncio.create_task(stream_audio())
                                    active_audio_streams[stream_ws_id] = stream_task
                                    
                                    try:
                                        await stream_task
                                    except asyncio.CancelledError:
                                        pass
                                    finally:
                                        # Clean up
                                        if stream_ws_id in active_audio_streams:
                                            del active_audio_streams[stream_ws_id]
                                    
                                    # Clear audio chunks for next utterance
                                    audio_chunks.clear()
                                    
                                    # Reset EOU stream state
                                    await eou_websocket.send(json.dumps({"type": "reset"}))
                                    
                                finally:
                                    if os.path.exists(wav_path):
                                        os.unlink(wav_path)
                                    is_processing = False
                                    
            except Exception as e:
                logger.error(f"Error handling EOU responses: {e}")
        
        # Start EOU response handler
        eou_task = asyncio.create_task(handle_eou_responses())
        
        # Handle incoming audio
        while True:
            message = await websocket.receive()
            
            if "bytes" in message:
                # Audio chunk received
                audio_chunk = message["bytes"]
                
                # Cancel any active audio streaming for this connection
                if stream_ws_id in active_audio_streams and active_audio_streams[stream_ws_id]:
                    logger.info(f"Interrupting active audio stream for {stream_ws_id}")
                    active_audio_streams[stream_ws_id].cancel()
                    await websocket.send_json({
                        "type": "playback_interrupted",
                        "message": "Previous response interrupted"
                    })
                
                audio_chunks.append(audio_chunk)
                
                # Forward to EOU service
                await eou_websocket.send(audio_chunk)
                
                logger.debug(f"Received and forwarded audio chunk: {len(audio_chunk)} bytes")
                
            elif "text" in message:
                # Control message
                try:
                    control_msg = json.loads(message["text"])
                    
                    if control_msg.get("type") == "stop_stream":
                        logger.info("Stop stream requested")
                        break
                        
                except json.JSONDecodeError:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Invalid JSON control message"
                    })
                    
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        except:
            pass
    finally:
        # Clean up any active streams
        if stream_ws_id in active_audio_streams:
            active_audio_streams[stream_ws_id].cancel()
            del active_audio_streams[stream_ws_id]
        
        # Clean up EOU connection
        if eou_websocket:
            await eou_websocket.close()
        
        # Cancel EOU task if running
        if 'eou_task' in locals():
            eou_task.cancel()
            try:
                await eou_task
            except asyncio.CancelledError:
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