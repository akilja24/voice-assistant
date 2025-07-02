from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging
import os
from typing import Optional, Dict, Any, List, Tuple
import asyncio
import numpy as np
import json
import time
from collections import deque
import torch
import io
import wave

# Pipecat imports
try:
    from pipecat.vad.silero import SileroVADAnalyzer
    from pipecat.frames.frames import AudioRawFrame, Frame
except ImportError:
    # Try alternative import paths
    try:
        from pipecat.processors.audio.vad.silero import SileroVADAnalyzer
        from pipecat.frames import AudioRawFrame, Frame
    except ImportError:
        SileroVADAnalyzer = None
        AudioRawFrame = None
        Frame = None
        logger.warning("Pipecat imports failed, will use fallback VAD")

from deepmultilingualpunctuation import PunctuationModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
VAD_MODEL = os.getenv("VAD_MODEL", "silero_vad")
VAD_SILENCE_THRESHOLD_MS = int(os.getenv("VAD_SILENCE_THRESHOLD_MS", "400"))
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.5"))
EOU_PROBABILITY_THRESHOLD = float(os.getenv("EOU_PROBABILITY_THRESHOLD", "0.7"))
MAX_SILENCE_TIMEOUT_MS = int(os.getenv("MAX_SILENCE_TIMEOUT_MS", "2000"))
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
SHARED_SECRET = os.getenv("SHARED_SECRET", "")
MIN_SPEECH_DURATION_MS = int(os.getenv("MIN_SPEECH_DURATION_MS", "200"))

app = FastAPI(title="End-of-Utterance Detection Service")

# Global models
vad_analyzer: Optional[SileroVADAnalyzer] = None
punctuation_model: Optional[PunctuationModel] = None


class TranscriptSegment(BaseModel):
    text: str
    timestamp: float
    is_final: bool = False


class EOURequest(BaseModel):
    transcript: str
    audio_chunk: Optional[List[float]] = None
    sample_rate: int = 16000


class EOUResponse(BaseModel):
    is_end_of_utterance: bool
    probability: float
    silence_duration_ms: float
    punctuated_text: Optional[str] = None
    reason: str
    vad_confidence: Optional[float] = None


class AudioStreamState:
    """Maintains state for audio stream processing"""
    def __init__(self):
        self.silence_start_time = None
        self.speech_start_time = None
        self.last_speech_time = time.time()
        self.transcript_buffer = deque(maxlen=10)
        self.audio_buffer = []
        self.is_speaking = False
        self.consecutive_silence_chunks = 0
        self.vad_history = deque(maxlen=10)
        self.speech_segments = []
        self.total_speech_duration = 0


# Store active stream states
stream_states: Dict[str, AudioStreamState] = {}


@app.on_event("startup")
async def startup_event():
    global vad_analyzer, punctuation_model
    
    # Initialize Silero VAD
    logger.info("Loading Silero VAD analyzer...")
    try:
        if SileroVADAnalyzer is not None:
            vad_analyzer = SileroVADAnalyzer()
            logger.info("Silero VAD analyzer initialized successfully")
        else:
            logger.warning("SileroVADAnalyzer not available, using fallback")
            vad_analyzer = None
    except Exception as e:
        logger.error(f"Failed to initialize Silero VAD: {e}")
        vad_analyzer = None
    
    # Initialize punctuation model
    logger.info("Loading punctuation restoration model...")
    try:
        punctuation_model = PunctuationModel()
        logger.info("Punctuation model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load punctuation model: {e}")
        punctuation_model = None


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if vad_analyzer:
        await vad_analyzer.stop()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models": {
            "vad": {
                "type": "silero",
                "loaded": vad_analyzer is not None
            },
            "punctuation": punctuation_model is not None
        },
        "settings": {
            "vad_threshold": VAD_THRESHOLD,
            "silence_threshold_ms": VAD_SILENCE_THRESHOLD_MS,
            "eou_probability_threshold": EOU_PROBABILITY_THRESHOLD
        }
    }


def add_punctuation(text: str) -> str:
    """Add punctuation to text using the punctuation model"""
    if not punctuation_model or not text.strip():
        return text
    
    try:
        return punctuation_model.restore_punctuation(text)
    except Exception as e:
        logger.error(f"Punctuation restoration failed: {e}")
        return text


async def analyze_audio_vad(audio_data: np.ndarray, sample_rate: int) -> Tuple[bool, float]:
    """
    Analyze audio using Silero VAD
    Returns (is_speech, confidence)
    """
    if not vad_analyzer or len(audio_data) == 0:
        return False, 0.0
    
    try:
        # Create an AudioRawFrame for Pipecat
        audio_bytes = (audio_data * 32768).astype(np.int16).tobytes()
        frame = AudioRawFrame(
            audio=audio_bytes,
            sample_rate=sample_rate,
            num_channels=1
        )
        
        # Process through VAD
        confidence = await vad_analyzer.analyze_audio(frame)
        is_speech = confidence >= VAD_THRESHOLD
        
        return is_speech, float(confidence)
    except Exception as e:
        logger.error(f"VAD analysis failed: {e}")
        # Fallback to energy-based detection
        energy = np.sqrt(np.mean(audio_data ** 2))
        return energy > 0.01, energy


def calculate_semantic_eou_probability(text: str, speech_duration_ms: float = 0) -> float:
    """
    Calculate probability that text represents end of utterance
    Uses punctuation and linguistic cues
    """
    if not text.strip():
        return 0.0
    
    # Add punctuation first
    punctuated = add_punctuation(text)
    
    # Base probability
    probability = 0.3
    
    # Check for terminal punctuation
    terminal_punctuation = ['.', '!', '?']
    if any(punctuated.rstrip().endswith(p) for p in terminal_punctuation):
        probability += 0.4
    
    # Check for question patterns
    question_words = ['what', 'when', 'where', 'who', 'why', 'how', 'is', 'are', 'can', 'could', 'would', 'should']
    is_question = any(punctuated.lower().startswith(word) for word in question_words)
    
    if is_question and punctuated.endswith('?'):
        probability += 0.2
    elif is_question and not punctuated.endswith('?'):
        probability -= 0.2  # Incomplete question
    
    # Check for incomplete thought indicators
    incomplete_indicators = ['but', 'and', 'or', 'because', 'so', 'then', ',', 'if', 'when', 'while']
    last_word = punctuated.rstrip().split()[-1].lower() if punctuated.split() else ""
    if last_word in incomplete_indicators:
        probability -= 0.3
    
    # Length considerations
    word_count = len(punctuated.split())
    if word_count < 3:
        probability -= 0.2
    elif word_count > 15:
        probability += 0.1
    
    # Duration consideration
    if speech_duration_ms > 3000:  # More than 3 seconds
        probability += 0.1
    elif speech_duration_ms < 500:  # Less than 0.5 seconds
        probability -= 0.1
    
    # Ensure probability is in valid range
    return max(0.0, min(1.0, probability))


@app.post("/detect_eou")
async def detect_end_of_utterance(request: EOURequest) -> EOUResponse:
    """
    Detect if the given transcript and audio represent end of utterance
    """
    # Add punctuation to transcript
    punctuated_text = add_punctuation(request.transcript) if request.transcript else ""
    
    # Default response
    response = EOUResponse(
        is_end_of_utterance=False,
        probability=0.0,
        silence_duration_ms=0.0,
        punctuated_text=punctuated_text,
        reason="Processing",
        vad_confidence=None
    )
    
    # Check audio for speech using VAD
    is_speech = False
    vad_confidence = 0.0
    
    if request.audio_chunk:
        audio_array = np.array(request.audio_chunk, dtype=np.float32)
        if audio_array.max() > 1.0:
            audio_array = audio_array / 32768.0  # Normalize if needed
        
        is_speech, vad_confidence = await analyze_audio_vad(audio_array, request.sample_rate)
        response.vad_confidence = vad_confidence
    
    # Calculate semantic probability
    semantic_probability = calculate_semantic_eou_probability(punctuated_text)
    response.probability = semantic_probability
    
    # Determine EOU based on both signals
    if not is_speech and semantic_probability >= EOU_PROBABILITY_THRESHOLD:
        response.is_end_of_utterance = True
        response.reason = f"Semantic completion with silence (VAD: {vad_confidence:.2f})"
    elif not is_speech and len(punctuated_text.strip()) > 10:
        # Even without high semantic score, silence after reasonable speech is EOU
        response.is_end_of_utterance = True
        response.reason = f"Extended silence after speech (VAD: {vad_confidence:.2f})"
    else:
        response.reason = f"Speech detected (VAD: {vad_confidence:.2f})" if is_speech else "Continuing utterance"
    
    return response


@app.websocket("/ws/stream/{stream_id}")
async def websocket_stream_endpoint(websocket: WebSocket, stream_id: str):
    """
    WebSocket endpoint for continuous EOU detection on audio stream
    """
    await websocket.accept()
    logger.info(f"Stream {stream_id} connected")
    
    # Initialize stream state
    stream_states[stream_id] = AudioStreamState()
    state = stream_states[stream_id]
    
    try:
        while True:
            message = await websocket.receive()
            
            if "bytes" in message:
                # Process audio chunk
                audio_data = message["bytes"]
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Analyze with VAD
                is_speech, vad_confidence = await analyze_audio_vad(audio_array, SAMPLE_RATE)
                state.vad_history.append((is_speech, vad_confidence))
                
                current_time = time.time()
                
                if is_speech:
                    # Speech detected
                    state.consecutive_silence_chunks = 0
                    state.silence_start_time = None
                    
                    if not state.is_speaking:
                        # Speech just started
                        state.is_speaking = True
                        state.speech_start_time = current_time
                        logger.debug(f"Speech started (VAD: {vad_confidence:.2f})")
                    
                    state.last_speech_time = current_time
                    
                else:
                    # Silence detected
                    state.consecutive_silence_chunks += 1
                    
                    if state.is_speaking:
                        # Speech just ended
                        speech_duration = (current_time - state.speech_start_time) * 1000
                        state.total_speech_duration += speech_duration
                        state.speech_segments.append(speech_duration)
                        state.is_speaking = False
                        logger.debug(f"Speech ended after {speech_duration:.0f}ms")
                    
                    if not state.silence_start_time:
                        state.silence_start_time = current_time
                    
                    silence_duration_ms = (current_time - state.silence_start_time) * 1000
                    
                    # Check if we should evaluate EOU
                    if silence_duration_ms >= VAD_SILENCE_THRESHOLD_MS:
                        # Get current transcript
                        current_transcript = " ".join([seg.text for seg in state.transcript_buffer])
                        
                        if current_transcript or state.total_speech_duration > MIN_SPEECH_DURATION_MS:
                            # Add punctuation
                            punctuated = add_punctuation(current_transcript) if current_transcript else ""
                            
                            # Evaluate semantic EOU with speech duration context
                            semantic_prob = calculate_semantic_eou_probability(
                                punctuated, 
                                state.total_speech_duration
                            )
                            
                            # Adaptive threshold based on silence duration
                            adaptive_threshold = EOU_PROBABILITY_THRESHOLD
                            if silence_duration_ms > 1000:
                                adaptive_threshold *= 0.8
                            if silence_duration_ms > 1500:
                                adaptive_threshold *= 0.9
                            
                            # Smart turn detection logic
                            is_eou = False
                            reason = ""
                            
                            if semantic_prob >= adaptive_threshold:
                                is_eou = True
                                reason = "Semantic completion detected"
                            elif silence_duration_ms >= MAX_SILENCE_TIMEOUT_MS:
                                is_eou = True
                                reason = "Maximum silence timeout"
                            elif state.total_speech_duration > 500 and silence_duration_ms > 800:
                                # Reasonable speech followed by significant pause
                                is_eou = True
                                reason = "Natural pause after speech"
                            
                            # Calculate average VAD confidence
                            avg_vad = np.mean([v[1] for v in state.vad_history]) if state.vad_history else 0
                            
                            await websocket.send_json({
                                "type": "eou_status",
                                "is_end_of_utterance": is_eou,
                                "probability": semantic_prob,
                                "silence_duration_ms": silence_duration_ms,
                                "speech_duration_ms": state.total_speech_duration,
                                "punctuated_text": punctuated,
                                "vad_confidence": float(vad_confidence),
                                "avg_vad_confidence": float(avg_vad),
                                "reason": reason
                            })
                            
                            if is_eou:
                                # Reset state for next utterance
                                state.transcript_buffer.clear()
                                state.silence_start_time = None
                                state.speech_start_time = None
                                state.vad_history.clear()
                                state.speech_segments.clear()
                                state.total_speech_duration = 0
                
            elif "text" in message:
                # Handle control messages
                data = json.loads(message["text"])
                
                if data.get("type") == "transcript_update":
                    # Add new transcript segment
                    segment = TranscriptSegment(
                        text=data["text"],
                        timestamp=time.time(),
                        is_final=data.get("is_final", False)
                    )
                    state.transcript_buffer.append(segment)
                    
                    # Send acknowledgment
                    await websocket.send_json({
                        "type": "transcript_received",
                        "text": data["text"],
                        "buffer_size": len(state.transcript_buffer)
                    })
                    
                elif data.get("type") == "reset":
                    # Reset stream state
                    state = AudioStreamState()
                    stream_states[stream_id] = state
                    await websocket.send_json({"type": "reset_complete"})
                    
    except WebSocketDisconnect:
        logger.info(f"Stream {stream_id} disconnected")
    except Exception as e:
        logger.error(f"Error in stream {stream_id}: {e}")
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })
    finally:
        # Clean up stream state
        if stream_id in stream_states:
            del stream_states[stream_id]


@app.get("/models")
async def list_models():
    """List available models and their status"""
    return {
        "vad": {
            "type": "silero",
            "loaded": vad_analyzer is not None,
            "threshold": VAD_THRESHOLD
        },
        "punctuation": {
            "type": "deepmultilingualpunctuation",
            "loaded": punctuation_model is not None
        },
        "settings": {
            "silence_threshold_ms": VAD_SILENCE_THRESHOLD_MS,
            "eou_probability_threshold": EOU_PROBABILITY_THRESHOLD,
            "max_silence_timeout_ms": MAX_SILENCE_TIMEOUT_MS,
            "min_speech_duration_ms": MIN_SPEECH_DURATION_MS
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)