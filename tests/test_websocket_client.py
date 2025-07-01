#!/usr/bin/env python3
"""
WebSocket client for testing the orchestrator service with real audio files
"""
import asyncio
import websockets
import json
import wave
import struct
import os
from pathlib import Path


async def send_audio_file(websocket, audio_file_path):
    """
    Send an audio file through the WebSocket connection
    """
    print(f"\nTesting with audio file: {audio_file_path}")
    
    # Read WAV file and extract PCM data
    with wave.open(audio_file_path, 'rb') as wav_file:
        # Get audio parameters
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        framerate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        
        print(f"Audio info: {channels} channels, {sample_width} bytes/sample, {framerate} Hz, {n_frames} frames")
        
        # Read all frames
        frames = wav_file.readframes(n_frames)
    
    # Send audio in chunks (64KB chunks as specified)
    chunk_size = 64 * 1024
    total_sent = 0
    
    for i in range(0, len(frames), chunk_size):
        chunk = frames[i:i + chunk_size]
        await websocket.send(chunk)
        total_sent += len(chunk)
        print(f"Sent audio chunk: {len(chunk)} bytes (total: {total_sent}/{len(frames)})")
        await asyncio.sleep(0.05)  # Small delay between chunks
    
    # Send end_stream message
    end_message = json.dumps({"type": "end_stream"})
    await websocket.send(end_message)
    print("Sent end_stream message")
    
    # Receive responses
    response_audio = bytearray()
    metadata = None
    
    while True:
        message = await websocket.recv()
        
        if isinstance(message, bytes):
            # Binary audio data
            response_audio.extend(message)
            print(f"Received audio chunk: {len(message)} bytes")
        else:
            # JSON message
            data = json.loads(message)
            print(f"Received message: {data}")
            
            if data.get("type") == "metadata":
                metadata = data
            elif data.get("type") == "audio_complete":
                print("Audio stream complete!")
                break
            elif data.get("type") == "error":
                print(f"Error: {data.get('message')}")
                break
    
    return response_audio, metadata


async def test_audio_files():
    """
    Test the WebSocket endpoint with all available audio files
    """
    # Audio files to test
    audio_files = [
        "tests/audio_samples/Clear-Short_16k.wav",
        "tests/audio_samples/Clear-Medium_16k.wav",
        "tests/audio_samples/Noisy-Short_16k.wav"
    ]
    
    uri = "ws://localhost:8080/ws"
    print(f"Connecting to {uri}")
    
    for idx, audio_file in enumerate(audio_files):
        audio_path = Path(audio_file)
        if not audio_path.exists():
            print(f"Warning: {audio_file} not found, skipping...")
            continue
        
        async with websockets.connect(uri) as websocket:
            print(f"\n{'='*60}")
            print(f"Test {idx + 1}/{len(audio_files)}")
            
            response_audio, metadata = await send_audio_file(websocket, audio_file)
            
            # Save response audio
            if response_audio:
                output_filename = f"response_{audio_path.stem}.wav"
                output_path = Path("tests") / output_filename
                
                with open(output_path, "wb") as f:
                    f.write(bytes(response_audio))
                
                print(f"\nSaved response audio to {output_path}")
                print(f"Response size: {len(response_audio)} bytes")
                
                if metadata:
                    print(f"Audio metadata: {metadata.get('audio_info', {})}")
        
        # Small delay between tests
        await asyncio.sleep(1)
    
    print(f"\n{'='*60}")
    print("All tests completed!")


async def test_single_file(file_path):
    """
    Test with a single audio file
    """
    uri = "ws://localhost:8080/ws"
    
    async with websockets.connect(uri) as websocket:
        response_audio, metadata = await send_audio_file(websocket, file_path)
        
        if response_audio:
            output_path = "response_single.wav"
            with open(output_path, "wb") as f:
                f.write(bytes(response_audio))
            print(f"\nSaved response to {output_path}")


if __name__ == "__main__":
    import sys
    
    print("WebSocket Audio Streaming Test Client")
    print("=====================================")
    
    if len(sys.argv) > 1:
        # Test with specific file
        audio_file = sys.argv[1]
        print(f"Testing with specific file: {audio_file}")
        asyncio.run(test_single_file(audio_file))
    else:
        # Test with all sample files
        print("Testing with all sample audio files...")
        asyncio.run(test_audio_files())