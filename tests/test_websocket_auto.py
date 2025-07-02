#!/usr/bin/env python3
"""
WebSocket client for testing the auto-detection endpoint with automatic end-of-utterance
"""
import asyncio
import websockets
import json
import wave
import time
from pathlib import Path


async def stream_audio_with_pauses(websocket, audio_file_path, chunk_duration=0.1):
    """
    Stream audio file with realistic timing to simulate real-time speech
    """
    print(f"\nStreaming audio file: {audio_file_path}")
    
    # Read WAV file
    with wave.open(audio_file_path, 'rb') as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        framerate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        duration = n_frames / framerate
        
        print(f"Audio info: {channels} channels, {sample_width} bytes/sample, {framerate} Hz")
        print(f"Duration: {duration:.2f} seconds")
        
        # Read all frames
        frames = wav_file.readframes(n_frames)
    
    # Calculate chunk size based on duration
    bytes_per_second = framerate * channels * sample_width
    chunk_size = int(bytes_per_second * chunk_duration)
    
    print(f"Streaming with {chunk_duration}s chunks ({chunk_size} bytes each)")
    
    # Stream audio chunks with timing
    start_time = time.time()
    total_sent = 0
    
    for i in range(0, len(frames), chunk_size):
        chunk = frames[i:i + chunk_size]
        await websocket.send(chunk)
        total_sent += len(chunk)
        
        # Progress indicator
        progress = (total_sent / len(frames)) * 100
        elapsed = time.time() - start_time
        print(f"\rStreaming: {progress:.1f}% ({elapsed:.1f}s)", end='', flush=True)
        
        # Wait to simulate real-time streaming
        await asyncio.sleep(chunk_duration)
    
    print(f"\nFinished streaming in {time.time() - start_time:.1f}s")
    
    # Add extra silence at the end to trigger EOU
    print("Adding silence to trigger end-of-utterance detection...")
    silence_duration = 1.0  # 1 second of silence
    silence_frames = int(framerate * silence_duration)
    silence_data = b'\x00' * (silence_frames * channels * sample_width)
    
    # Send silence in chunks
    for i in range(0, len(silence_data), chunk_size):
        chunk = silence_data[i:i + chunk_size]
        await websocket.send(chunk)
        await asyncio.sleep(chunk_duration)
    
    print("Silence sent, waiting for automatic detection...")


async def test_auto_detection(audio_file_path):
    """
    Test the auto-detection WebSocket endpoint
    """
    uri = "ws://localhost:8080/ws/auto"
    print(f"Connecting to {uri}")
    
    async with websockets.connect(uri) as websocket:
        print("Connected to auto-detection endpoint")
        
        # Create tasks for sending and receiving
        send_task = asyncio.create_task(stream_audio_with_pauses(websocket, audio_file_path))
        
        # Receive responses
        response_audio = bytearray()
        metadata = None
        eou_info = None
        
        try:
            while True:
                message = await websocket.recv()
                
                if isinstance(message, bytes):
                    # Audio response
                    response_audio.extend(message)
                    if len(response_audio) % 10240 == 0:  # Log every 10KB
                        print(f"\rReceiving audio: {len(response_audio)} bytes", end='', flush=True)
                else:
                    # JSON message
                    data = json.loads(message)
                    print(f"\nReceived: {data}")
                    
                    if data.get("type") == "eou_detected":
                        eou_info = data
                        print("\n🎯 End-of-utterance detected!")
                        print(f"   Probability: {data.get('probability', 0):.2f}")
                        print(f"   Reason: {data.get('reason', '')}")
                        print(f"   Punctuated text: {data.get('punctuated_text', '')}")
                        
                    elif data.get("type") == "metadata":
                        metadata = data
                        
                    elif data.get("type") == "audio_complete":
                        print("\n✅ Audio response complete!")
                        break
                        
                    elif data.get("type") == "error":
                        print(f"\n❌ Error: {data.get('message')}")
                        break
                        
        except websockets.exceptions.ConnectionClosed:
            print("\nConnection closed")
        
        # Cancel send task if still running
        if not send_task.done():
            send_task.cancel()
            try:
                await send_task
            except asyncio.CancelledError:
                pass
        
        # Save response if received
        if response_audio:
            output_path = Path("tests") / f"auto_response_{Path(audio_file_path).stem}.wav"
            with open(output_path, "wb") as f:
                f.write(bytes(response_audio))
            print(f"\n💾 Saved response to {output_path}")
            print(f"   Size: {len(response_audio)} bytes")
            
            if metadata:
                print(f"   Format: {metadata.get('audio_info', {})}")


async def test_all_samples():
    """
    Test all sample audio files with auto-detection
    """
    audio_files = [
        "tests/audio_samples/Clear-Short_16k.wav",
        "tests/audio_samples/Clear-Medium_16k.wav",
        "tests/audio_samples/Noisy-Short_16k.wav"
    ]
    
    for idx, audio_file in enumerate(audio_files):
        if Path(audio_file).exists():
            print(f"\n{'='*60}")
            print(f"Test {idx + 1}/{len(audio_files)}: {Path(audio_file).name}")
            print('='*60)
            
            await test_auto_detection(audio_file)
            
            if idx < len(audio_files) - 1:
                print("\nWaiting 3 seconds before next test...")
                await asyncio.sleep(3)
        else:
            print(f"\n⚠️  File not found: {audio_file}")
    
    print(f"\n{'='*60}")
    print("All tests completed!")


if __name__ == "__main__":
    import sys
    
    print("WebSocket Auto-Detection Test Client")
    print("===================================")
    print("This tests the automatic end-of-utterance detection")
    print()
    
    if len(sys.argv) > 1:
        # Test specific file
        audio_file = sys.argv[1]
        asyncio.run(test_auto_detection(audio_file))
    else:
        # Test all samples
        asyncio.run(test_all_samples())