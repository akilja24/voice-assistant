#!/usr/bin/env python3
"""
Test automatic EOU detection with interruption
"""
import asyncio
import websockets
import json
import numpy as np
import wave

async def test_auto_eou():
    """Test automatic EOU detection endpoint"""
    uri = "ws://localhost:8080/ws/auto"
    
    print("Testing automatic EOU detection with /ws/auto endpoint")
    print("=" * 60)
    
    async with websockets.connect(uri) as websocket:
        print("Connected to WebSocket /ws/auto")
        
        # Load real audio file
        with wave.open("audio_samples/Clear-Short_16k.wav", 'rb') as wf:
            audio_data = wf.readframes(wf.getnframes())
        
        print("\n1. Streaming audio in chunks (simulating real-time speech)...")
        
        # Stream audio in smaller chunks to simulate real-time
        chunk_size = 3200  # 200ms chunks at 16kHz
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i+chunk_size]
            await websocket.send(chunk)
            await asyncio.sleep(0.05)  # Small delay between chunks
            
            # Check for any EOU messages
            try:
                msg = await asyncio.wait_for(websocket.recv(), timeout=0.01)
                if isinstance(msg, str):
                    data = json.loads(msg)
                    if data['type'] == 'eou_detected':
                        print(f"\n✅ EOU Detected!")
                        print(f"   Probability: {data['probability']:.2f}")
                        print(f"   Text: {data['punctuated_text']}")
                        print(f"   Reason: {data['reason']}")
            except asyncio.TimeoutError:
                pass
        
        print("\n2. Sending silence to trigger EOU...")
        
        # Send silence to trigger EOU
        silence = np.zeros(1600, dtype=np.int16)  # 100ms of silence
        for _ in range(15):  # 1.5 seconds of silence
            await websocket.send(silence.tobytes())
            await asyncio.sleep(0.1)
            
            # Check for EOU or response
            try:
                msg = await asyncio.wait_for(websocket.recv(), timeout=0.05)
                if isinstance(msg, str):
                    data = json.loads(msg)
                    print(f"Received: {data['type']}")
                    if data['type'] == 'eou_detected':
                        print(f"✅ EOU Detected after silence!")
                        print(f"   Probability: {data['probability']:.2f}")
                        print(f"   Text: {data['punctuated_text']}")
                        print(f"   Reason: {data['reason']}")
                    elif data['type'] == 'metadata':
                        print("Response started!")
                        break
            except asyncio.TimeoutError:
                pass
        
        # Wait for response
        print("\n3. Receiving response...")
        audio_chunks = 0
        while True:
            try:
                msg = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                if isinstance(msg, bytes):
                    audio_chunks += 1
                elif isinstance(msg, str):
                    data = json.loads(msg)
                    if data['type'] == 'audio_complete':
                        print(f"✅ Response completed. Received {audio_chunks} audio chunks")
                        break
            except asyncio.TimeoutError:
                print("Timeout waiting for response")
                break
        
        # Test interruption
        print("\n4. Testing interruption - sending new audio immediately...")
        
        # Send new audio to interrupt any ongoing playback
        with wave.open("audio_samples/Clear-Medium_16k.wav", 'rb') as wf:
            audio_data = wf.readframes(wf.getnframes())
        
        # Send first chunk
        await websocket.send(audio_data[:6400])
        
        # Check for interruption message
        try:
            msg = await asyncio.wait_for(websocket.recv(), timeout=0.5)
            if isinstance(msg, str):
                data = json.loads(msg)
                if data['type'] == 'playback_interrupted':
                    print(f"✅ INTERRUPTION DETECTED: {data['message']}")
        except asyncio.TimeoutError:
            print("No interruption message received")


if __name__ == "__main__":
    asyncio.run(test_auto_eou())