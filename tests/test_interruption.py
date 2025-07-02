#!/usr/bin/env python3
"""
Test interruption handling in WebSocket audio streaming
"""
import asyncio
import websockets
import json
import numpy as np
import wave

async def test_interruption():
    """Test that sending new audio interrupts the current playback"""
    uri = "ws://localhost:8080/ws"
    
    async with websockets.connect(uri) as websocket:
        print("Connected to WebSocket")
        
        # First interaction - send short audio
        print("\n1. Sending first audio (short question)...")
        
        # Create a short audio sample (0.5 seconds)
        sample_rate = 16000
        duration = 0.5
        frequency = 440.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio = np.sin(frequency * 2 * np.pi * t)
        audio = (audio * 32768).astype(np.int16)
        
        # Send audio
        await websocket.send(audio.tobytes())
        await websocket.send(json.dumps({"type": "end_stream"}))
        
        # Wait for response to start
        print("Waiting for response...")
        response_started = False
        audio_chunks_received = 0
        
        while True:
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                
                if isinstance(message, bytes):
                    audio_chunks_received += 1
                    if not response_started:
                        print(f"Response started, receiving audio chunks...")
                        response_started = True
                else:
                    msg = json.loads(message)
                    print(f"Received: {msg['type']}")
                    
                    if msg['type'] == 'audio_complete':
                        print(f"First response completed. Received {audio_chunks_received} chunks")
                        break
                    
                # After receiving 10 chunks, interrupt with new audio
                if audio_chunks_received == 10:
                    print("\n2. INTERRUPTING - Sending new audio while response is playing...")
                    
                    # Send new audio to interrupt
                    new_audio = np.sin(frequency * 1.5 * 2 * np.pi * t)  # Different frequency
                    new_audio = (new_audio * 32768).astype(np.int16)
                    await websocket.send(new_audio.tobytes())
                    
                    # Look for interruption message
                    interrupt_msg = await websocket.recv()
                    if isinstance(interrupt_msg, str):
                        msg = json.loads(interrupt_msg)
                        if msg['type'] == 'playback_interrupted':
                            print(f"✅ INTERRUPTION CONFIRMED: {msg['message']}")
                        else:
                            print(f"Received: {msg}")
                    
                    # Send end_stream for the interrupting audio
                    await websocket.send(json.dumps({"type": "end_stream"}))
                    
                    # Now wait for the new response
                    print("\n3. Waiting for new response after interruption...")
                    new_chunks = 0
                    while True:
                        message = await websocket.recv()
                        if isinstance(message, bytes):
                            new_chunks += 1
                        else:
                            msg = json.loads(message)
                            print(f"Received: {msg['type']}")
                            if msg['type'] == 'audio_complete':
                                print(f"✅ New response completed. Received {new_chunks} chunks")
                                return
                    
            except asyncio.TimeoutError:
                if not response_started:
                    continue
                else:
                    print("Timeout waiting for response")
                    break


async def test_multiple_interruptions():
    """Test multiple rapid interruptions"""
    uri = "ws://localhost:8080/ws"
    
    async with websockets.connect(uri) as websocket:
        print("\n\nTesting multiple rapid interruptions...")
        print("=" * 50)
        
        sample_rate = 16000
        duration = 0.3
        
        for i in range(3):
            print(f"\nInterruption test {i+1}:")
            
            # Create audio with different frequency each time
            frequency = 440.0 * (i + 1)
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            audio = np.sin(frequency * 2 * np.pi * t)
            audio = (audio * 32768).astype(np.int16)
            
            # Send audio
            await websocket.send(audio.tobytes())
            
            # Quick delay
            await asyncio.sleep(0.5)
            
            # Check for interruption message if not the first one
            if i > 0:
                try:
                    msg = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                    if isinstance(msg, str):
                        data = json.loads(msg)
                        if data['type'] == 'playback_interrupted':
                            print(f"✅ Interruption {i}: {data['message']}")
                except asyncio.TimeoutError:
                    pass
        
        # Send final end_stream
        await websocket.send(json.dumps({"type": "end_stream"}))
        
        # Wait for final response
        print("\nWaiting for final response...")
        while True:
            msg = await websocket.recv()
            if isinstance(msg, str):
                data = json.loads(msg)
                if data['type'] == 'audio_complete':
                    print("✅ Final response completed")
                    break


if __name__ == "__main__":
    print("WebSocket Interruption Test")
    print("===========================")
    
    # Run interruption tests
    asyncio.run(test_interruption())
    asyncio.run(test_multiple_interruptions())
    
    print("\n✅ All interruption tests completed!")