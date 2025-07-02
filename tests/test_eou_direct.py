#!/usr/bin/env python3
"""
Direct test of EOU service endpoints
"""
import asyncio
import websockets
import json
import numpy as np
import requests

def test_eou_detect():
    """Test the /detect_eou endpoint"""
    url = "http://localhost:8004/detect_eou"
    
    # Test with complete sentence
    data = {
        "transcript": "I want to learn more about quantum computing.",
        "sample_rate": 16000
    }
    
    print("Testing EOU detection...")
    print(f"Transcript: {data['transcript']}")
    
    response = requests.post(url, json=data)
    if response.status_code == 200:
        result = response.json()
        print(f"Result: {json.dumps(result, indent=2)}")
    else:
        print(f"Error: {response.status_code} - {response.text}")
    
    print("\n" + "="*50 + "\n")
    
    # Test with incomplete sentence
    data = {
        "transcript": "I want to learn more about",
        "sample_rate": 16000
    }
    
    print("Testing with incomplete sentence...")
    print(f"Transcript: {data['transcript']}")
    
    response = requests.post(url, json=data)
    if response.status_code == 200:
        result = response.json()
        print(f"Result: {json.dumps(result, indent=2)}")
    else:
        print(f"Error: {response.status_code} - {response.text}")


async def test_eou_websocket():
    """Test the WebSocket streaming endpoint"""
    uri = "ws://localhost:8004/ws/stream/test123"
    
    async with websockets.connect(uri) as websocket:
        print("\nTesting WebSocket streaming...")
        print("Connected to EOU WebSocket")
        
        # Send a transcript update
        await websocket.send(json.dumps({
            "type": "transcript_update",
            "text": "Hello, how are you doing today?",
            "is_final": True
        }))
        
        # Wait for response
        response = await websocket.recv()
        print(f"Response: {json.loads(response)}")
        
        # Simulate some silence (send silent audio)
        silent_audio = np.zeros(1600, dtype=np.int16)  # 100ms of silence at 16kHz
        
        # Send multiple chunks to trigger EOU detection
        for i in range(10):  # 1 second of silence
            await websocket.send(silent_audio.tobytes())
            await asyncio.sleep(0.1)
            
            # Try to receive any EOU status
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                data = json.loads(response)
                if data.get("type") == "eou_status":
                    print(f"\nEOU Status at {i*100}ms:")
                    print(f"  End of utterance: {data.get('is_end_of_utterance')}")
                    print(f"  Probability: {data.get('probability')}")
                    print(f"  Silence duration: {data.get('silence_duration_ms')}ms")
                    print(f"  Reason: {data.get('reason')}")
                    
                    if data.get('is_end_of_utterance'):
                        print("\n✅ End of utterance detected!")
                        break
            except asyncio.TimeoutError:
                pass
        
        # Send reset
        await websocket.send(json.dumps({"type": "reset"}))
        response = await websocket.recv()
        print(f"\nReset response: {json.loads(response)}")


if __name__ == "__main__":
    print("EOU Service Direct Test")
    print("=" * 50)
    
    # Test REST endpoint
    test_eou_detect()
    
    # Test WebSocket endpoint
    asyncio.run(test_eou_websocket())