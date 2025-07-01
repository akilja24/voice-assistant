#!/usr/bin/env python3
"""
Direct test of whisper service
"""
import requests
import time
import sys

def test_whisper_service():
    """Test the whisper service directly"""
    
    # Wait for service to be ready
    print("Waiting for whisper service to be ready...")
    for i in range(30):
        try:
            response = requests.get("http://localhost:8001/health")
            if response.status_code == 200:
                print(f"Service is healthy: {response.json()}")
                break
        except:
            pass
        time.sleep(1)
    else:
        print("Service failed to become healthy")
        return
    
    # Test transcription
    audio_file = "tests/audio_samples/Clear-Short_16k.wav"
    print(f"\nTesting transcription with: {audio_file}")
    
    with open(audio_file, 'rb') as f:
        files = {'audio': ('test.wav', f, 'audio/wav')}
        
        start_time = time.time()
        try:
            response = requests.post(
                "http://localhost:8001/transcribe",
                files=files,
                timeout=120
            )
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                print(f"\nTranscription successful in {elapsed:.2f}s:")
                print(f"Text: {result['text']}")
                print(f"Language: {result.get('language', 'N/A')}")
                print(f"Duration: {result.get('duration', 'N/A')}s")
                print(f"Processing time: {result.get('processing_time', 'N/A')}s")
                print(f"RTF: {result.get('rtf', 'N/A')}")
                if 'gpu_stats' in result:
                    print(f"GPU Memory: {result['gpu_stats'].get('memory_used_mb', 'N/A')} MB")
            else:
                print(f"Error: {response.status_code} - {response.text}")
                
        except requests.exceptions.Timeout:
            print("Request timed out after 120 seconds")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_whisper_service()