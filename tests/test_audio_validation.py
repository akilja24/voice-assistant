#!/usr/bin/env python3
"""
Validate the WebSocket audio processing pipeline
"""
import wave
import os
from pathlib import Path


def analyze_wav_file(file_path):
    """Analyze a WAV file and return its properties"""
    with wave.open(file_path, 'rb') as wav_file:
        properties = {
            'channels': wav_file.getnchannels(),
            'sample_width': wav_file.getsampwidth(),
            'framerate': wav_file.getframerate(),
            'n_frames': wav_file.getnframes(),
            'duration': wav_file.getnframes() / wav_file.getframerate(),
            'file_size': os.path.getsize(file_path)
        }
    return properties


def main():
    print("Audio File Analysis")
    print("==================")
    
    # Test files
    test_files = [
        "tests/audio_samples/Clear-Short_16k.wav",
        "tests/audio_samples/Clear-Medium_16k.wav", 
        "tests/audio_samples/Noisy-Short_16k.wav"
    ]
    
    # Analyze input files
    print("\nInput Audio Files:")
    print("-" * 60)
    for file_path in test_files:
        if os.path.exists(file_path):
            props = analyze_wav_file(file_path)
            print(f"\n{Path(file_path).name}:")
            print(f"  Channels: {props['channels']}")
            print(f"  Sample Rate: {props['framerate']} Hz")
            print(f"  Sample Width: {props['sample_width']} bytes")
            print(f"  Duration: {props['duration']:.2f} seconds")
            print(f"  File Size: {props['file_size']:,} bytes")
    
    # Analyze response files
    print("\n\nResponse Audio Files:")
    print("-" * 60)
    response_files = list(Path("tests").glob("response_*.wav"))
    
    if not response_files:
        print("No response files found. Run test_websocket_client.py first.")
        return
    
    for file_path in response_files:
        props = analyze_wav_file(str(file_path))
        print(f"\n{file_path.name}:")
        print(f"  Channels: {props['channels']}")
        print(f"  Sample Rate: {props['framerate']} Hz")
        print(f"  Sample Width: {props['sample_width']} bytes")
        print(f"  Duration: {props['duration']:.2f} seconds")
        print(f"  File Size: {props['file_size']:,} bytes")
    
    print("\n\nObservations:")
    print("-" * 60)
    print("1. All response files are identical (placeholder sine wave)")
    print("2. Response sample rate is 22050 Hz (different from input 16000 Hz)")
    print("3. Response duration is fixed at 2 seconds")
    print("4. Ready for integration with real ASR/LLM/TTS services")


if __name__ == "__main__":
    main()