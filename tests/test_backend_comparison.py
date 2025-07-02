#!/usr/bin/env python3
"""
Benchmark and compare Whisper/Piper vs NeMo ASR/TTS backends
"""
import asyncio
import httpx
import time
import json
import statistics
from pathlib import Path
import wave
import numpy as np

# Service URLs
ORCHESTRATOR_URL = "http://localhost:8080"
WHISPER_URL = "http://localhost:8001"
NEMO_ASR_URL = "http://localhost:8005"
PIPER_URL = "http://localhost:8003"
NEMO_TTS_URL = "http://localhost:8006"

# Test data
TEST_AUDIO_FILES = [
    "audio_samples/Clear-Short_16k.wav",
    "audio_samples/Clear-Medium_16k.wav",
    "audio_samples/Noisy-Short_16k.wav"
]

TEST_TEXTS = [
    "Hello, this is a test of the speech synthesis system.",
    "The quick brown fox jumps over the lazy dog. This pangram contains all letters of the alphabet.",
    "Quantum computing leverages quantum mechanical phenomena to process information in fundamentally new ways."
]


async def test_asr_service(service_url: str, audio_file: str, service_name: str) -> dict:
    """Test an ASR service with an audio file"""
    async with httpx.AsyncClient(timeout=60.0) as client:
        with open(audio_file, 'rb') as f:
            files = {"audio": (Path(audio_file).name, f.read(), "audio/wav")}
        
        start_time = time.time()
        try:
            response = await client.post(f"{service_url}/transcribe", files=files)
            response.raise_for_status()
            result = response.json()
            end_time = time.time()
            
            # Get audio duration
            with wave.open(audio_file, 'rb') as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                duration = frames / float(rate)
            
            processing_time = end_time - start_time
            rtf = processing_time / duration
            
            return {
                "service": service_name,
                "file": Path(audio_file).name,
                "transcription": result.get("text", ""),
                "processing_time": processing_time,
                "audio_duration": duration,
                "rtf": rtf,
                "success": True
            }
        except Exception as e:
            return {
                "service": service_name,
                "file": Path(audio_file).name,
                "error": str(e),
                "success": False
            }


async def test_tts_service(service_url: str, text: str, service_name: str) -> dict:
    """Test a TTS service with text"""
    async with httpx.AsyncClient(timeout=60.0) as client:
        start_time = time.time()
        try:
            response = await client.post(
                f"{service_url}/speak",
                json={"text": text}
            )
            response.raise_for_status()
            end_time = time.time()
            
            audio_data = response.content
            processing_time = end_time - start_time
            
            # Calculate audio duration from WAV data
            import io
            import wave
            with io.BytesIO(audio_data) as audio_io:
                with wave.open(audio_io, 'rb') as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    duration = frames / float(rate)
            
            return {
                "service": service_name,
                "text_length": len(text),
                "audio_size_kb": len(audio_data) / 1024,
                "audio_duration": duration,
                "processing_time": processing_time,
                "chars_per_second": len(text) / processing_time,
                "rtf": processing_time / duration,
                "success": True
            }
        except Exception as e:
            return {
                "service": service_name,
                "text_length": len(text),
                "error": str(e),
                "success": False
            }


async def benchmark_asr():
    """Benchmark ASR services"""
    print("\n=== ASR Benchmark ===")
    print("Testing Whisper vs NeMo ASR...\n")
    
    results = []
    
    for audio_file in TEST_AUDIO_FILES:
        if not Path(audio_file).exists():
            print(f"Skipping {audio_file} - file not found")
            continue
        
        print(f"Testing with: {audio_file}")
        
        # Test Whisper
        whisper_result = await test_asr_service(WHISPER_URL, audio_file, "Whisper")
        results.append(whisper_result)
        if whisper_result["success"]:
            print(f"  Whisper: {whisper_result['processing_time']:.2f}s (RTF: {whisper_result['rtf']:.2f})")
        else:
            print(f"  Whisper: Error - {whisper_result['error']}")
        
        # Test NeMo
        nemo_result = await test_asr_service(NEMO_ASR_URL, audio_file, "NeMo")
        results.append(nemo_result)
        if nemo_result["success"]:
            print(f"  NeMo: {nemo_result['processing_time']:.2f}s (RTF: {nemo_result['rtf']:.2f})")
        else:
            print(f"  NeMo: Error - {nemo_result['error']}")
        
        print()
    
    # Summary statistics
    whisper_rtfs = [r["rtf"] for r in results if r["service"] == "Whisper" and r["success"]]
    nemo_rtfs = [r["rtf"] for r in results if r["service"] == "NeMo" and r["success"]]
    
    if whisper_rtfs:
        print(f"Whisper Average RTF: {statistics.mean(whisper_rtfs):.2f}")
    if nemo_rtfs:
        print(f"NeMo Average RTF: {statistics.mean(nemo_rtfs):.2f}")
    
    return results


async def benchmark_tts():
    """Benchmark TTS services"""
    print("\n=== TTS Benchmark ===")
    print("Testing Piper vs NeMo TTS...\n")
    
    results = []
    
    for text in TEST_TEXTS:
        print(f"Testing with {len(text)} character text...")
        
        # Test Piper
        piper_result = await test_tts_service(PIPER_URL, text, "Piper")
        results.append(piper_result)
        if piper_result["success"]:
            print(f"  Piper: {piper_result['processing_time']:.2f}s ({piper_result['chars_per_second']:.1f} chars/s)")
        else:
            print(f"  Piper: Error - {piper_result['error']}")
        
        # Test NeMo
        nemo_result = await test_tts_service(NEMO_TTS_URL, text, "NeMo")
        results.append(nemo_result)
        if nemo_result["success"]:
            print(f"  NeMo: {nemo_result['processing_time']:.2f}s ({nemo_result['chars_per_second']:.1f} chars/s)")
        else:
            print(f"  NeMo: Error - {nemo_result['error']}")
        
        print()
    
    # Summary statistics
    piper_times = [r["processing_time"] for r in results if r["service"] == "Piper" and r["success"]]
    nemo_times = [r["processing_time"] for r in results if r["service"] == "NeMo" and r["success"]]
    
    if piper_times:
        print(f"Piper Average Time: {statistics.mean(piper_times):.2f}s")
    if nemo_times:
        print(f"NeMo Average Time: {statistics.mean(nemo_times):.2f}s")
    
    return results


async def test_backend_switching():
    """Test backend switching via orchestrator"""
    print("\n=== Backend Switching Test ===")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Check current configuration
        response = await client.get(f"{ORCHESTRATOR_URL}/health")
        health = response.json()
        print(f"Current backends: ASR={health['backends']['asr']}, TTS={health['backends']['tts']}")
        
        print("\nTo switch backends, restart with environment variables:")
        print("  ASR_BACKEND=nemo TTS_BACKEND=nemo docker compose up -d")
        print("  ASR_BACKEND=whisper TTS_BACKEND=piper docker compose up -d")


async def main():
    """Run all benchmarks"""
    print("Voice Assistant Backend Comparison")
    print("==================================")
    
    # Wait for services to be ready
    print("Checking service health...")
    services = [
        ("Orchestrator", ORCHESTRATOR_URL),
        ("Whisper", WHISPER_URL),
        ("NeMo ASR", NEMO_ASR_URL),
        ("Piper", PIPER_URL),
        ("NeMo TTS", NEMO_TTS_URL)
    ]
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        for name, url in services:
            try:
                response = await client.get(f"{url}/health")
                if response.status_code == 200:
                    print(f"✓ {name} is healthy")
                else:
                    print(f"✗ {name} returned {response.status_code}")
            except Exception as e:
                print(f"✗ {name} is not available: {e}")
    
    # Run benchmarks
    await benchmark_asr()
    await benchmark_tts()
    await test_backend_switching()
    
    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    asyncio.run(main())