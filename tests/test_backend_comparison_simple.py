#!/usr/bin/env python3
"""
Benchmark existing Whisper/Piper services and demonstrate backend switching
"""
import asyncio
import httpx
import time
import json
import statistics
from pathlib import Path
import wave

# Service URLs
ORCHESTRATOR_URL = "http://localhost:8080"
WHISPER_URL = "http://localhost:8001"
PIPER_URL = "http://localhost:8003"

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


async def benchmark_whisper():
    """Benchmark Whisper ASR service"""
    print("\n=== Whisper ASR Benchmark ===")
    
    results = []
    
    for audio_file in TEST_AUDIO_FILES:
        if not Path(audio_file).exists():
            print(f"Skipping {audio_file} - file not found")
            continue
        
        print(f"\nTesting: {audio_file}")
        
        result = await test_asr_service(WHISPER_URL, audio_file, "Whisper")
        results.append(result)
        
        if result["success"]:
            print(f"  Processing time: {result['processing_time']:.2f}s")
            print(f"  Real-time factor (RTF): {result['rtf']:.3f}")
            print(f"  Transcription: {result['transcription'][:80]}...")
        else:
            print(f"  Error: {result['error']}")
    
    # Summary statistics
    successful = [r for r in results if r["success"]]
    if successful:
        avg_rtf = statistics.mean([r["rtf"] for r in successful])
        avg_time = statistics.mean([r["processing_time"] for r in successful])
        print(f"\n📊 Whisper Summary:")
        print(f"  Average RTF: {avg_rtf:.3f} (lower is better)")
        print(f"  Average processing time: {avg_time:.2f}s")
        print(f"  Success rate: {len(successful)}/{len(results)}")
    
    return results


async def benchmark_piper():
    """Benchmark Piper TTS service"""
    print("\n=== Piper TTS Benchmark ===")
    
    results = []
    
    for i, text in enumerate(TEST_TEXTS):
        print(f"\nTest {i+1}: {len(text)} characters")
        print(f"  Text: \"{text[:50]}...\"")
        
        result = await test_tts_service(PIPER_URL, text, "Piper")
        results.append(result)
        
        if result["success"]:
            print(f"  Processing time: {result['processing_time']:.2f}s")
            print(f"  Characters per second: {result['chars_per_second']:.1f}")
            print(f"  Audio duration: {result['audio_duration']:.2f}s")
            print(f"  Audio size: {result['audio_size_kb']:.1f} KB")
        else:
            print(f"  Error: {result['error']}")
    
    # Summary statistics
    successful = [r for r in results if r["success"]]
    if successful:
        avg_time = statistics.mean([r["processing_time"] for r in successful])
        avg_cps = statistics.mean([r["chars_per_second"] for r in successful])
        print(f"\n📊 Piper Summary:")
        print(f"  Average processing time: {avg_time:.2f}s")
        print(f"  Average chars/second: {avg_cps:.1f}")
        print(f"  Success rate: {len(successful)}/{len(results)}")
    
    return results


async def test_orchestrator_backends():
    """Test orchestrator backend configuration"""
    print("\n=== Orchestrator Backend Configuration ===")
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        # Check health and configuration
        response = await client.get(f"{ORCHESTRATOR_URL}/health")
        health = response.json()
        
        print(f"Status: {health.get('status', 'unknown')}")
        
        if 'backends' in health:
            print(f"\nActive Backends:")
            print(f"  ASR: {health['backends']['asr']}")
            print(f"  TTS: {health['backends']['tts']}")
            
            print(f"\nAvailable Services:")
            for service_type, services in health.get('services', {}).items():
                print(f"  {service_type}:")
                if isinstance(services, dict):
                    for name, url in services.items():
                        print(f"    - {name}: {url}")
                else:
                    print(f"    {services}")
        else:
            print("\n⚠️  Backend switching not available in current orchestrator")
            print("Orchestrator health response:", json.dumps(health, indent=2))
        
        print("\n💡 To switch backends (when NeMo services are available):")
        print("  export ASR_BACKEND=nemo")
        print("  export TTS_BACKEND=nemo")
        print("  docker compose up -d")


async def main():
    """Run all benchmarks"""
    print("🎤 Voice Assistant Service Benchmark")
    print("=" * 50)
    
    # Check service health
    print("\n📋 Checking service health...")
    services = [
        ("Orchestrator", ORCHESTRATOR_URL),
        ("Whisper ASR", WHISPER_URL),
        ("Piper TTS", PIPER_URL)
    ]
    
    async with httpx.AsyncClient(timeout=5.0) as client:
        for name, url in services:
            try:
                response = await client.get(f"{url}/health")
                if response.status_code == 200:
                    print(f"  ✅ {name} is healthy")
                else:
                    print(f"  ❌ {name} returned {response.status_code}")
            except Exception as e:
                print(f"  ❌ {name} is not available: {type(e).__name__}")
    
    # Run benchmarks
    whisper_results = await benchmark_whisper()
    piper_results = await benchmark_piper()
    await test_orchestrator_backends()
    
    # Performance comparison placeholder
    print("\n=== Performance Comparison ===")
    print("\n📊 Current Services Performance:")
    print("┌─────────────┬─────────────┬──────────────┬─────────────┐")
    print("│ Service     │ Metric      │ Value        │ Notes       │")
    print("├─────────────┼─────────────┼──────────────┼─────────────┤")
    
    # Whisper stats
    if whisper_results:
        successful = [r for r in whisper_results if r["success"]]
        if successful:
            avg_rtf = statistics.mean([r["rtf"] for r in successful])
            print(f"│ Whisper ASR │ Avg RTF     │ {avg_rtf:>12.3f} │ GPU accel.  │")
    
    # Piper stats
    if piper_results:
        successful = [r for r in piper_results if r["success"]]
        if successful:
            avg_cps = statistics.mean([r["chars_per_second"] for r in successful])
            print(f"│ Piper TTS   │ Chars/sec   │ {avg_cps:>12.1f} │ CPU only    │")
    
    print("└─────────────┴─────────────┴──────────────┴─────────────┘")
    
    print("\n📝 When NeMo services are available, this script will compare:")
    print("  • Whisper vs NeMo ASR (accuracy, speed, GPU usage)")
    print("  • Piper vs NeMo TTS (quality, speed, voice options)")
    print("  • Backend switching latency")
    print("  • Resource utilization")
    
    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    asyncio.run(main())