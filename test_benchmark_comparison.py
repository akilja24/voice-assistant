#!/usr/bin/env python3
"""
Benchmark comparison between Whisper/Piper and NeMo ASR/TTS services
"""

import requests
import time
import json
import os
import statistics
from typing import Dict, List, Tuple

# Service URLs
WHISPER_URL = "http://localhost:8001/transcribe"
NEMO_ASR_URL = "http://localhost:8005/transcribe"
PIPER_TTS_URL = "http://localhost:8003/speak"
NEMO_TTS_URL = "http://localhost:8006/speak"

# Test configurations
TEST_AUDIO_DIR = "tests/audio_samples"
TEST_TEXTS = [
    "Hello, this is a test of the text to speech system.",
    "The quick brown fox jumps over the lazy dog.",
    "I want to learn more about quantum computing. Can you explain to me the basics?",
    "This is a longer sentence to test the performance of the text to speech synthesis engine with more complex content and multiple clauses.",
]

def benchmark_asr(audio_file: str, service_url: str, service_name: str, runs: int = 3) -> Dict:
    """Benchmark ASR service"""
    print(f"\nBenchmarking {service_name} with {audio_file}...")
    
    times = []
    results = []
    
    for i in range(runs):
        with open(audio_file, 'rb') as f:
            files = {'audio': (os.path.basename(audio_file), f, 'audio/wav')}
            
            start_time = time.time()
            try:
                response = requests.post(service_url, files=files, timeout=30)
                elapsed = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    times.append(elapsed)
                    results.append(result)
                    print(f"  Run {i+1}: {elapsed:.3f}s - Text: {result.get('text', '')[:50]}...")
                else:
                    print(f"  Run {i+1}: Failed - Status: {response.status_code}")
            except Exception as e:
                print(f"  Run {i+1}: Error - {str(e)}")
    
    if times:
        return {
            "service": service_name,
            "audio_file": os.path.basename(audio_file),
            "runs": len(times),
            "avg_time": statistics.mean(times),
            "min_time": min(times),
            "max_time": max(times),
            "std_dev": statistics.stdev(times) if len(times) > 1 else 0,
            "sample_result": results[0] if results else None
        }
    else:
        return {
            "service": service_name,
            "audio_file": os.path.basename(audio_file),
            "error": "All runs failed"
        }

def benchmark_tts(text: str, service_url: str, service_name: str, runs: int = 3) -> Dict:
    """Benchmark TTS service"""
    print(f"\nBenchmarking {service_name} with text: '{text[:50]}...'")
    
    times = []
    audio_sizes = []
    
    for i in range(runs):
        payload = {"text": text}
        headers = {"Content-Type": "application/json"}
        
        start_time = time.time()
        try:
            response = requests.post(service_url, json=payload, headers=headers, timeout=30)
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                audio_size = len(response.content)
                times.append(elapsed)
                audio_sizes.append(audio_size)
                print(f"  Run {i+1}: {elapsed:.3f}s - Audio size: {audio_size/1024:.1f} KB")
            else:
                print(f"  Run {i+1}: Failed - Status: {response.status_code}")
        except Exception as e:
            print(f"  Run {i+1}: Error - {str(e)}")
    
    if times:
        return {
            "service": service_name,
            "text_length": len(text),
            "runs": len(times),
            "avg_time": statistics.mean(times),
            "min_time": min(times),
            "max_time": max(times),
            "std_dev": statistics.stdev(times) if len(times) > 1 else 0,
            "avg_audio_size_kb": statistics.mean(audio_sizes) / 1024
        }
    else:
        return {
            "service": service_name,
            "text_length": len(text),
            "error": "All runs failed"
        }

def main():
    print("=" * 80)
    print("Voice Assistant Service Benchmark Comparison")
    print("Whisper/Piper vs NeMo ASR/TTS")
    print("=" * 80)
    
    # Check services are healthy
    print("\nChecking service health...")
    services = [
        ("Whisper", "http://localhost:8001/health"),
        ("NeMo ASR", "http://localhost:8005/health"),
        ("Piper TTS", "http://localhost:8003/health"),
        ("NeMo TTS", "http://localhost:8006/health"),
    ]
    
    all_healthy = True
    for name, url in services:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✓ {name} is healthy")
            else:
                print(f"✗ {name} returned status {response.status_code}")
                all_healthy = False
        except Exception as e:
            print(f"✗ {name} is not responding: {str(e)}")
            all_healthy = False
    
    if not all_healthy:
        print("\nSome services are not healthy. Please ensure all services are running.")
        return
    
    # ASR Benchmarks
    print("\n" + "=" * 40)
    print("ASR BENCHMARKS")
    print("=" * 40)
    
    asr_results = []
    test_files = [
        os.path.join(TEST_AUDIO_DIR, "Clear-Short_16k.wav"),
        os.path.join(TEST_AUDIO_DIR, "Clear-Medium_16k.wav"),
        os.path.join(TEST_AUDIO_DIR, "Noisy-Short_16k.wav"),
    ]
    
    for audio_file in test_files:
        if os.path.exists(audio_file):
            # Test Whisper
            result = benchmark_asr(audio_file, WHISPER_URL, "Whisper", runs=3)
            asr_results.append(result)
            
            # Test NeMo ASR
            result = benchmark_asr(audio_file, NEMO_ASR_URL, "NeMo ASR", runs=3)
            asr_results.append(result)
        else:
            print(f"Warning: {audio_file} not found")
    
    # TTS Benchmarks
    print("\n" + "=" * 40)
    print("TTS BENCHMARKS")
    print("=" * 40)
    
    tts_results = []
    for text in TEST_TEXTS:
        # Test Piper
        result = benchmark_tts(text, PIPER_TTS_URL, "Piper TTS", runs=3)
        tts_results.append(result)
        
        # Test NeMo TTS
        result = benchmark_tts(text, NEMO_TTS_URL, "NeMo TTS", runs=3)
        tts_results.append(result)
    
    # Summary
    print("\n" + "=" * 40)
    print("SUMMARY")
    print("=" * 40)
    
    # ASR Summary
    print("\nASR Performance Summary:")
    print("-" * 60)
    print(f"{'Service':<15} {'Audio File':<20} {'Avg Time (s)':<12} {'Speedup':<10}")
    print("-" * 60)
    
    for i in range(0, len(asr_results), 2):
        if i+1 < len(asr_results):
            whisper = asr_results[i]
            nemo = asr_results[i+1]
            if 'avg_time' in whisper and 'avg_time' in nemo:
                speedup = whisper['avg_time'] / nemo['avg_time']
                print(f"{whisper['service']:<15} {whisper['audio_file']:<20} {whisper['avg_time']:.3f} {'':<10}")
                print(f"{nemo['service']:<15} {nemo['audio_file']:<20} {nemo['avg_time']:.3f} {speedup:.2f}x")
                print()
    
    # TTS Summary
    print("\nTTS Performance Summary:")
    print("-" * 70)
    print(f"{'Service':<15} {'Text Length':<15} {'Avg Time (s)':<12} {'Audio Size (KB)':<15} {'Speedup':<10}")
    print("-" * 70)
    
    for i in range(0, len(tts_results), 2):
        if i+1 < len(tts_results):
            piper = tts_results[i]
            nemo = tts_results[i+1]
            if 'avg_time' in piper and 'avg_time' in nemo:
                speedup = piper['avg_time'] / nemo['avg_time']
                print(f"{piper['service']:<15} {piper['text_length']:<15} {piper['avg_time']:.3f} {piper['avg_audio_size_kb']:.1f} {'':<15}")
                print(f"{nemo['service']:<15} {nemo['text_length']:<15} {nemo['avg_time']:.3f} {nemo['avg_audio_size_kb']:.1f} {speedup:.2f}x")
                print()
    
    # Save results
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "asr_results": asr_results,
        "tts_results": tts_results
    }
    
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to benchmark_results.json")

if __name__ == "__main__":
    main()