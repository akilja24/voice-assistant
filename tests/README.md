# Voice Assistant Tests

This directory contains test scripts and sample audio files for testing the voice assistant services.

## Audio Samples

The `audio_samples/` directory contains three 16kHz WAV files for testing:
- `Clear-Short_16k.wav` - Clear audio, short duration
- `Clear-Medium_16k.wav` - Clear audio, medium duration  
- `Noisy-Short_16k.wav` - Noisy audio, short duration

## Test Scripts

### test_websocket_client.py

Tests the WebSocket audio streaming endpoint of the orchestrator service.

#### Installation
```bash
pip install -r requirements.txt
```

#### Usage

Test with all sample files:
```bash
python test_websocket_client.py
```

Test with a specific file:
```bash
python test_websocket_client.py tests/audio_samples/Clear-Short_16k.wav
```

#### What it does:
1. Connects to the WebSocket endpoint at `ws://localhost:8000/ws/interact`
2. Reads the WAV file and sends PCM audio data in 64KB chunks
3. Sends an `end_stream` message to indicate completion
4. Receives and saves the response audio
5. Displays any metadata or error messages

#### Output:
- Response audio files are saved as `response_<original_name>.wav`
- Console output shows the progress and any messages received

## Running the Tests

1. First, start the orchestrator service:
```bash
docker compose up orchestrator
```

2. Run the test script:
```bash
cd /home/ubuntu/voice-assistant
python tests/test_websocket_client.py
```