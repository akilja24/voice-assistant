#!/usr/bin/env python3
"""
Test script to verify NeMo can be imported and models can be loaded
"""
import sys

print("Testing NeMo setup...")
print("=" * 50)

# Test 1: Import NeMo
print("\n1. Testing NeMo imports...")
try:
    import nemo
    print("✓ NeMo imported successfully")
    print(f"  Version: {nemo.__version__ if hasattr(nemo, '__version__') else 'Unknown'}")
except ImportError as e:
    print(f"✗ Failed to import NeMo: {e}")
    sys.exit(1)

# Test 2: Import ASR collection
print("\n2. Testing ASR imports...")
try:
    import nemo.collections.asr as nemo_asr
    print("✓ NeMo ASR collection imported")
except ImportError as e:
    print(f"✗ Failed to import ASR collection: {e}")

# Test 3: Import TTS collection
print("\n3. Testing TTS imports...")
try:
    import nemo.collections.tts as nemo_tts
    from nemo.collections.tts.models import Tacotron2Model, HifiGanModel
    print("✓ NeMo TTS collection imported")
    print("✓ Tacotron2Model and HifiGanModel classes available")
except ImportError as e:
    print(f"✗ Failed to import TTS collection: {e}")

# Test 4: Check PyTorch
print("\n4. Testing PyTorch...")
try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ CUDA device: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"✗ Failed to import PyTorch: {e}")

# Test 5: Test model loading (dry run)
print("\n5. Testing model loading capabilities...")
try:
    # Just check if the model loading methods exist
    if hasattr(nemo_asr.models, 'EncDecCTCModel'):
        print("✓ ASR EncDecCTCModel class available")
    
    if hasattr(nemo_tts.models, 'Tacotron2Model'):
        print("✓ TTS Tacotron2Model class available")
    
    if hasattr(nemo_tts.models, 'HifiGanModel'):
        print("✓ TTS HifiGanModel class available")
    
    print("\nModel loading methods available:")
    print("  - ASR: nemo_asr.models.EncDecCTCModel.from_pretrained()")
    print("  - TTS: Tacotron2Model.from_pretrained()")
    print("  - Vocoder: HifiGanModel.from_pretrained()")
    
except Exception as e:
    print(f"✗ Error checking model classes: {e}")

print("\n" + "=" * 50)
print("Setup verification complete!")
print("\nTo test actual model loading, run with environment variable:")
print("  TEST_MODEL_LOAD=1 python test_nemo_setup.py")

# Optional: Actually try to load models
import os
if os.getenv("TEST_MODEL_LOAD") == "1":
    print("\n" + "=" * 50)
    print("TESTING ACTUAL MODEL LOADING...")
    
    print("\nAttempting to load QuartzNet15x5...")
    try:
        model = nemo_asr.models.EncDecCTCModel.from_pretrained("stt_en_quartznet15x5")
        print("✓ Successfully loaded QuartzNet15x5!")
        del model
    except Exception as e:
        print(f"✗ Failed to load QuartzNet15x5: {e}")
    
    print("\nAttempting to load Tacotron2...")
    try:
        model = Tacotron2Model.from_pretrained("tts_en_tacotron2")
        print("✓ Successfully loaded Tacotron2!")
        del model
    except Exception as e:
        print(f"✗ Failed to load Tacotron2: {e}")