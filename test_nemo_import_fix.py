#!/usr/bin/env python3
"""
Test if we can fix the NeMo import issue by patching huggingface_hub
"""

import sys

# Try to monkey-patch before importing NeMo
try:
    # Create a mock ModelFilter class
    class ModelFilter:
        def __init__(self, *args, **kwargs):
            pass
    
    # Try to patch huggingface_hub before NeMo imports it
    import huggingface_hub
    if not hasattr(huggingface_hub, 'ModelFilter'):
        print("Patching ModelFilter into huggingface_hub...")
        huggingface_hub.ModelFilter = ModelFilter
        # Also add it to the module's __all__ if it exists
        if hasattr(huggingface_hub, '__all__'):
            huggingface_hub.__all__.append('ModelFilter')
    
    print("✓ huggingface_hub patched successfully")
    
    # Now try importing NeMo
    print("\nTrying to import NeMo ASR...")
    import nemo.collections.asr as nemo_asr
    print("✓ NeMo ASR imported successfully!")
    
    print("\nTrying to import NeMo TTS...")
    import nemo.collections.tts as nemo_tts
    print("✓ NeMo TTS imported successfully!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\nTest complete.")