#!/usr/bin/env python3
"""Quick test to verify app can start without errors"""

import sys
import os
from pathlib import Path

# Add paths
sys.path.append(str(Path("/Users/home/HM-labs/mac-xtts")))

try:
    # Test XTTS import
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    print("✅ XTTS imports successful")
    
    # Test model loading
    model_path = Path("/Users/home/HM-labs/mac-xtts/finetune_models/ready")
    config_path = model_path / "config.json"
    
    if config_path.exists():
        config = XttsConfig()
        config.load_json(str(config_path))
        print("✅ XTTS config loaded")
        
        # Test model initialization (without loading weights)
        xtts_model = Xtts.init_from_config(config)
        print("✅ XTTS model initialized")
        
    # Test placeholder files
    files = {
        "audio": "/Users/home/Downloads/reference.wav",
        "image": "/Users/home/Pictures/sandy.jpg", 
        "video": "/Users/home/Pictures/vid.mp4"
    }
    
    for name, path in files.items():
        if os.path.exists(path):
            print(f"✅ {name}: {os.path.basename(path)}")
        else:
            print(f"❌ {name}: {path} not found")
    
    print("\n🎉 App should start successfully!")
    print("Run: ./start_memory_optimized.sh")
    
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
