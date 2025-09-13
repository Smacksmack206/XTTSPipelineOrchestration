#!/usr/bin/env python3

import subprocess
import sys
import os
from pathlib import Path

def check_setup():
    """Check if professional setup is complete"""
    required_files = [
        'professional_deepfake_engine.py',
        'requirements_professional.txt',
        'model_config.json',
        'models/opencv_face_detector_uint8.pb'
    ]
    
    missing = []
    for file in required_files:
        if not Path(file).exists():
            missing.append(file)
    
    if missing:
        print("❌ Missing required files:")
        for file in missing:
            print(f"   - {file}")
        print("\n🔧 Run setup first: python setup_professional.py")
        return False
    
    return True

def launch_app():
    """Launch the professional deepfake app"""
    if not check_setup():
        return
    
    print("🚀 Launching Professional Deepfake Engine...")
    print("📊 Features enabled:")
    print("   ✅ InsightFace detection")
    print("   ✅ 3D face alignment") 
    print("   ✅ Multi-band blending")
    print("   ✅ Advanced color matching")
    print("   ✅ Photorealistic enhancement")
    print("\n🌐 Opening browser interface...")
    
    try:
        subprocess.run([sys.executable, 'app_upgraded.py'], check=True)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except Exception as e:
        print(f"❌ Launch failed: {e}")

if __name__ == "__main__":
    launch_app()
