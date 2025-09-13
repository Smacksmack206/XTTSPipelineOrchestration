#!/usr/bin/env python3

import subprocess
import sys
import os
from pathlib import Path

def setup_m3_environment():
    """Setup M3-specific environment variables"""
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    os.environ['OMP_NUM_THREADS'] = '8'  # M3 has 8 performance cores
    os.environ['MKL_NUM_THREADS'] = '8'
    
    print("🔥 M3 environment configured")
    print("   • Metal Performance Shaders enabled")
    print("   • 8-core parallel processing")
    print("   • Unified memory optimization")
    print("   • Neural Engine ready")

def check_m3_requirements():
    """Check M3-specific requirements"""
    required_files = [
        'm3_optimized_engine.py',
        'professional_deepfake_engine.py',
        'requirements_m3.txt'
    ]
    
    missing = []
    for file in required_files:
        if not Path(file).exists():
            missing.append(file)
    
    if missing:
        print("❌ Missing M3 optimization files:")
        for file in missing:
            print(f"   - {file}")
        return False
    
    return True

def launch_m3_app():
    """Launch M3-optimized deepfake app"""
    if not check_m3_requirements():
        return
    
    setup_m3_environment()
    
    print("\n🚀 Launching M3-Optimized Deepfake Studio...")
    print("📊 M3 MacBook Air Features:")
    print("   ✅ Metal GPU acceleration")
    print("   ✅ Neural Engine (16-core)")
    print("   ✅ Unified memory optimization")
    print("   ✅ Thermal management")
    print("   ✅ Batch processing (8 cores)")
    print("   ✅ Smart quality presets")
    print("   ✅ Real-time preview")
    print("   ✅ Multiple format export")
    print("\n🌐 Opening browser interface...")
    
    try:
        subprocess.run([sys.executable, 'app_upgraded.py'], check=True)
    except KeyboardInterrupt:
        print("\n👋 M3 Deepfake Studio shutting down...")
    except Exception as e:
        print(f"❌ Launch failed: {e}")

if __name__ == "__main__":
    launch_m3_app()
