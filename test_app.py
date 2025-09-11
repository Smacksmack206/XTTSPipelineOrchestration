#!/usr/bin/env python3
"""
Test script for Mac NetNavi memory-optimized app
"""
import os
import sys
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported"""
    print("=== TESTING IMPORTS ===")
    try:
        import gradio as gr
        print("✅ gradio")
    except ImportError as e:
        print(f"❌ gradio: {e}")
        return False
    
    try:
        import psutil
        print("✅ psutil")
    except ImportError as e:
        print(f"❌ psutil: {e}")
        return False
    
    try:
        import torch
        print("✅ torch")
    except ImportError as e:
        print(f"❌ torch: {e}")
        return False
    
    try:
        import cv2
        print("✅ opencv-python")
    except ImportError as e:
        print(f"❌ opencv-python: {e}")
        return False
    
    try:
        import librosa
        print("✅ librosa")
    except ImportError as e:
        print(f"❌ librosa: {e}")
        return False
    
    try:
        import face_recognition
        print("✅ face-recognition")
    except ImportError as e:
        print(f"❌ face-recognition: {e}")
        return False
    
    return True

def test_files():
    """Test if all required files exist"""
    print("\n=== TESTING FILES ===")
    
    # Test placeholder files
    reference_audio = "/Users/home/Downloads/reference.wav"
    face_image = "/Users/home/Pictures/sandy.jpg"
    source_video = "/Users/home/Pictures/vid.mp4"
    
    files_ok = True
    
    if os.path.exists(reference_audio):
        size = os.path.getsize(reference_audio) / 1024 / 1024
        print(f"✅ Audio: reference.wav ({size:.1f}MB)")
    else:
        print("❌ Audio: reference.wav not found")
        files_ok = False
    
    if os.path.exists(face_image):
        size = os.path.getsize(face_image) / 1024
        print(f"✅ Image: sandy.jpg ({size:.0f}KB)")
    else:
        print("❌ Image: sandy.jpg not found")
        files_ok = False
    
    if os.path.exists(source_video):
        size = os.path.getsize(source_video) / 1024 / 1024
        print(f"✅ Video: vid.mp4 ({size:.1f}MB)")
    else:
        print("❌ Video: vid.mp4 not found")
        files_ok = False
    
    return files_ok

def test_xtts_model():
    """Test if XTTS model files exist"""
    print("\n=== TESTING XTTS MODEL ===")
    
    model_path = Path("/Users/home/HM-labs/mac-xtts/finetune_models/ready")
    if not model_path.exists():
        print("❌ XTTS model path not found")
        return False
    
    print("✅ XTTS model path exists")
    
    required_files = ["config.json", "model.pth", "vocab.json", "speakers_xtts.pth"]
    all_files_exist = True
    
    for file in required_files:
        if (model_path / file).exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file} missing")
            all_files_exist = False
    
    return all_files_exist

def test_memory():
    """Test memory management"""
    print("\n=== TESTING MEMORY ===")
    
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"✅ Total RAM: {memory.total / 1024 / 1024 / 1024:.1f}GB")
        print(f"✅ Available RAM: {memory.available / 1024 / 1024 / 1024:.1f}GB")
        print(f"✅ RAM Usage: {memory.percent:.1f}%")
        
        if memory.percent > 80:
            print("⚠️  High memory usage detected")
        
        return True
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Mac NetNavi App Test Suite")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 4
    
    if test_imports():
        tests_passed += 1
    
    if test_files():
        tests_passed += 1
    
    if test_xtts_model():
        tests_passed += 1
    
    if test_memory():
        tests_passed += 1
    
    print("\n" + "=" * 50)
    print(f"RESULTS: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("✅ All tests passed! App should work correctly.")
        print("\nRun the app with:")
        print("./start_memory_optimized.sh")
    else:
        print("❌ Some tests failed. Check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
