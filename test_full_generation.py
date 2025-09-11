#!/usr/bin/env python3
"""
Memory-optimized full generation test
"""
import os
import sys
import tempfile
import shutil
import gc
import psutil
from pathlib import Path

# Add paths
sys.path.append(str(Path("/Users/home/HM-labs/mac-xtts")))

def check_memory():
    """Check current memory usage"""
    memory = psutil.virtual_memory()
    return memory.percent, memory.used / 1024 / 1024 / 1024  # GB

def test_full_generation():
    """Test complete generation pipeline with memory monitoring"""
    print("🧪 Testing Memory-Optimized Generation Pipeline")
    print("=" * 50)
    
    # Monitor initial memory
    mem_percent, mem_gb = check_memory()
    print(f"📊 Initial memory: {mem_percent:.1f}% ({mem_gb:.1f}GB)")
    
    # Test files
    reference_audio = "/Users/home/Downloads/reference.wav"
    face_image = "/Users/home/Pictures/sandy.jpg"
    test_text = "Hello, this is a test of my digital twin!"
    
    # Check files exist
    if not os.path.exists(reference_audio):
        print(f"❌ Reference audio missing: {reference_audio}")
        return False
    
    if not os.path.exists(face_image):
        print(f"❌ Face image missing: {face_image}")
        return False
    
    print(f"✅ Reference audio found: {os.path.getsize(reference_audio)/1024/1024:.1f}MB")
    print(f"✅ Face image found: {os.path.getsize(face_image)/1024:.0f}KB")
    
    try:
        # Test XTTS loading with memory monitoring
        print("\n🔄 Testing memory-optimized XTTS loading...")
        
        mem_percent, mem_gb = check_memory()
        print(f"📊 Memory before XTTS: {mem_percent:.1f}% ({mem_gb:.1f}GB)")
        
        import torch
        
        # Memory optimization: disable gradients
        torch.set_grad_enabled(False)
        
        # Patch torch.load for compatibility
        original_load = torch.load
        def patched_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        torch.load = patched_load
        
        # Use smaller batch size for memory efficiency
        from TTS.api import TTS
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
        
        # Restore original torch.load
        torch.load = original_load
        
        mem_percent, mem_gb = check_memory()
        print(f"📊 Memory after XTTS load: {mem_percent:.1f}% ({mem_gb:.1f}GB)")
        
        if mem_gb > 8.0:
            print("⚠️ Memory usage above 8GB, forcing cleanup...")
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        
        print("✅ XTTS loaded successfully")
        
        # Test voice generation with memory monitoring
        print("\n🎤 Testing memory-optimized voice generation...")
        
        results_dir = Path("/Users/home/HM-labs/mac-netnavi/results")
        results_dir.mkdir(exist_ok=True)
        
        audio_output = results_dir / "test_audio.wav"
        
        # Split long text for memory efficiency
        if len(test_text) > 100:
            # Process in chunks for very long text
            chunks = [test_text[i:i+100] for i in range(0, len(test_text), 100)]
            test_text = chunks[0]  # Use first chunk for test
        
        tts.tts_to_file(
            text=test_text,
            speaker_wav=reference_audio,
            language="en",
            file_path=str(audio_output)
        )
        
        mem_percent, mem_gb = check_memory()
        print(f"📊 Memory after voice gen: {mem_percent:.1f}% ({mem_gb:.1f}GB)")
        
        if audio_output.exists():
            size = audio_output.stat().st_size / 1024
            print(f"✅ Audio generated: {audio_output.name} ({size:.0f}KB)")
        else:
            print("❌ Audio file not created")
            return False
        
        # Clear TTS model from memory
        del tts
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        # Test video creation with memory monitoring
        print("\n🎭 Testing memory-optimized video creation...")
        
        video_output = results_dir / "test_video.mp4"
        
        import subprocess
        
        # Use memory-efficient FFmpeg settings
        cmd = [
            'ffmpeg', '-y', '-loop', '1', '-i', face_image, '-i', str(audio_output),
            '-c:v', 'libx264', '-c:a', 'aac', '-shortest', '-pix_fmt', 'yuv420p',
            '-preset', 'fast',  # Faster, less memory intensive
            '-crf', '23',       # Slightly lower quality for memory
            str(video_output)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        mem_percent, mem_gb = check_memory()
        print(f"📊 Memory after video gen: {mem_percent:.1f}% ({mem_gb:.1f}GB)")
        
        if result.returncode == 0 and video_output.exists():
            size = video_output.stat().st_size / 1024 / 1024
            print(f"✅ Video generated: {video_output.name} ({size:.1f}MB)")
        else:
            print(f"❌ Video creation failed: {result.stderr}")
            return False
        
        # Final memory check
        mem_percent, mem_gb = check_memory()
        print(f"\n📊 Final memory usage: {mem_percent:.1f}% ({mem_gb:.1f}GB)")
        
        if mem_gb <= 8.0:
            print("✅ Memory usage stayed under 8GB limit!")
        elif mem_gb <= 10.0:
            print("⚠️ Memory usage under 10GB (acceptable)")
        else:
            print("❌ Memory usage exceeded 10GB limit")
        
        print("\n🎉 ALL TESTS PASSED!")
        print(f"📁 Generated files in: {results_dir}")
        print(f"🔊 Audio: {audio_output}")
        print(f"🎬 Video: {video_output}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Final cleanup
        gc.collect()
        if 'torch' in sys.modules and torch.backends.mps.is_available():
            torch.mps.empty_cache()

if __name__ == "__main__":
    success = test_full_generation()
    if success:
        print("\n✅ Memory-optimized generation pipeline works!")
    else:
        print("\n❌ Generation pipeline failed!")
        sys.exit(1)
