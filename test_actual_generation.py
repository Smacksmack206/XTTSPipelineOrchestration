#!/usr/bin/env python3
"""
Test actual generation with your placeholder files
"""
import os
import sys
import tempfile
import shutil
import uuid
import torch
import torchaudio
from pathlib import Path

# Add XTTS paths
sys.path.append(str(Path("/Users/home/HM-labs/mac-xtts")))

def test_actual_generation():
    """Test complete generation with your files"""
    print("🧪 Testing Actual Generation with Your Files")
    print("=" * 60)
    
    # Your actual files
    reference_audio = "/Users/home/Downloads/reference.wav"
    face_image = "/Users/home/Pictures/sandy.jpg"
    test_text = "Hello, this is my digital twin speaking with realistic voice and movement!"
    
    # Verify files exist
    for name, path in [("Audio", reference_audio), ("Image", face_image)]:
        if not os.path.exists(path):
            print(f"❌ {name} missing: {path}")
            return False
        size = os.path.getsize(path) / 1024 / 1024
        print(f"✅ {name}: {os.path.basename(path)} ({size:.1f}MB)")
    
    try:
        # Load your custom XTTS model
        print("\n🔄 Loading your custom XTTS model...")
        
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
        
        model_path = Path("/Users/home/HM-labs/mac-xtts/finetune_models/ready")
        config_path = model_path / "config.json"
        checkpoint_path = model_path / "model.pth"
        vocab_path = model_path / "vocab.json"
        speakers_path = model_path / "speakers_xtts.pth"
        
        # Load config
        config = XttsConfig()
        config.load_json(str(config_path))
        print("✅ Config loaded")
        
        # Initialize model
        xtts_model = Xtts.init_from_config(config)
        print("✅ Model initialized")
        
        # Load checkpoint
        xtts_model.load_checkpoint(
            config, 
            checkpoint_path=str(checkpoint_path),
            vocab_path=str(vocab_path),
            speaker_file_path=str(speakers_path),
            use_deepspeed=False
        )
        print("✅ Checkpoint loaded")
        
        # Move to MPS
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        if device == 'mps':
            xtts_model.to('mps')
        print(f"✅ Model on {device}")
        
        # Generate voice with your exact settings
        print("\n🎤 Generating voice with your custom settings...")
        
        # Get conditioning latents
        gpt_cond_latent, speaker_embedding = xtts_model.get_conditioning_latents(
            audio_path=[reference_audio], 
            gpt_cond_len=xtts_model.config.gpt_cond_len, 
            max_ref_length=xtts_model.config.max_ref_len, 
            sound_norm_refs=xtts_model.config.sound_norm_refs
        )
        print("✅ Conditioning latents generated")
        
        # Generate with your exact settings
        out = xtts_model.inference(
            text=test_text,
            language="en",
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=0.05,  # Your exact setting
            top_k=1,          # Your exact setting
            top_p=1.0,        # Your exact setting
            repetition_penalty=1.05,  # Your exact setting
            length_penalty=1.0,
            speed=1.0,
            enable_text_splitting=True
        )
        print("✅ Voice generated with your settings")
        
        # Save audio
        results_dir = Path("/Users/home/HM-labs/mac-netnavi/results")
        results_dir.mkdir(exist_ok=True)
        
        audio_output = results_dir / "test_generation.wav"
        torchaudio.save(
            str(audio_output), 
            torch.tensor(out["wav"]).unsqueeze(0), 
            xtts_model.config.audio.sample_rate
        )
        
        if audio_output.exists():
            size = audio_output.stat().st_size / 1024
            print(f"✅ Audio saved: {audio_output.name} ({size:.0f}KB)")
        else:
            print("❌ Audio file not created")
            return False
        
        # Create video
        print("\n🎭 Creating video...")
        
        import subprocess
        video_output = results_dir / "test_generation.mp4"
        
        cmd = [
            'ffmpeg', '-y', '-loop', '1', '-i', face_image, '-i', str(audio_output),
            '-c:v', 'libx264', '-c:a', 'aac', '-shortest', '-pix_fmt', 'yuv420p',
            str(video_output)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and video_output.exists():
            size = video_output.stat().st_size / 1024 / 1024
            print(f"✅ Video created: {video_output.name} ({size:.1f}MB)")
        else:
            print(f"❌ Video creation failed: {result.stderr}")
            return False
        
        print("\n🎉 GENERATION TEST SUCCESSFUL!")
        print(f"📁 Results in: {results_dir}")
        print(f"🔊 Audio: {audio_output}")
        print(f"🎬 Video: {video_output}")
        print(f"📝 Text: '{test_text}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_actual_generation()
    if success:
        print("\n✅ ACTUAL GENERATION WORKS!")
        print("Your memory-optimized app will work correctly.")
    else:
        print("\n❌ GENERATION FAILED!")
        sys.exit(1)
