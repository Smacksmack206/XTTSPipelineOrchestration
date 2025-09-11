#!/usr/bin/env python3
"""
Test actual voice generation with your placeholder files
"""
import os
import sys
import torch
import torchaudio
from pathlib import Path

# Add XTTS path
sys.path.append(str(Path("/Users/home/HM-labs/mac-xtts")))

def test_voice_generation():
    """Test actual voice generation with your files"""
    print("🧪 Testing Voice Generation with Your Placeholders")
    print("=" * 60)
    
    # Your placeholder files
    reference_audio = "/Users/home/Downloads/reference.wav"
    test_text = "Hello, this is my digital twin speaking with realistic voice!"
    
    # Check if files exist
    if not os.path.exists(reference_audio):
        print(f"❌ Reference audio not found: {reference_audio}")
        return False
    
    try:
        # Import XTTS modules
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
        print("✅ XTTS modules imported")
        
        # Load your custom model
        model_path = Path("/Users/home/HM-labs/mac-xtts/finetune_models/ready")
        config_path = model_path / "config.json"
        checkpoint_path = model_path / "model.pth"
        vocab_path = model_path / "vocab.json"
        speakers_path = model_path / "speakers_xtts.pth"
        
        print("🔄 Loading XTTS model...")
        
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
        
        # Move to device
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        if device == 'mps':
            xtts_model.to('mps')
        print(f"✅ Model moved to {device}")
        
        print("🎤 Generating voice...")
        
        # Get conditioning latents
        gpt_cond_latent, speaker_embedding = xtts_model.get_conditioning_latents(
            audio_path=[reference_audio], 
            gpt_cond_len=xtts_model.config.gpt_cond_len, 
            max_ref_length=xtts_model.config.max_ref_len, 
            sound_norm_refs=xtts_model.config.sound_norm_refs
        )
        print("✅ Conditioning latents generated")
        
        # Generate audio with your settings
        out = xtts_model.inference(
            text=test_text,
            language="en",
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=0.05,
            top_k=1,
            top_p=1.0,
            repetition_penalty=1.05,
            length_penalty=1.0,
            speed=1.0,
            enable_text_splitting=True
        )
        print("✅ Audio generated")
        
        # Save test output
        output_path = "/Users/home/HM-labs/mac-netnavi/test_output.wav"
        torchaudio.save(
            output_path, 
            torch.tensor(out["wav"]).unsqueeze(0), 
            xtts_model.config.audio.sample_rate
        )
        
        # Check output file
        if os.path.exists(output_path):
            size = os.path.getsize(output_path) / 1024
            print(f"✅ Test audio saved: test_output.wav ({size:.0f}KB)")
            print(f"📁 Location: {output_path}")
            return True
        else:
            print("❌ Output file not created")
            return False
            
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_voice_generation()
    if success:
        print("\n🎉 SUCCESS! Voice generation works with your placeholders.")
        print("The app should work correctly now.")
    else:
        print("\n❌ FAILED! Check the errors above.")
        sys.exit(1)
