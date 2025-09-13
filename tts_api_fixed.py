#!/usr/bin/env python3
"""
Fixed TTS API with optimized XTTS settings to eliminate scratching
"""

import os
import sys
import time
import uuid
import shutil
import tempfile
import subprocess
import torch
import torchaudio
from pathlib import Path
from flask import Flask, request, jsonify
import psutil
import gc
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class MemoryManager:
    def __init__(self):
        self.operation_count = 0
        self.tts_model = None
        
    def check_memory(self):
        memory = psutil.virtual_memory()
        return memory.percent / 100.0
    
    def cleanup(self):
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if hasattr(self, 'tts_model') and self.tts_model:
            del self.tts_model
            self.tts_model = None
        self.operation_count = 0

memory_manager = MemoryManager()

def normalize_and_filter_audio(audio_tensor, sample_rate):
    """
    Normalize and filter audio to eliminate scratching and improve quality
    """
    try:
        import torchaudio.transforms as T
        
        # Convert to float32 for processing
        audio = audio_tensor.float()
        
        # 1. Normalize volume to prevent clipping
        max_val = torch.max(torch.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.95  # Leave some headroom
        
        # 2. Apply high-pass filter to remove low-frequency noise
        highpass = T.Highpass(sample_rate=sample_rate, cutoff_freq=80.0)
        audio = highpass(audio)
        
        # 3. Apply low-pass filter to remove high-frequency artifacts/scratching
        lowpass = T.Lowpass(sample_rate=sample_rate, cutoff_freq=8000.0)
        audio = lowpass(audio)
        
        # 4. Apply gentle compression to smooth dynamics
        # Simple soft limiting
        audio = torch.tanh(audio * 1.2) * 0.8
        
        # 5. Final normalization
        max_val = torch.max(torch.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.9
        
        return audio
        
    except Exception as e:
        logger.warning(f"Audio filtering failed, using raw audio: {e}")
        # Fallback: just normalize
        max_val = torch.max(torch.abs(audio_tensor))
        if max_val > 0:
            return audio_tensor / max_val * 0.9
        return audio_tensor

def setup_xtts():
    """Setup XTTS model with fallback support"""
    try:
        # Try to import TTS
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
        
        # Check for custom model first
        custom_model_paths = [
            Path("/Users/home/HM-labs/mac-xtts/finetune_models/ready"),
            Path("models/XTTS-v2"),
            Path("models/xtts")
        ]
        
        for model_path in custom_model_paths:
            if model_path.exists():
                config_path = model_path / "config.json"
                checkpoint_path = model_path / "model.pth"
                vocab_path = model_path / "vocab.json"
                speakers_path = model_path / "speakers_xtts.pth"
                
                if all(p.exists() for p in [config_path, checkpoint_path, vocab_path, speakers_path]):
                    logger.info(f"Loading custom XTTS model from {model_path}")
                    
                    config = XttsConfig()
                    config.load_json(str(config_path))
                    xtts_model = Xtts.init_from_config(config)
                    
                    xtts_model.load_checkpoint(
                        config, 
                        checkpoint_path=str(checkpoint_path),
                        vocab_path=str(vocab_path),
                        speaker_file_path=str(speakers_path),
                        use_deepspeed=False
                    )
                    
                    logger.info("✅ Custom XTTS model loaded successfully")
                    return xtts_model
        
        # Fallback to default TTS model
        logger.info("Custom XTTS not found, trying default TTS model...")
        from TTS.api import TTS
        
        # Use a lightweight model for fallback
        tts = TTS("tts_models/en/ljspeech/tacotron2-DDC", gpu=False)
        logger.info("✅ Fallback TTS model loaded")
        return tts
        
    except ImportError:
        logger.warning("TTS library not available")
        return None
    except Exception as e:
        logger.error(f"TTS setup failed: {e}")
        return None

def preprocess_reference_audio(speaker_wav: str) -> str:
    """
    Preprocess reference audio for better voice cloning
    """
    try:
        workdir = tempfile.mkdtemp()
        processed_wav = os.path.join(workdir, "processed_reference.wav")
        
        # Use FFmpeg to normalize and clean the reference audio
        cmd = [
            'ffmpeg', '-y', '-i', speaker_wav,
            '-ar', '22050',  # Standard sample rate for XTTS
            '-ac', '1',      # Mono
            '-af', 'highpass=f=80,lowpass=f=8000,dynaudnorm=f=75:g=25:p=0.95',  # Clean audio
            '-c:a', 'pcm_s16le',  # Uncompressed format
            processed_wav
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and os.path.exists(processed_wav):
            return processed_wav
        else:
            logger.warning("Reference audio preprocessing failed, using original")
            return speaker_wav
            
    except Exception as e:
        logger.warning(f"Reference audio preprocessing error: {e}")
        return speaker_wav

# Global XTTS settings that can be adjusted
XTTS_SETTINGS = {
    "temperature": 0.75,
    "length_penalty": 1.0,
    "repetition_penalty": 5.0,
    "top_k": 50,
    "top_p": 0.85,
    "speed": 1.0,
    "gpt_cond_len": 30,
    "max_ref_length": 60
}

def clone_voice_with_xtts(text: str, speaker_wav: str, output_path: str, tts_model) -> str:
    """Clone voice using XTTS with optimized settings"""
    if memory_manager.check_memory() > 0.6:
        memory_manager.cleanup()
    
    try:
        # Check if it's the custom XTTS model
        if hasattr(tts_model, 'inference'):
            # Custom XTTS model
            device = 'mps' if torch.backends.mps.is_available() else 'cpu'
            if device == 'mps':
                tts_model.to('mps')
            
            try:
                # Preprocess reference audio for better quality
                processed_speaker_wav = preprocess_reference_audio(speaker_wav)
                
                # Enhanced conditioning for better voice matching
                gpt_cond_latent, speaker_embedding = tts_model.get_conditioning_latents(
                    audio_path=[processed_speaker_wav], 
                    gpt_cond_len=XTTS_SETTINGS["gpt_cond_len"],
                    max_ref_length=XTTS_SETTINGS["max_ref_length"],
                    sound_norm_refs=True  # Always normalize reference audio
                )
                
                # Use configurable settings for optimal voice quality
                out = tts_model.inference(
                    text=text,
                    language="en",
                    gpt_cond_latent=gpt_cond_latent,
                    speaker_embedding=speaker_embedding,
                    temperature=XTTS_SETTINGS["temperature"],
                    length_penalty=XTTS_SETTINGS["length_penalty"],
                    repetition_penalty=XTTS_SETTINGS["repetition_penalty"],
                    top_k=XTTS_SETTINGS["top_k"],
                    top_p=XTTS_SETTINGS["top_p"],
                    speed=XTTS_SETTINGS["speed"],
                    enable_text_splitting=True,
                    # Additional quality settings
                    do_sample=True,  # Enable sampling for natural speech
                    num_beams=1,  # Single beam to avoid artifacts
                )
                
                # Post-process audio to eliminate scratching and improve quality
                audio_tensor = torch.tensor(out["wav"]).unsqueeze(0)
                
                # Apply audio normalization and filtering
                audio_tensor = normalize_and_filter_audio(audio_tensor, tts_model.config.audio.sample_rate)
                
                torchaudio.save(
                    output_path, 
                    audio_tensor, 
                    tts_model.config.audio.sample_rate
                )
                
            finally:
                tts_model.to('cpu')
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                gc.collect()
        
        elif hasattr(tts_model, 'tts_to_file'):
            # Fallback TTS model
            tts_model.tts_to_file(text=text, file_path=output_path)
        
        else:
            # No TTS available, create fallback
            return create_fallback_audio(text, output_path)
        
        return output_path
        
    except Exception as e:
        logger.error(f"Voice cloning failed: {e}")
        return create_fallback_audio(text, output_path)

def create_fallback_audio(text: str, output_path: str) -> str:
    """Create fallback audio when TTS fails"""
    try:
        # Create a simple tone audio file
        duration = max(3, len(text.split()) * 0.5)  # Estimate duration
        cmd = [
            'ffmpeg', '-y', '-f', 'lavfi', 
            '-i', f'sine=frequency=440:duration={duration}',
            '-ar', '22050', '-ac', '1',
            output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info(f"Created fallback audio: {duration}s")
        return output_path
    except Exception as e:
        logger.error(f"Fallback audio creation failed: {e}")
        return None

# Initialize TTS model
TTS_MODEL = setup_xtts()

@app.route("/tts", methods=["POST"])
def tts():
    """TTS endpoint with error handling"""
    try:
        data = request.get_json()
        text = data.get("text", "")
        speaker_wav = data.get("speaker_wav", "")
        
        if not text:
            return jsonify({"error": "text is required"}), 400

        workdir = tempfile.mkdtemp()
        uid = uuid.uuid4().hex
        audio_output = os.path.join(workdir, f"audio_{uid}.wav")

        if TTS_MODEL and speaker_wav and os.path.exists(speaker_wav):
            # Use TTS model
            output_path = clone_voice_with_xtts(text, speaker_wav, audio_output, TTS_MODEL)
        else:
            # Fallback mode
            output_path = create_fallback_audio(text, audio_output)
        
        if output_path and os.path.exists(output_path):
            return jsonify({"audio_path": output_path})
        else:
            return jsonify({"error": "Audio generation failed"}), 500
            
    except Exception as e:
        logger.error(f"TTS endpoint error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/settings", methods=["GET", "POST"])
def settings():
    """Get or update XTTS settings"""
    global XTTS_SETTINGS
    
    if request.method == "GET":
        return jsonify(XTTS_SETTINGS)
    
    elif request.method == "POST":
        try:
            new_settings = request.get_json()
            XTTS_SETTINGS.update(new_settings)
            logger.info(f"Updated XTTS settings: {XTTS_SETTINGS}")
            return jsonify({"status": "updated", "settings": XTTS_SETTINGS})
        except Exception as e:
            return jsonify({"error": str(e)}), 400

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    status = {
        "status": "healthy",
        "tts_model": "available" if TTS_MODEL else "fallback",
        "memory_usage": f"{psutil.virtual_memory().percent:.1f}%",
        "current_settings": XTTS_SETTINGS
    }
    return jsonify(status)

if __name__ == "__main__":
    logger.info("🎤 Starting TTS API server...")
    logger.info(f"TTS Model Status: {'✅ Available' if TTS_MODEL else '🟡 Fallback mode'}")
    
    app.run(
        host="0.0.0.0",
        port=7864,
        debug=False,
        threaded=True
    )