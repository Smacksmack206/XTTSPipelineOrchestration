import os
import sys
import time
import uuid
import shutil
import subprocess
import tempfile
import requests
import gradio as gr
import psutil
import gc
import torch
import torchaudio
import cv2
import numpy as np
import librosa
import face_recognition
from pathlib import Path

# Add XTTS paths
parent_dir = Path(__file__).parent.parent
xtts_dir = parent_dir / "xtts-finetune-webui"
mac_xtts_dir = Path("/Users/home/HM-labs/mac-xtts")
sys.path.append(str(xtts_dir))
sys.path.append(str(mac_xtts_dir))

# Memory management settings for 8GB limit on M3 MacBook Air
RAM_THRESHOLD = 0.50  # 50% RAM usage threshold (4GB max)
CLEANUP_INTERVAL = 1  # Clean up after every operation
MAX_VIDEO_FRAMES = 100  # Limit frames in memory

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
        # Force cleanup of large objects
        if hasattr(self, 'tts_model') and self.tts_model:
            del self.tts_model
            self.tts_model = None
        self.operation_count = 0
    
    def should_cleanup(self):
        self.operation_count += 1
        return (self.operation_count % CLEANUP_INTERVAL == 0 or 
                self.check_memory() > RAM_THRESHOLD)
    
    def get_tts_model(self):
        """Lazy load TTS model only when needed"""
        if self.tts_model is None:
            self.tts_model = self._load_tts_model()
        return self.tts_model
    
    def _load_tts_model(self):
        """Load TTS model with memory optimization"""
        try:
            from TTS.api import TTS
            # Use smaller model for memory efficiency
            model = TTS("tts_models/en/ljspeech/tacotron2-DDC", gpu=False)
            return model
        except:
            return None

memory_manager = MemoryManager()

def setup_xtts():
    """Load your custom fine-tuned XTTS model with memory management"""
    try:
        if memory_manager.check_memory() > RAM_THRESHOLD:
            memory_manager.cleanup()
        
        # Import XTTS modules
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
        
        # Load your custom model
        model_path = Path("/Users/home/HM-labs/mac-xtts/finetune_models/ready")
        config_path = model_path / "config.json"
        checkpoint_path = model_path / "model.pth"
        vocab_path = model_path / "vocab.json"
        speakers_path = model_path / "speakers_xtts.pth"
        
        if not all(p.exists() for p in [config_path, checkpoint_path, vocab_path, speakers_path]):
            raise Exception(f"Custom XTTS model files missing in {model_path}")
        
        print("🔄 Loading custom XTTS model...")
        
        # Load config
        config = XttsConfig()
        config.load_json(str(config_path))
        
        # Initialize model with memory optimization
        xtts_model = Xtts.init_from_config(config)
        
        # Load checkpoint with memory management
        xtts_model.load_checkpoint(
            config, 
            checkpoint_path=str(checkpoint_path),
            vocab_path=str(vocab_path),
            speaker_file_path=str(speakers_path),
            use_deepspeed=False
        )
        
        # Use CPU to save GPU memory, only move to MPS when needed
        device = 'cpu'  # Keep on CPU by default
        
        print("✅ Custom XTTS model loaded with memory optimization")
        return xtts_model
        
    except Exception as e:
        print(f"❌ Custom XTTS failed: {e}")
        return None

def get_ai_response(text: str) -> str:
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2", "prompt": text, "stream": False},
            timeout=30
        )
        return response.json()["response"]
    except:
        return "AI unavailable - using input text"

def clone_voice_with_custom_settings(text: str, speaker_wav: str, output_path: str, tts_model) -> str:
    """Clone voice using your exact custom XTTS settings with memory optimization"""
    if memory_manager.check_memory() > RAM_THRESHOLD:
        memory_manager.cleanup()
    
    # Temporarily move to MPS only for inference
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    if device == 'mps':
        tts_model.to('mps')
    
    try:
        # Get conditioning latents
        gpt_cond_latent, speaker_embedding = tts_model.get_conditioning_latents(
            audio_path=[speaker_wav], 
            gpt_cond_len=tts_model.config.gpt_cond_len, 
            max_ref_length=tts_model.config.max_ref_len, 
            sound_norm_refs=tts_model.config.sound_norm_refs
        )
        
        # Generate with your exact custom settings
        out = tts_model.inference(
            text=text,
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
        
        # Save audio
        torchaudio.save(
            output_path, 
            torch.tensor(out["wav"]).unsqueeze(0), 
            tts_model.config.audio.sample_rate
        )
        
    finally:
        # Move back to CPU to free GPU memory
        tts_model.to('cpu')
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()
    
    return output_path

def apply_realistic_mouth_movement(frame, amplitude):
    """Apply realistic mouth movement for deepfake"""
    try:
        face_locations = face_recognition.face_locations(frame)
        
        if face_locations:
            top, right, bottom, left = face_locations[0]
            
            # More precise mouth region detection
            mouth_y = top + int((bottom - top) * 0.65)
            mouth_h = int((bottom - top) * 0.35)
            mouth_region = frame[mouth_y:mouth_y + mouth_h, left:right]
            
            if mouth_region.size > 0:
                # Enhanced movement based on audio amplitude
                movement = int(amplitude * 50)  # Increased sensitivity
                if movement > 3:  # More responsive threshold
                    new_h = mouth_h + movement
                    stretched = cv2.resize(mouth_region, (right - left, new_h))
                    end_y = min(mouth_y + new_h, frame.shape[0])
                    blend_h = end_y - mouth_y
                    
                    # Smooth blending for realistic effect
                    alpha = 0.8
                    frame[mouth_y:end_y, left:right] = cv2.addWeighted(
                        frame[mouth_y:end_y, left:right], 1-alpha,
                        stretched[:blend_h, :], alpha, 0
                    )
        
        return frame
    except:
        return frame

def create_realistic_deepfake_video(image_path: str, audio_path: str, source_video: str = None) -> str:
    """Create realistic deepfake video with memory-optimized processing"""
    output_path = Path(__file__).parent / "results" / f"deepfake_{uuid.uuid4().hex[:8]}.mp4"
    output_path.parent.mkdir(exist_ok=True)
    
    if source_video and os.path.exists(source_video):
        # Memory-optimized deepfake processing
        print("🎭 Creating memory-optimized deepfake...")
        
        audio, sr = librosa.load(audio_path, sr=22050)
        duration = len(audio) / sr
        fps = 25
        total_frames = int(duration * fps)
        
        # Process video in chunks to manage memory
        cap = cv2.VideoCapture(source_video)
        temp_video = str(output_path).replace('.mp4', '_temp.mp4')
        
        # Get video properties
        ret, first_frame = cap.read()
        if not ret:
            cap.release()
            raise Exception("Cannot read source video")
        
        height, width = first_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
        
        # Reset video to beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Process frames in small batches
        batch_size = 25  # Process 1 second at a time
        frame_count = 0
        
        while frame_count < total_frames:
            batch_frames = []
            
            # Read batch of frames
            for _ in range(min(batch_size, total_frames - frame_count)):
                ret, frame = cap.read()
                if not ret:
                    # Loop source video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        break
                
                # Calculate audio amplitude for this frame
                time_pos = frame_count / fps
                audio_start = int(time_pos * sr)
                audio_end = int((time_pos + 1/fps) * sr)
                frame_audio = audio[audio_start:min(audio_end, len(audio))]
                amplitude = np.mean(np.abs(frame_audio)) if len(frame_audio) > 0 else 0
                
                # Apply mouth movement
                animated_frame = apply_realistic_mouth_movement(frame, amplitude)
                batch_frames.append(animated_frame)
                frame_count += 1
            
            # Write batch to video
            for frame in batch_frames:
                out.write(frame)
            
            # Clear batch from memory
            del batch_frames
            gc.collect()
            
            # Memory check
            if memory_manager.check_memory() > RAM_THRESHOLD:
                memory_manager.cleanup()
        
        cap.release()
        out.release()
        
        # Add audio with memory-efficient encoding
        cmd = [
            'ffmpeg', '-y', '-i', temp_video, '-i', audio_path,
            '-c:v', 'libx264', '-c:a', 'aac', '-shortest',
            '-pix_fmt', 'yuv420p', '-crf', '23',  # Slightly lower quality for memory
            '-preset', 'fast',  # Faster encoding, less memory
            str(output_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        
        if os.path.exists(temp_video):
            os.remove(temp_video)
    else:
        # Static image processing
        cmd = [
            'ffmpeg', '-y', '-loop', '1', '-i', image_path, '-i', audio_path,
            '-c:v', 'libx264', '-c:a', 'aac', '-shortest', '-pix_fmt', 'yuv420p',
            '-preset', 'fast', str(output_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    
    return str(output_path)

def pipeline(user_text, voice_file, image_file, source_video, use_ai, history, progress=gr.Progress()):
    progress(0.1, desc="🔄 Starting...")
    yield ("🔄 Starting...", None, None, history)
    
    # Memory check before starting
    if memory_manager.check_memory() > RAM_THRESHOLD:
        memory_manager.cleanup()
        yield ("⚠️ Memory optimized, continuing...", None, None, history)
    
    tts_model = setup_xtts()
    if not tts_model:
        yield ("❌ Custom XTTS failed to load", None, None, history)
        return
    
    workdir = tempfile.mkdtemp()
    uid = uuid.uuid4().hex
    
    try:
        src_wav = os.path.join(workdir, f"{uid}_src.wav")
        img_in = os.path.join(workdir, f"{uid}_img{Path(image_file.name).suffix}")
        
        shutil.copy(voice_file.name, src_wav)
        shutil.copy(image_file.name, img_in)
        
        progress(0.3, desc="🧠 Processing text...")
        yield ("🧠 Processing text...", None, None, history)
        
        text = get_ai_response(user_text) if use_ai else user_text
        
        progress(0.6, desc="🎤 Cloning voice with custom settings...")
        yield (text, None, None, history)
        
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        audio_output = results_dir / f"audio_{uid}.wav"
        
        clone_voice_with_custom_settings(text, src_wav, str(audio_output), tts_model)
        
        progress(0.9, desc="🎭 Creating realistic deepfake...")
        yield (text, str(audio_output), None, history)
        
        video_output = create_realistic_deepfake_video(img_in, str(audio_output), source_video)
        
        history.append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "input": user_text,
            "response": text,
            "audio": str(audio_output),
            "video": video_output,
        })
        
        # Limit history for memory management
        if len(history) > 3:  # Reduced from 5 to 3
            history = history[-3:]
        
        progress(1.0, desc="✅ Realistic deepfake complete!")
        yield (text, str(audio_output), video_output, history)
        
        # Cleanup after operation
        if memory_manager.should_cleanup():
            memory_manager.cleanup()
        
    except Exception as e:
        yield (f"❌ Error: {e}", None, None, history)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        del tts_model
        gc.collect()

# Auto-load placeholder files
def get_placeholder_files():
    reference_audio = "/Users/home/Downloads/reference.wav"
    face_image = "/Users/home/Pictures/sandy.jpg"
    source_video = "/Users/home/Pictures/vid.mp4"
    
    return (
        reference_audio if os.path.exists(reference_audio) else None,
        face_image if os.path.exists(face_image) else None,
        source_video if os.path.exists(source_video) else None
    )

if __name__ == "__main__":
    # Browser memory optimization
    custom_css = """
    <style>
    * { max-width: 100%; }
    video { max-height: 500px; }
    audio { max-width: 400px; }
    </style>
    """
    
    # Get placeholder files
    default_audio, default_image, default_video = get_placeholder_files()
    
    with gr.Blocks(title="Mac NetNavi - Realistic Deepfakes", css=custom_css, analytics_enabled=False) as demo:
        state = gr.State([])
        
        gr.Markdown("""
        # Mac NetNavi - Realistic Deepfakes
        AI Digital Twin for M3 MacBook Air - Memory Optimized
        
        **Custom XTTS Settings:** temp=0.05, top_k=1, top_p=1.0, rep_penalty=1.05
        **Features:** Advanced lip-sync, source video movement, memory management
        """)
        
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(
                    label="Text", 
                    lines=3,
                    value="Hello, this is my digital twin speaking with realistic voice and movement!",
                    placeholder="Enter text for your digital twin to speak..."
                )
                use_ai = gr.Checkbox(label="🤖 Use AI (Ollama)", value=False)
                voice = gr.Audio(
                    label="Voice Sample", 
                    type="filepath",
                    value=default_audio
                )
                image = gr.Image(
                    label="Portrait", 
                    type="filepath",
                    value=default_image
                )
                source_video = gr.Video(
                    label="Source Video (optional - for realistic movement)",
                    value=default_video
                )
                generate = gr.Button("🎭 Generate Realistic Deepfake", variant="primary")
            
            with gr.Column():
                response = gr.Textbox(label="Response", interactive=False)
                audio_out = gr.Audio(label="Generated Audio", type="filepath")
                video_out = gr.Video(label="Realistic Deepfake Video")
        
        def run(txt, v_path, img_path, src_vid, ai_flag, hist, progress=gr.Progress()):
            if not v_path or not img_path:
                gr.Error("Voice sample and portrait image required")
                return
            
            class F:
                def __init__(self, name): self.name = name
            
            for r, a, v, h in pipeline(txt or "Hello, this is my digital twin!", F(v_path), F(img_path), src_vid, ai_flag, hist, progress):
                yield r, a, v, h
        
        generate.click(
            run,
            [text, voice, image, source_video, use_ai, state],
            [response, audio_out, video_out, state]
        )
    
    demo.launch(
        server_name="127.0.0.1", 
        server_port=7861,
        max_file_size="100mb",
        show_error=True
    )
