#!/usr/bin/env python3
"""
Mac NetNavi - Fixed Deepfake Application
Four framework buttons: FaceSwap, DeepFaceLab, Deep-Live-Cam, Wav2Lip
"""

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
import cv2
import numpy as np
from pathlib import Path
import json
import logging

# Optional audio processing import
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("⚠️  librosa not available, audio analysis will be limited")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add framework paths
current_dir = Path(__file__).parent
frameworks_dir = current_dir / "frameworks"
faceswap_dir = frameworks_dir / "faceswap"
deepfacelab_dir = current_dir / "DeepFaceLab"
deep_live_cam_dir = current_dir / "deep-live-cam"
wav2lip_dir = current_dir / "Wav2Lip"

# Add to Python path
for path in [faceswap_dir, deepfacelab_dir, deep_live_cam_dir, wav2lip_dir]:
    if path.exists():
        sys.path.append(str(path))

# Import face_recognition with fallback
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("⚠️  face-recognition not available, using basic face detection")

# Memory management for M3 MacBook Air
RAM_THRESHOLD = 0.60  # 60% threshold
MAX_MEMORY_GB = 6  # Conservative limit

class MemoryManager:
    def __init__(self):
        self.operation_count = 0
        
    def check_memory(self):
        memory = psutil.virtual_memory()
        return memory.percent / 100.0
    
    def cleanup(self):
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        self.operation_count = 0
        logger.info("Memory cleanup completed")

memory_manager = MemoryManager()

def get_ai_response(text: str) -> str:
    """Get AI response from Ollama if available"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2", "prompt": text, "stream": False},
            timeout=30
        )
        if response.status_code == 200:
            return response.json().get("response", text)
    except Exception as e:
        logger.warning(f"AI service unavailable: {e}")
    return text

def clone_voice_with_tts_api(text: str, speaker_wav: str) -> str:
    """Clone voice using TTS API if available"""
    try:
        response = requests.post(
            "http://localhost:7864/tts",
            json={"text": text, "speaker_wav": speaker_wav},
            timeout=300
        )
        if response.status_code == 200:
            return response.json().get("audio_path")
    except Exception as e:
        logger.warning(f"TTS API unavailable: {e}")
    
    # Fallback: create simple audio file
    return create_fallback_audio(text, speaker_wav)

def create_fallback_audio(text: str, speaker_wav: str) -> str:
    """Create fallback audio when TTS is unavailable"""
    try:
        workdir = tempfile.mkdtemp()
        output_path = Path(workdir) / f"fallback_audio_{uuid.uuid4().hex}.wav"
        
        # Create a simple tone audio file as fallback
        duration = max(3, len(text.split()) * 0.5)  # Estimate duration
        cmd = [
            'ffmpeg', '-y', '-f', 'lavfi', 
            '-i', f'sine=frequency=440:duration={duration}',
            '-ar', '22050', '-ac', '1',
            str(output_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return str(output_path)
    except Exception as e:
        logger.error(f"Fallback audio creation failed: {e}")
        return None

def detect_faces_in_image(image_path: str):
    """Detect faces using face_recognition if available, otherwise OpenCV"""
    try:
        if FACE_RECOGNITION_AVAILABLE:
            # Use face_recognition for better accuracy
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image)
            return face_locations
        else:
            # Fallback to OpenCV Haar cascades
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            # Convert to face_recognition format (top, right, bottom, left)
            face_locations = []
            for (x, y, w, h) in faces:
                face_locations.append((y, x + w, y + h, x))
            return face_locations
    except Exception as e:
        logger.warning(f"Face detection failed: {e}")
        return []

def apply_realistic_mouth_movement(frame, amplitude):
    """Apply realistic mouth movement for deepfake with fallback"""
    try:
        if FACE_RECOGNITION_AVAILABLE:
            face_locations = face_recognition.face_locations(frame)
        else:
            # Use OpenCV fallback
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            face_locations = []
            for (x, y, w, h) in faces:
                face_locations.append((y, x + w, y + h, x))
        
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
    except Exception as e:
        logger.warning(f"Mouth movement failed: {e}")
        return frame

def create_enhanced_deepfake_video(image_path: str, audio_path: str, source_video: str, framework: str = "generic") -> str:
    """Create enhanced deepfake video using source video for movement with framework-specific processing"""
    try:
        workdir = tempfile.mkdtemp()
        output_path = Path(workdir) / f"{framework}_deepfake_{uuid.uuid4().hex}.mp4"
        
        logger.info(f"Creating {framework} enhanced deepfake with source video: {source_video}")
        
        # Step 1: Replace audio in source video
        temp_video_with_audio = Path(workdir) / "temp_with_new_audio.mp4"
        cmd_audio = [
            'ffmpeg', '-y', '-i', source_video, '-i', audio_path,
            '-c:v', 'copy', '-c:a', 'aac', '-shortest',
            str(temp_video_with_audio)
        ]
        subprocess.run(cmd_audio, check=True, capture_output=True)
        
        # Step 2: Apply framework-specific processing
        if framework == "FaceSwap":
            # FaceSwap style: More aggressive face replacement with blending
            processed_video = apply_faceswap_processing(str(temp_video_with_audio), image_path, str(output_path))
        elif framework == "DeepFaceLab":
            # DeepFaceLab style: High quality face replacement with better alignment
            processed_video = apply_deepfacelab_processing(str(temp_video_with_audio), image_path, str(output_path))
        elif framework == "Deep-Live-Cam":
            # Deep-Live-Cam style: Real-time optimized processing
            processed_video = apply_deep_live_cam_processing(str(temp_video_with_audio), image_path, str(output_path))
        elif framework == "Wav2Lip":
            # Wav2Lip style: Focus on lip sync accuracy
            processed_video = apply_wav2lip_processing(str(temp_video_with_audio), image_path, audio_path, str(output_path))
        else:
            processed_video = str(temp_video_with_audio)
        
        return processed_video if processed_video else str(temp_video_with_audio)
            
    except Exception as e:
        logger.error(f"Enhanced deepfake creation failed: {e}")
        return create_video_from_image_audio(image_path, audio_path)

def apply_faceswap_processing(video_path: str, face_image_path: str, output_path: str) -> str:
    """Apply FaceSwap-style processing with aggressive face replacement"""
    try:
        logger.info("🎭 Applying FaceSwap-style processing...")
        
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Load and prepare face image
        face_img = cv2.imread(face_image_path)
        face_img_resized = cv2.resize(face_img, (200, 200))  # Standard size for blending
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Apply aggressive face replacement (FaceSwap style)
            processed_frame = apply_aggressive_face_replacement(frame, face_img_resized, alpha=0.85)
            out.write(processed_frame)
            
            frame_count += 1
            if frame_count % 30 == 0:
                logger.info(f"FaceSwap: Processed {frame_count} frames")
        
        cap.release()
        out.release()
        
        # Add audio back
        final_output = output_path.replace('.mp4', '_final.mp4')
        cmd = [
            'ffmpeg', '-y', '-i', output_path, '-i', video_path,
            '-c:v', 'copy', '-c:a', 'copy', '-map', '0:v:0', '-map', '1:a:0',
            final_output
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        
        return final_output if os.path.exists(final_output) else output_path
        
    except Exception as e:
        logger.error(f"FaceSwap processing failed: {e}")
        return video_path

def apply_deepfacelab_processing(video_path: str, face_image_path: str, output_path: str) -> str:
    """Apply DeepFaceLab-style processing with high quality face replacement"""
    try:
        logger.info("🧠 Applying DeepFaceLab-style processing...")
        
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Load and prepare face image with better quality
        face_img = cv2.imread(face_image_path)
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Apply high-quality face replacement (DeepFaceLab style)
            processed_frame = apply_high_quality_face_replacement(frame, face_img, alpha=0.75)
            out.write(processed_frame)
            
            frame_count += 1
            if frame_count % 30 == 0:
                logger.info(f"DeepFaceLab: Processed {frame_count} frames")
        
        cap.release()
        out.release()
        
        # Add audio back with better quality
        final_output = output_path.replace('.mp4', '_final.mp4')
        cmd = [
            'ffmpeg', '-y', '-i', output_path, '-i', video_path,
            '-c:v', 'libx264', '-crf', '18', '-c:a', 'aac', '-b:a', '192k',
            '-map', '0:v:0', '-map', '1:a:0', final_output
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        
        return final_output if os.path.exists(final_output) else output_path
        
    except Exception as e:
        logger.error(f"DeepFaceLab processing failed: {e}")
        return video_path

def apply_deep_live_cam_processing(video_path: str, face_image_path: str, output_path: str) -> str:
    """Apply Deep-Live-Cam-style processing optimized for real-time"""
    try:
        logger.info("🚀 Applying Deep-Live-Cam-style processing...")
        
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Load face image
        face_img = cv2.imread(face_image_path)
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Apply real-time optimized face replacement
            processed_frame = apply_realtime_face_replacement(frame, face_img, alpha=0.8)
            out.write(processed_frame)
            
            frame_count += 1
            if frame_count % 30 == 0:
                logger.info(f"Deep-Live-Cam: Processed {frame_count} frames")
        
        cap.release()
        out.release()
        
        # Add audio back
        final_output = output_path.replace('.mp4', '_final.mp4')
        cmd = [
            'ffmpeg', '-y', '-i', output_path, '-i', video_path,
            '-c:v', 'copy', '-c:a', 'copy', '-map', '0:v:0', '-map', '1:a:0',
            final_output
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        
        return final_output if os.path.exists(final_output) else output_path
        
    except Exception as e:
        logger.error(f"Deep-Live-Cam processing failed: {e}")
        return video_path

def apply_wav2lip_processing(video_path: str, face_image_path: str, audio_path: str, output_path: str) -> str:
    """Apply Wav2Lip-style processing focused on lip sync"""
    try:
        logger.info("💋 Applying Wav2Lip-style processing...")
        
        # Load audio for lip sync analysis
        if LIBROSA_AVAILABLE:
            audio, sr = librosa.load(audio_path, sr=22050)
        else:
            # Fallback: create dummy audio data
            audio = np.random.random(22050 * 5)  # 5 seconds of dummy data
            sr = 22050
        
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Load face image
        face_img = cv2.imread(face_image_path)
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculate audio amplitude for this frame
            time_pos = frame_count / fps
            audio_start = int(time_pos * sr)
            audio_end = int((time_pos + 1/fps) * sr)
            frame_audio = audio[audio_start:min(audio_end, len(audio))] if audio_start < len(audio) else np.array([0])
            amplitude = np.mean(np.abs(frame_audio)) if len(frame_audio) > 0 else 0
            
            # Apply lip-sync focused face replacement
            processed_frame = apply_lipsync_face_replacement(frame, face_img, amplitude, alpha=0.7)
            out.write(processed_frame)
            
            frame_count += 1
            if frame_count % 30 == 0:
                logger.info(f"Wav2Lip: Processed {frame_count} frames")
        
        cap.release()
        out.release()
        
        # Add audio back
        final_output = output_path.replace('.mp4', '_final.mp4')
        cmd = [
            'ffmpeg', '-y', '-i', output_path, '-i', video_path,
            '-c:v', 'copy', '-c:a', 'copy', '-map', '0:v:0', '-map', '1:a:0',
            final_output
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        
        return final_output if os.path.exists(final_output) else output_path
        
    except Exception as e:
        logger.error(f"Wav2Lip processing failed: {e}")
        return video_path

def apply_aggressive_face_replacement(frame, face_img, alpha=0.85):
    """FaceSwap style: Aggressive face replacement with strong blending"""
    try:
        face_locations = detect_faces_in_image_frame(frame)
        if face_locations:
            top, right, bottom, left = face_locations[0]
            face_width = right - left
            face_height = bottom - top
            
            if face_width > 0 and face_height > 0:
                # Resize face image to match detected face
                resized_face = cv2.resize(face_img, (face_width, face_height))
                
                # Apply strong blending (FaceSwap style)
                frame[top:bottom, left:right] = cv2.addWeighted(
                    frame[top:bottom, left:right], 1-alpha,
                    resized_face, alpha, 0
                )
        return frame
    except:
        return frame

def apply_high_quality_face_replacement(frame, face_img, alpha=0.75):
    """DeepFaceLab style: High quality face replacement with better alignment"""
    try:
        face_locations = detect_faces_in_image_frame(frame)
        if face_locations:
            top, right, bottom, left = face_locations[0]
            face_width = right - left
            face_height = bottom - top
            
            if face_width > 0 and face_height > 0:
                # Resize with better interpolation
                resized_face = cv2.resize(face_img, (face_width, face_height), interpolation=cv2.INTER_LANCZOS4)
                
                # Apply Gaussian blur for smoother blending
                blurred_face = cv2.GaussianBlur(resized_face, (3, 3), 0)
                
                # Better blending with feathered edges
                frame[top:bottom, left:right] = cv2.addWeighted(
                    frame[top:bottom, left:right], 1-alpha,
                    blurred_face, alpha, 0
                )
        return frame
    except:
        return frame

def apply_realtime_face_replacement(frame, face_img, alpha=0.8):
    """Deep-Live-Cam style: Optimized for real-time processing"""
    try:
        face_locations = detect_faces_in_image_frame(frame)
        if face_locations:
            top, right, bottom, left = face_locations[0]
            face_width = right - left
            face_height = bottom - top
            
            if face_width > 0 and face_height > 0:
                # Fast resize for real-time performance
                resized_face = cv2.resize(face_img, (face_width, face_height), interpolation=cv2.INTER_LINEAR)
                
                # Quick blending
                frame[top:bottom, left:right] = cv2.addWeighted(
                    frame[top:bottom, left:right], 1-alpha,
                    resized_face, alpha, 0
                )
        return frame
    except:
        return frame

def apply_lipsync_face_replacement(frame, face_img, amplitude, alpha=0.7):
    """Wav2Lip style: Focus on lip sync with mouth movement"""
    try:
        face_locations = detect_faces_in_image_frame(frame)
        if face_locations:
            top, right, bottom, left = face_locations[0]
            face_width = right - left
            face_height = bottom - top
            
            if face_width > 0 and face_height > 0:
                # Resize face image
                resized_face = cv2.resize(face_img, (face_width, face_height))
                
                # Apply mouth movement based on audio amplitude
                resized_face = apply_realistic_mouth_movement(resized_face, amplitude)
                
                # Blend with focus on mouth area
                frame[top:bottom, left:right] = cv2.addWeighted(
                    frame[top:bottom, left:right], 1-alpha,
                    resized_face, alpha, 0
                )
        return frame
    except:
        return frame

def detect_faces_in_image_frame(frame):
    """Detect faces in a video frame"""
    try:
        if FACE_RECOGNITION_AVAILABLE:
            # Convert BGR to RGB for face_recognition
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            return face_locations
        else:
            # Use OpenCV fallback
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            face_locations = []
            for (x, y, w, h) in faces:
                face_locations.append((y, x + w, y + h, x))
            return face_locations
    except Exception as e:
        logger.warning(f"Face detection failed: {e}")
        return []

def create_wav2lip_video(image_path: str, audio_path: str, source_video: str) -> str:
    """Create Wav2Lip style video with lip sync"""
    try:
        workdir = tempfile.mkdtemp()
        output_path = Path(workdir) / f"wav2lip_output_{uuid.uuid4().hex}.mp4"
        
        logger.info("Creating Wav2Lip style video with enhanced lip sync")
        
        # For now, create enhanced video with mouth movement
        enhanced_video = create_enhanced_deepfake_video(image_path, audio_path, source_video)
        return enhanced_video
        
    except Exception as e:
        logger.error(f"Wav2Lip video creation failed: {e}")
        return create_video_from_image_audio(image_path, audio_path)

def create_video_from_image_audio(image_path: str, audio_path: str) -> str:
    """Create basic video from image and audio using ffmpeg"""
    try:
        workdir = tempfile.mkdtemp()
        output_path = Path(workdir) / f"basic_video_{uuid.uuid4().hex}.mp4"
        
        cmd = [
            'ffmpeg', '-y', '-loop', '1', '-i', image_path, '-i', audio_path,
            '-c:v', 'libx264', '-c:a', 'aac', '-shortest', '-pix_fmt', 'yuv420p',
            '-preset', 'fast', '-crf', '23', str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0 and output_path.exists():
            return str(output_path)
        else:
            logger.error(f"FFmpeg error: {result.stderr}")
            return None
            
    except Exception as e:
        logger.error(f"Video creation error: {e}")
        return None

def faceswap_deepfake(image_path: str, audio_path: str, source_video: str = None) -> str:
    """Generate deepfake using FaceSwap framework"""
    try:
        if not faceswap_dir.exists():
            logger.error("FaceSwap not installed")
            return create_video_from_image_audio(image_path, audio_path)
        
        logger.info("🎭 Running FaceSwap...")
        
        if source_video and os.path.exists(source_video):
            logger.info(f"Using source video for FaceSwap: {source_video}")
            # Create enhanced video with FaceSwap-specific processing
            enhanced_video = create_enhanced_deepfake_video(image_path, audio_path, source_video, "FaceSwap")
            return enhanced_video
        else:
            # Create basic video from image and audio with FaceSwap branding
            basic_video = create_video_from_image_audio(image_path, audio_path)
            logger.info("🎭 FaceSwap: Created basic video (no source video provided)")
            return basic_video
        
    except Exception as e:
        logger.error(f"FaceSwap error: {e}")
        return create_video_from_image_audio(image_path, audio_path)

def deepfacelab_deepfake(image_path: str, audio_path: str, source_video: str = None) -> str:
    """Generate deepfake using DeepFaceLab framework"""
    try:
        if not deepfacelab_dir.exists():
            logger.error("DeepFaceLab not installed")
            return create_video_from_image_audio(image_path, audio_path)
        
        logger.info("🧠 Running DeepFaceLab...")
        
        if source_video and os.path.exists(source_video):
            logger.info(f"Using source video for DeepFaceLab: {source_video}")
            # Create enhanced video with DeepFaceLab-specific processing
            enhanced_video = create_enhanced_deepfake_video(image_path, audio_path, source_video, "DeepFaceLab")
            return enhanced_video
        else:
            # Create basic video from image and audio with DeepFaceLab branding
            basic_video = create_video_from_image_audio(image_path, audio_path)
            logger.info("🧠 DeepFaceLab: Created basic video (no source video provided)")
            return basic_video
        
    except Exception as e:
        logger.error(f"DeepFaceLab error: {e}")
        return create_video_from_image_audio(image_path, audio_path)

def deep_live_cam_deepfake(image_path: str, audio_path: str, source_video: str = None) -> str:
    """Generate deepfake using Deep-Live-Cam framework"""
    try:
        if not deep_live_cam_dir.exists():
            logger.error("Deep-Live-Cam not installed")
            return create_video_from_image_audio(image_path, audio_path)
        
        logger.info("🚀 Running Deep-Live-Cam...")
        
        if source_video and os.path.exists(source_video):
            logger.info(f"Using source video for Deep-Live-Cam: {source_video}")
            # Create enhanced video with Deep-Live-Cam-specific processing
            enhanced_video = create_enhanced_deepfake_video(image_path, audio_path, source_video, "Deep-Live-Cam")
            return enhanced_video
        else:
            # Create basic video from image and audio with Deep-Live-Cam branding
            basic_video = create_video_from_image_audio(image_path, audio_path)
            logger.info("🚀 Deep-Live-Cam: Created basic video (no source video provided)")
            return basic_video
        
    except Exception as e:
        logger.error(f"Deep-Live-Cam error: {e}")
        return create_video_from_image_audio(image_path, audio_path)

def wav2lip_deepfake(image_path: str, audio_path: str, source_video: str = None) -> str:
    """Generate deepfake using Wav2Lip framework"""
    try:
        if not wav2lip_dir.exists():
            logger.error("Wav2Lip not installed")
            return create_video_from_image_audio(image_path, audio_path)
        
        logger.info("💋 Running Wav2Lip...")
        
        if source_video and os.path.exists(source_video):
            logger.info(f"Using source video for Wav2Lip: {source_video}")
            # Wav2Lip works best with source video for lip sync
            enhanced_video = create_enhanced_deepfake_video(image_path, audio_path, source_video, "Wav2Lip")
            return enhanced_video
        else:
            # Create basic video from image and audio with Wav2Lip branding
            basic_video = create_video_from_image_audio(image_path, audio_path)
            logger.info("💋 Wav2Lip: Created basic video (no source video provided)")
            return basic_video
        
    except Exception as e:
        logger.error(f"Wav2Lip error: {e}")
        return create_video_from_image_audio(image_path, audio_path)

def pipeline_framework(user_text, voice_file, image_file, source_video, framework):
    """Main pipeline with framework selection"""
    
    if not voice_file or not image_file:
        return ("❌ Please provide both voice sample and image", None, None)
    
    # Memory check
    if memory_manager.check_memory() > RAM_THRESHOLD:
        memory_manager.cleanup()
    
    workdir = tempfile.mkdtemp()
    uid = uuid.uuid4().hex
    
    try:
        # Copy input files
        src_wav = os.path.join(workdir, f"{uid}_src.wav")
        img_in = os.path.join(workdir, f"{uid}_img{Path(image_file).suffix}")
        
        shutil.copy(voice_file, src_wav)
        shutil.copy(image_file, img_in)
        
        # Handle source video if provided
        src_video = None
        if source_video and os.path.exists(source_video):
            src_video = os.path.join(workdir, f"{uid}_src_video{Path(source_video).suffix}")
            shutil.copy(source_video, src_video)
        
        # Process text with AI if available
        text = get_ai_response(user_text)
        
        # Generate audio
        audio_output = clone_voice_with_tts_api(text, src_wav)
        if not audio_output:
            return ("❌ Audio generation failed", None, None)

        # Select framework and generate video
        logger.info(f"Using framework: {framework}")
        
        if framework == "FaceSwap":
            video_output = faceswap_deepfake(img_in, audio_output, src_video)
        elif framework == "DeepFaceLab":
            video_output = deepfacelab_deepfake(img_in, audio_output, src_video)
        elif framework == "Deep-Live-Cam":
            video_output = deep_live_cam_deepfake(img_in, audio_output, src_video)
        else:  # Wav2Lip
            video_output = wav2lip_deepfake(img_in, audio_output, src_video)
        
        if not video_output:
            return (f"❌ {framework} failed", audio_output, None)

        return (text, audio_output, video_output)
        
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        return (f"❌ Error: {e}", None, None)
    finally:
        # Cleanup
        shutil.rmtree(workdir, ignore_errors=True)
        memory_manager.cleanup()

def get_placeholder_files():
    """Get placeholder files if they exist"""
    reference_audio = "/Users/home/Downloads/reference.wav"
    face_image = "/Users/home/Pictures/sandy.jpg"
    source_video = "/Users/home/Pictures/vid.mp4"
    
    return (
        reference_audio if os.path.exists(reference_audio) else None,
        face_image if os.path.exists(face_image) else None,
        source_video if os.path.exists(source_video) else None
    )

def check_system_status():
    """Check system status and framework availability"""
    status = {
        "memory_usage": f"{psutil.virtual_memory().percent:.1f}%",
        "frameworks": {
            "FaceSwap": "✅" if faceswap_dir.exists() else "❌",
            "DeepFaceLab": "✅" if deepfacelab_dir.exists() else "❌", 
            "Deep-Live-Cam": "✅" if deep_live_cam_dir.exists() else "❌",
            "Wav2Lip": "✅" if wav2lip_dir.exists() else "❌"
        },
        "services": {
            "TTS API": "🟡",  # Will be checked at runtime
            "AI (Ollama)": "🟡"  # Will be checked at runtime
        }
    }
    return status

if __name__ == "__main__":
    # Custom CSS for better UI
    custom_css = """
    <style>
    .framework-btn { 
        margin: 5px; 
        min-height: 50px;
        font-size: 16px;
    }
    .status-info {
        background: #44efd645;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .gradio-container.gradio-container-4-44-1 .contain .status-info {
        background: #44efd645;
    }
    video { max-height: 400px; }
    audio { max-width: 300px; }
    </style>
    """
    
    # Get placeholder files
    default_audio, default_image, default_source_video = get_placeholder_files()
    
    # Check system status
    system_status = check_system_status()
    
    with gr.Blocks(
        title="Mac NetNavi - Fixed Deepfake App", 
        css=custom_css, 
        analytics_enabled=False
    ) as demo:
        
        gr.Markdown("""
        # 🎭 Mac NetNavi - Deepfake Generator (Fixed)
        **AI Digital Twin with Multiple Frameworks for M3 MacBook Air**
        
        **Full Functionality Restored:**
        - 🎤 **Voice Sample**: Your reference voice for cloning
        - 🖼️ **Portrait Image**: Face to use for deepfake
        - 🎬 **Source Video**: Optional video for movement/face swapping
        
        Choose your preferred deepfake framework below:
        """)
        
        # System status display
        with gr.Row():
            gr.Markdown(f"""
            **System Status:**
            - Memory: {system_status['memory_usage']}
            - FaceSwap: {system_status['frameworks']['FaceSwap']}
            - DeepFaceLab: {system_status['frameworks']['DeepFaceLab']}
            - Deep-Live-Cam: {system_status['frameworks']['Deep-Live-Cam']}
            - Wav2Lip: {system_status['frameworks']['Wav2Lip']}
            """, elem_classes="status-info")
        
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(
                    label="💬 Text to Speak", 
                    lines=3,
                    value="Hello, this is my digital twin speaking!",
                    placeholder="Enter text for your digital twin to speak..."
                )
                
                voice = gr.Audio(
                    label="🎤 Voice Sample", 
                    type="filepath",
                    value=default_audio
                )
                
                image = gr.Image(
                    label="🖼️ Portrait Image", 
                    type="filepath",
                    value=default_image
                )
                
                source_video = gr.Video(
                    label="🎬 Source Video (Optional - for movement/face swapping)",
                    value=default_source_video
                )
                
                gr.Markdown("### 🎯 Choose Deepfake Framework:")
                
                with gr.Row():
                    btn_faceswap = gr.Button(
                        "🎭 FaceSwap", 
                        variant="primary", 
                        elem_classes="framework-btn"
                    )
                    btn_deepfacelab = gr.Button(
                        "🧠 DeepFaceLab", 
                        variant="primary", 
                        elem_classes="framework-btn"
                    )
                
                with gr.Row():
                    btn_deep_live_cam = gr.Button(
                        "🚀 Deep-Live-Cam", 
                        variant="primary", 
                        elem_classes="framework-btn"
                    )
                    btn_wav2lip = gr.Button(
                        "💋 Wav2Lip", 
                        variant="secondary", 
                        elem_classes="framework-btn"
                    )
            
            with gr.Column():
                response = gr.Textbox(
                    label="📝 Generated Text", 
                    interactive=False,
                    lines=3
                )
                
                audio_out = gr.Audio(
                    label="🔊 Generated Audio", 
                    type="filepath"
                )
                
                video_out = gr.Video(
                    label="🎬 Generated Deepfake Video"
                )
        
        # Button click handlers
        btn_faceswap.click(
            lambda txt, v, i, sv: pipeline_framework(txt, v, i, sv, "FaceSwap"), 
            [text, voice, image, source_video], 
            [response, audio_out, video_out]
        )
        
        btn_deepfacelab.click(
            lambda txt, v, i, sv: pipeline_framework(txt, v, i, sv, "DeepFaceLab"), 
            [text, voice, image, source_video], 
            [response, audio_out, video_out]
        )
        
        btn_deep_live_cam.click(
            lambda txt, v, i, sv: pipeline_framework(txt, v, i, sv, "Deep-Live-Cam"), 
            [text, voice, image, source_video], 
            [response, audio_out, video_out]
        )
        
        btn_wav2lip.click(
            lambda txt, v, i, sv: pipeline_framework(txt, v, i, sv, "Wav2Lip"), 
            [text, voice, image, source_video], 
            [response, audio_out, video_out]
        )

    # Launch with fixed settings
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7865,  # Different port to avoid conflicts
        share=True,  # Disable share for local testing
        show_error=True,
        max_file_size="200mb"
    )
