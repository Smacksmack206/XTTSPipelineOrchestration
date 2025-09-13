#!/usr/bin/env python3

"""
Mac NetNavi - Enhanced Deepfake Studio
Advanced AI-powered deepfake generation with quality enhancement
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
from datetime import datetime

# Set up comprehensive logging for real-time status tracking
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deepfake_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProcessingLogger:
    def __init__(self):
        self.start_time = None
        self.current_operation = None
        self.total_frames = 0
        self.processed_frames = 0
        
    def start_processing(self, operation, total_frames=0):
        self.start_time = time.time()
        self.current_operation = operation
        self.total_frames = total_frames
        self.processed_frames = 0
        logger.info(f"🚀 STARTED: {operation} - Total frames: {total_frames}")
        
    def update_progress(self, frames_processed=1):
        self.processed_frames += frames_processed
        if self.total_frames > 0:
            progress = (self.processed_frames / self.total_frames) * 100
            elapsed = time.time() - self.start_time
            logger.info(f"📊 PROGRESS: {self.processed_frames}/{self.total_frames} frames ({progress:.1f}%) - Elapsed: {elapsed:.1f}s")
        else:
            logger.info(f"📊 PROGRESS: {self.processed_frames} frames processed")
            
    def log_status(self, message):
        logger.info(f"ℹ️  STATUS: {message}")
        
    def log_error(self, error):
        logger.error(f"❌ ERROR: {error}")
        
    def finish_processing(self):
        if self.start_time:
            elapsed = time.time() - self.start_time
            logger.info(f"✅ COMPLETED: {self.current_operation} - Total time: {elapsed:.1f}s - Frames: {self.processed_frames}")

# Global logger instance
processing_logger = ProcessingLogger()

# Optional audio processing import
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("librosa not available - some audio features may be limited")

# Optional TTS imports
try:
    from TTS.api import TTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("TTS not available - using system TTS")

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    print("pyttsx3 not available - using system say command")

# Import face_recognition with fallback
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("face_recognition not available - using OpenCV face detection")

# Enhanced model imports with fallbacks
try:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    REALESRGAN_AVAILABLE = True
except ImportError:
    REALESRGAN_AVAILABLE = False
    print("Real-ESRGAN not available - using OpenCV enhancement")

try:
    from gfpgan import GFPGANer
    GFPGAN_AVAILABLE = True
except ImportError:
    GFPGAN_AVAILABLE = False
    print("GFPGAN not available - using basic face enhancement")

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("Whisper not available - using basic audio processing")

try:
    from skimage.metrics import structural_similarity as ssim
    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False
    print("scikit-image not available - quality metrics limited")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from professional_deepfake_engine import ProfessionalDeepfakeEngine
from m3_optimized_engine import M3OptimizedProcessor
from real_time_tracker import frame_tracker
from chunked_processor import ChunkedVideoProcessor, StreamingProcessor

class EnhancedDeepfakeProcessor:
    def __init__(self):
        self.device = self.get_best_device()
        self.target_resolution = (1920, 1080)
        self.face_enhance_factor = 2.0
        self.temporal_consistency = True
        
        # Initialize processors
        self.m3_processor = M3OptimizedProcessor()
        self.professional_engine = ProfessionalDeepfakeEngine()
        self.chunked_processor = ChunkedVideoProcessor(chunk_size=30, max_workers=4)
        self.streaming_processor = StreamingProcessor(buffer_size=10)
        
        # Initialize all available models
        self.setup_face_enhancer()
        self.setup_voice_cloner()
        self.setup_audio_processor()
        
    def get_best_device(self):
        """Get the best available device"""
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    def setup_face_enhancer(self):
        """Initialize face enhancement models"""
        self.face_enhancer = None
        self.upsampler = None
        
        if REALESRGAN_AVAILABLE:
            try:
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
                self.upsampler = RealESRGANer(
                    scale=4,
                    model_path='weights/RealESRGAN_x4plus.pth',
                    model=model,
                    tile=0,
                    tile_pad=10,
                    pre_pad=0,
                    half=False
                )
                logger.info("Real-ESRGAN upsampler initialized")
            except Exception as e:
                logger.warning(f"Real-ESRGAN setup failed: {e}")
        
        if GFPGAN_AVAILABLE:
            try:
                self.face_enhancer = GFPGANer(
                    model_path='weights/GFPGANv1.3.pth',
                    upscale=2,
                    arch='clean',
                    channel_multiplier=2,
                    bg_upsampler=self.upsampler
                )
                logger.info("GFPGAN face enhancer initialized")
            except Exception as e:
                logger.warning(f"GFPGAN setup failed: {e}")
    
    def setup_voice_cloner(self):
        """Initialize voice cloning models"""
        self.tts = None
        self.pyttsx3_engine = None
        
        if TTS_AVAILABLE:
            try:
                self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
                logger.info("XTTS v2 voice cloner initialized")
            except Exception as e:
                logger.warning(f"TTS setup failed: {e}")
        
        if PYTTSX3_AVAILABLE:
            try:
                self.pyttsx3_engine = pyttsx3.init()
                logger.info("pyttsx3 engine initialized")
            except Exception as e:
                logger.warning(f"pyttsx3 setup failed: {e}")
    
    def setup_face_detector(self):
        """Initialize face detection"""
        self.face_detector = None
        self.opencv_face_cascade = None
        
        if FACE_RECOGNITION_AVAILABLE:
            self.face_detector = face_recognition
            logger.info("face_recognition detector initialized")
        else:
            # Fallback to OpenCV
            try:
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.opencv_face_cascade = cv2.CascadeClassifier(cascade_path)
                logger.info("OpenCV face cascade initialized")
            except Exception as e:
                logger.warning(f"OpenCV face cascade setup failed: {e}")
    
    def setup_audio_processor(self):
        """Initialize audio processing"""
        self.whisper_model = None
        
        if WHISPER_AVAILABLE:
            try:
                self.whisper_model = whisper.load_model("base")
                logger.info("Whisper model initialized")
            except Exception as e:
                logger.warning(f"Whisper setup failed: {e}")
    def enhance_face_quality(self, image: np.ndarray) -> np.ndarray:
        """Apply advanced face enhancement"""
        if self.face_enhancer is not None:
            try:
                # Use GFPGAN for professional face enhancement
                _, _, enhanced_img = self.face_enhancer.enhance(
                    image, 
                    has_aligned=False, 
                    only_center_face=False, 
                    paste_back=True
                )
                return enhanced_img
            except Exception as e:
                logger.error(f"GFPGAN enhancement failed: {e}")
        
        # Fallback to OpenCV enhancement
        try:
            # Convert to LAB color space for better enhancement
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            
            # Merge channels and convert back
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
            
            # Apply sharpening
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            
            # Blend original and sharpened
            result = cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)
            
            return result
        except Exception as e:
            logger.error(f"OpenCV enhancement failed: {e}")
            return image

    def upscale_image(self, image: np.ndarray, scale: int = 4) -> np.ndarray:
        """Upscale image using Real-ESRGAN or fallback"""
        if self.upsampler is not None:
            try:
                output, _ = self.upsampler.enhance(image, outscale=scale)
                return output
            except Exception as e:
                logger.error(f"Real-ESRGAN upscaling failed: {e}")
        
        # Fallback to OpenCV upscaling
        height, width = image.shape[:2]
        return cv2.resize(image, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC)

    def detect_faces(self, image: np.ndarray):
        """Detect faces using multiple methods with fallbacks"""
        faces = []
        encodings = []
        
        # Method 1: face_recognition (best quality)
        if FACE_RECOGNITION_AVAILABLE:
            try:
                face_locations = face_recognition.face_locations(image, model="hog")
                if len(face_locations) == 0:
                    # Try CNN model for better detection
                    face_locations = face_recognition.face_locations(image, model="cnn")
                
                if len(face_locations) > 0:
                    face_encodings = face_recognition.face_encodings(image, face_locations)
                    return face_locations, face_encodings
            except Exception as e:
                logger.warning(f"face_recognition failed: {e}")
        
        # Method 2: OpenCV DNN face detector (high quality)
        try:
            net = cv2.dnn.readNetFromTensorflow('opencv_face_detector_uint8.pb', 'opencv_face_detector.pbtxt')
            blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123])
            net.setInput(blob)
            detections = net.forward()
            
            h, w = image.shape[:2]
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    x1 = int(detections[0, 0, i, 3] * w)
                    y1 = int(detections[0, 0, i, 4] * h)
                    x2 = int(detections[0, 0, i, 5] * w)
                    y2 = int(detections[0, 0, i, 6] * h)
                    faces.append((y1, x2, y2, x1))  # Convert to face_recognition format
            
            if len(faces) > 0:
                return faces, None
        except Exception as e:
            logger.warning(f"OpenCV DNN detection failed: {e}")
        
        # Method 3: MTCNN (if available)
        try:
            from mtcnn import MTCNN
            detector = MTCNN()
            result = detector.detect_faces(image)
            
            for face in result:
                if face['confidence'] > 0.9:
                    x, y, w, h = face['box']
                    faces.append((y, x+w, y+h, x))  # Convert to face_recognition format
            
            if len(faces) > 0:
                return faces, None
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"MTCNN detection failed: {e}")
        
        # Method 4: OpenCV Haar Cascade (fallback)
        if self.opencv_face_cascade is not None:
            try:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                detected_faces = self.opencv_face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                
                # Convert to face_recognition format
                for (x, y, w, h) in detected_faces:
                    faces.append((y, x+w, y+h, x))
                
                return faces, None
            except Exception as e:
                logger.warning(f"Haar cascade detection failed: {e}")
        
        # Method 5: MediaPipe (if available)
        try:
            import mediapipe as mp
            mp_face_detection = mp.solutions.face_detection
            
            with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
                results = face_detection.process(image)
                
                if results.detections:
                    h, w = image.shape[:2]
                    for detection in results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        x = int(bbox.xmin * w)
                        y = int(bbox.ymin * h)
                        width = int(bbox.width * w)
                        height = int(bbox.height * h)
                        faces.append((y, x+width, y+height, x))
                    
                    return faces, None
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"MediaPipe detection failed: {e}")
        
        return [], []

    def preprocess_for_detection(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better face detection"""
        try:
            # Ensure proper size (not too small, not too large)
            h, w = image.shape[:2]
            if w < 300 or h < 300:
                scale = max(300/w, 300/h)
                new_w, new_h = int(w * scale), int(h * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            elif w > 1920 or h > 1920:
                scale = min(1920/w, 1920/h)
                new_w, new_h = int(w * scale), int(h * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Enhance contrast and brightness
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
            
            # Reduce noise
            denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
            
            return denoised
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            return image

    def calculate_quality_metrics(self, original: np.ndarray, generated: np.ndarray) -> dict:
        """Calculate comprehensive quality metrics"""
        try:
            metrics = {}
            
            # Ensure same dimensions
            if original.shape != generated.shape:
                generated = cv2.resize(generated, (original.shape[1], original.shape[0]))
            
            # SSIM calculation
            if SSIM_AVAILABLE:
                ssim_score = ssim(original, generated, multichannel=True, data_range=255)
                metrics['ssim'] = float(ssim_score)
            else:
                metrics['ssim'] = 0.0
            
            # MSE and PSNR calculation
            mse = np.mean((original.astype(float) - generated.astype(float)) ** 2)
            metrics['mse'] = float(mse)
            
            if mse > 0:
                psnr = 20 * np.log10(255.0 / np.sqrt(mse))
                metrics['psnr'] = float(psnr)
            else:
                metrics['psnr'] = float('inf')
            
            # Additional metrics
            metrics['mean_diff'] = float(np.mean(np.abs(original.astype(float) - generated.astype(float))))
            metrics['std_diff'] = float(np.std(original.astype(float) - generated.astype(float)))
            
            return metrics
        except Exception as e:
            logger.error(f"Quality metrics calculation failed: {e}")
            return {'ssim': 0.0, 'psnr': 0.0, 'mse': float('inf'), 'mean_diff': 0.0, 'std_diff': 0.0}

    def enhanced_face_swap(self, source_image: str, target_video: str, output_path: str, enhance_quality: bool = True, quality_preset: str = 'auto') -> tuple:
        """M3-optimized face swapping with all performance improvements"""
        try:
            processing_logger.start_processing("M3-Optimized Face Swap")
            processing_logger.log_status(f"Using M3 Metal acceleration and Neural Engine")
            
            # Use M3 optimized processor
            result_path, metrics = self.m3_processor.process_video(
                source_image, target_video, output_path, quality_preset
            )
            
            if result_path:
                processing_logger.log_status(f"M3 processing completed successfully")
                processing_logger.log_status(f"Preview available: {metrics.get('preview_path', 'N/A')}")
                processing_logger.log_status(f"Formats exported: {list(metrics.get('formats', {}).keys())}")
            
            processing_logger.finish_processing()
            return result_path, metrics
            
        except Exception as e:
            processing_logger.log_error(f"M3 face swap failed: {e}")
            # Fallback to professional engine
            processing_logger.log_status("Falling back to professional engine")
            return self.professional_engine_fallback(source_image, target_video, output_path, enhance_quality)
    
    def professional_engine_fallback(self, source_image: str, target_video: str, output_path: str, enhance_quality: bool = True) -> tuple:
        """Fallback to professional engine if M3 optimization fails"""
        try:
            processing_logger.start_processing("Professional Engine Fallback")
            
            # Load and preprocess source image
            source_img = cv2.imread(source_image)
            if source_img is None:
                processing_logger.log_error("Could not load source image")
                return "", {"error": "Could not load source image"}
            
            source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
            processing_logger.log_status("Source image loaded successfully")
            
            # Process video
            processing_logger.log_status(f"Opening target video: {target_video}")
            cap = cv2.VideoCapture(target_video)
            if not cap.isOpened():
                processing_logger.log_error("Could not open target video")
                return "", {"error": "Could not open target video"}
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            processing_logger.log_status(f"Video info - FPS: {fps}, Total frames: {total_frames}")
            
            # Enhanced output settings
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, self.target_resolution)
            
            frame_count = 0
            quality_scores = []
            
            processing_logger.start_processing(f"Processing {total_frames} frames at {fps} FPS", total_frames)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Apply face swapping
                processed_frame = self.process_frame_with_face_swap(source_img, frame_rgb)
                
                # Apply face enhancement only if enabled
                if enhance_quality:
                    enhanced_frame = self.enhance_face_quality(processed_frame)
                else:
                    enhanced_frame = processed_frame
                
                # Calculate quality metrics
                metrics = self.calculate_quality_metrics(frame_rgb, enhanced_frame)
                quality_scores.append(metrics)
                
                # Resize to target resolution only once at the end
                if enhanced_frame.shape[:2] != (self.target_resolution[1], self.target_resolution[0]):
                    enhanced_frame = cv2.resize(enhanced_frame, self.target_resolution)
                
                # Write frame
                out_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_RGB2BGR)
                out.write(out_frame)
                
                frame_count += 1
                
                # Log progress every 30 frames
                if frame_count % 30 == 0:
                    processing_logger.update_progress(30)
                    
                # Log detailed progress every 100 frames
                if frame_count % 100 == 0:
                    avg_ssim = np.mean([s['ssim'] for s in quality_scores[-100:]])
                    processing_logger.log_status(f"Quality check - Average SSIM: {avg_ssim:.3f}")
            
            cap.release()
            out.release()
            
            # Calculate comprehensive metrics
            if quality_scores:
                avg_metrics = {
                    'avg_ssim': np.mean([s['ssim'] for s in quality_scores]),
                    'avg_psnr': np.mean([s['psnr'] for s in quality_scores if s['psnr'] != float('inf')]),
                    'avg_mse': np.mean([s['mse'] for s in quality_scores]),
                    'total_frames': frame_count,
                    'fps': fps,
                    'resolution': f"{self.target_resolution[0]}x{self.target_resolution[1]}",
                    'enhancement_used': 'GFPGAN' if self.face_enhancer else 'OpenCV',
                    'upscaling_used': 'Real-ESRGAN' if self.upsampler else 'OpenCV',
                    'engine_used': 'Professional (Fallback)'
                }
                processing_logger.log_status(f"Final quality metrics - SSIM: {avg_metrics['avg_ssim']:.3f}, PSNR: {avg_metrics['avg_psnr']:.1f}")
            else:
                avg_metrics = {'error': 'No frames processed'}
            
            processing_logger.finish_processing()
            return output_path, avg_metrics
            
        except Exception as e:
            processing_logger.log_error(f"Professional engine fallback failed: {e}")
            return "", {"error": str(e)}

    def process_frame_with_face_swap(self, source_face: np.ndarray, target_frame: np.ndarray) -> np.ndarray:
        """Process frame with professional face swapping"""
        try:
            # Use professional engine for photorealistic results
            result = self.professional_engine.professional_face_swap(source_face, target_frame)
            return result
            
        except Exception as e:
            logger.error(f"Professional face swap failed: {e}")
            return target_frame
    def enhanced_voice_clone(self, text: str, reference_audio: str, output_path: str) -> str:
        """Enhanced voice cloning with multiple fallbacks"""
        if not text.strip():
            return ""
        
        # Try XTTS first (best quality)
        if self.tts is not None and reference_audio:
            try:
                # Preprocess reference audio with Whisper if available
                if self.whisper_model and reference_audio:
                    result = self.whisper_model.transcribe(reference_audio)
                    logger.info(f"Reference audio transcription: {result['text'][:100]}...")
                
                # Generate high-quality voice clone
                self.tts.tts_to_file(
                    text=text,
                    speaker_wav=reference_audio,
                    file_path=output_path,
                    language="en"
                )
                logger.info("XTTS voice cloning completed")
                return output_path
            except Exception as e:
                logger.error(f"XTTS voice cloning failed: {e}")
        
        # Try pyttsx3 (good quality, no reference needed)
        if self.pyttsx3_engine is not None:
            try:
                self.pyttsx3_engine.save_to_file(text, output_path)
                self.pyttsx3_engine.runAndWait()
                logger.info("pyttsx3 voice synthesis completed")
                return output_path
            except Exception as e:
                logger.error(f"pyttsx3 voice synthesis failed: {e}")
        
        # Fallback to system TTS (macOS say command)
        try:
            temp_aiff = output_path.replace('.wav', '.aiff')
            subprocess.run(['say', '-o', temp_aiff, text], check=True)
            
            # Convert to wav if needed
            if output_path.endswith('.wav'):
                subprocess.run([
                    'ffmpeg', '-i', temp_aiff, '-y', output_path
                ], check=True, capture_output=True)
                os.remove(temp_aiff)
            
            logger.info("System TTS completed")
            return output_path
        except Exception as e:
            logger.error(f"System TTS failed: {e}")
            return ""

    def process_audio_with_whisper(self, audio_path: str) -> dict:
        """Process audio with Whisper for transcription and analysis"""
        if self.whisper_model is None:
            return {"error": "Whisper not available"}
        
        try:
            result = self.whisper_model.transcribe(audio_path)
            return {
                "text": result["text"],
                "language": result["language"],
                "segments": len(result.get("segments", [])),
                "duration": result.get("duration", 0)
            }
        except Exception as e:
            logger.error(f"Whisper processing failed: {e}")
            return {"error": str(e)}

def get_comprehensive_system_info():
    """Get comprehensive system information"""
    info = {
        'Python': sys.version.split()[0],
        'Platform': sys.platform,
        'CPU Cores': psutil.cpu_count(),
        'CPU Usage': f"{psutil.cpu_percent()}%",
        'Memory Usage': f"{psutil.virtual_memory().percent}%",
        'Available Memory': f"{psutil.virtual_memory().available / (1024**3):.1f} GB",
        'GPU': 'CUDA Available' if torch.cuda.is_available() else 'MPS Available' if torch.backends.mps.is_available() else 'CPU Only',
        'PyTorch': torch.__version__,
        'OpenCV': cv2.__version__,
        'NumPy': np.__version__,
    }
    
    # Add availability of enhanced features
    enhanced_features = {
        'Real-ESRGAN': 'Available' if REALESRGAN_AVAILABLE else 'Not Available',
        'GFPGAN': 'Available' if GFPGAN_AVAILABLE else 'Not Available',
        'TTS (XTTS)': 'Available' if TTS_AVAILABLE else 'Not Available',
        'pyttsx3': 'Available' if PYTTSX3_AVAILABLE else 'Not Available',
        'face_recognition': 'Available' if FACE_RECOGNITION_AVAILABLE else 'Not Available',
        'Whisper': 'Available' if WHISPER_AVAILABLE else 'Not Available',
        'librosa': 'Available' if LIBROSA_AVAILABLE else 'Not Available',
        'SSIM Metrics': 'Available' if SSIM_AVAILABLE else 'Not Available'
    }
    
    info.update(enhanced_features)
    return "\n".join([f"{k}: {v}" for k, v in info.items()])

def check_enhanced_system_status():
    """Check enhanced system status with all components"""
    status = {
        "system": "✅ System Online",
        "memory": f"📊 Memory: {psutil.virtual_memory().percent}%",
        "cpu": f"⚡ CPU: {psutil.cpu_percent()}%",
        "gpu": "🎮 GPU: " + ("CUDA" if torch.cuda.is_available() else "MPS" if torch.backends.mps.is_available() else "CPU"),
        "models": []
    }
    
    # Check model availability
    if REALESRGAN_AVAILABLE:
        status["models"].append("🔍 Real-ESRGAN: Ready")
    if GFPGAN_AVAILABLE:
        status["models"].append("👤 GFPGAN: Ready")
    if TTS_AVAILABLE:
        status["models"].append("🎤 XTTS: Ready")
    if WHISPER_AVAILABLE:
        status["models"].append("👂 Whisper: Ready")
    if FACE_RECOGNITION_AVAILABLE:
        status["models"].append("😊 Face Recognition: Ready")
    
    if not status["models"]:
        status["models"].append("⚠️ Using fallback methods")
    
    return status

def get_placeholder_files():
    """Get placeholder files from results directory"""
    results_dir = Path("results")
    
    # Default placeholders
    default_audio = None
    default_image = None
    default_video = None
    
    if results_dir.exists():
        # Look for audio files
        for ext in ['.wav', '.mp3', '.m4a', '.aac']:
            audio_files = list(results_dir.glob(f"*{ext}"))
            if audio_files:
                default_audio = str(audio_files[0])
                break
        
        # Look for image files
        for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
            image_files = list(results_dir.glob(f"*{ext}"))
            if image_files:
                default_image = str(image_files[0])
                break
        
        # Look for video files
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
            video_files = list(results_dir.glob(f"*{ext}"))
            if video_files:
                default_video = str(video_files[0])
                break
    
    return default_audio, default_image, default_video
def main():
    """Main application with enhanced interface"""
    processor = EnhancedDeepfakeProcessor()
    
    # Enhanced CSS with all features
    custom_css = """
    <style>
    .framework-btn { 
        margin: 5px; 
        min-height: 50px;
        font-size: 16px;
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
    }
    .status-info {
        background: #44efd645;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #44efd6;
    }
    .quality-metrics {
        background: linear-gradient(135deg, #e8f4fd 0%, #f0f8ff 100%);
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        border: 1px solid #b3d9ff;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .enhancement-panel {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border: 1px solid #dee2e6;
    }
    .model-status {
        background: #e8f5e8;
        padding: 10px;
        border-radius: 6px;
        margin: 5px 0;
        border-left: 3px solid #28a745;
    }
    video { 
        max-height: 600px; 
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    audio { 
        max-width: 500px;
        border-radius: 8px;
    }
    .gradio-container {
        max-width: 1400px !important;
    }
    .tab-nav {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    </style>
    """
    
    # Get placeholder files
    default_audio, default_image, default_video = get_placeholder_files()
    
    # Check system status
    system_status = check_enhanced_system_status()
    
    with gr.Blocks(
        title="Mac NetNavi - Enhanced Deepfake Studio", 
        css=custom_css, 
        analytics_enabled=False,
        theme=gr.themes.Soft()
    ) as demo:
        
        gr.Markdown("""
        # 🚀 Mac NetNavi - Enhanced Deepfake Studio
        ### Advanced AI-powered deepfake generation with professional quality enhancement
        
        **Features:** Real-ESRGAN upscaling • GFPGAN face restoration • XTTS voice cloning • Whisper audio processing • Quality metrics
        """)
        
        with gr.Tab("🎭 Enhanced Face Swap"):
            with gr.Row():
                with gr.Column(scale=1):
                    source_img = gr.Image(
                        label="Source Face Image", 
                        type="filepath",
                        value=default_image
                    )
                    
                    with gr.Group(elem_classes=["enhancement-panel"]):
                        gr.Markdown("### Enhancement Options")
                        enhance_quality = gr.Checkbox(
                            label="🔍 Enable Face Enhancement (GFPGAN)", 
                            value=True
                        )
                        upscale_enabled = gr.Checkbox(
                            label="📈 Enable Upscaling (Real-ESRGAN)", 
                            value=REALESRGAN_AVAILABLE
                        )
                        target_res = gr.Dropdown(
                            choices=["1280x720", "1920x1080", "2560x1440", "3840x2160"],
                            value="1920x1080",
                            label="🎯 Output Resolution"
                        )
                        quality_preset = gr.Dropdown(
                            choices=["auto", "fast", "balanced", "high", "ultra"],
                            value="auto",
                            label="🎯 M3 Quality Preset"
                        )
                        
                        gr.Markdown("""
                        **M3 Presets:**
                        - **Auto**: Smart detection based on video
                        - **Fast**: 720p, 15fps processing, 2-3x speed
                        - **Balanced**: 1080p, optimal quality/speed
                        - **High**: 1440p, enhanced quality
                        - **Ultra**: 4K, maximum quality
                        """)
                        
                        enable_preview = gr.Checkbox(
                            label="⚡ Generate Preview First", 
                            value=True
                        )
                        
                        processing_method = gr.Dropdown(
                            choices=["m3_optimized", "chunked_parallel", "streaming", "professional_fallback"],
                            value="chunked_parallel",
                            label="🔧 Processing Method"
                        )
                        
                        gr.Markdown("""
                        **Processing Methods:**
                        - **M3 Optimized**: Full M3 acceleration (best quality)
                        - **Chunked Parallel**: Process in chunks (best speed)
                        - **Streaming**: Memory efficient (large videos)
                        - **Professional Fallback**: Maximum compatibility
                        """)
                        
                        thermal_management = gr.Checkbox(
                            label="🌡️ Enable Thermal Management", 
                            value=True
                        )
                
                with gr.Column(scale=1):
                    target_vid = gr.Video(
                        label="Target Video",
                        value=default_video
                    )
            
            with gr.Row():
                swap_btn = gr.Button(
                    "🎭 Start Enhanced Face Swap", 
                    variant="primary",
                    elem_classes=["framework-btn"],
                    size="lg"
                )
                
            with gr.Row():
                with gr.Column(scale=2):
                    output_video = gr.Video(
                        label="🎬 Enhanced Output Video",
                        show_download_button=True
                    )
                with gr.Column(scale=1):
                    # Real-time frame tracking display
                    with gr.Group():
                        gr.Markdown("### 📊 Real-Time Processing Status")
                        
                        frame_progress = gr.Textbox(
                            label="🎬 Frame Progress",
                            value="Ready to process...",
                            interactive=False
                        )
                        
                        processing_speed = gr.Textbox(
                            label="⚡ Processing Speed",
                            value="0 FPS",
                            interactive=False
                        )
                        
                        time_remaining = gr.Textbox(
                            label="⏱️ Time Remaining",
                            value="--:--",
                            interactive=False
                        )
                        
                        system_stats = gr.Textbox(
                            label="🖥️ System Stats",
                            value="Memory: -- | GPU: -- | Thermal: Normal",
                            interactive=False
                        )
                        
                        # Auto-refresh every 2 seconds
                        refresh_timer = gr.Timer(2)
                    
                    quality_metrics = gr.JSON(
                        label="📊 Quality Metrics & Analysis",
                        elem_classes=["quality-metrics"]
                    )
        
        with gr.Tab("🎤 Enhanced Voice Clone"):
            with gr.Row():
                with gr.Column():
                    input_text = gr.Textbox(
                        label="📝 Text to Speak", 
                        lines=5,
                        placeholder="Enter the text you want to convert to speech..."
                    )
                    
                    ref_audio = gr.Audio(
                        label="🎵 Reference Voice (Optional)", 
                        type="filepath",
                        value=default_audio
                    )
                    
                    with gr.Group(elem_classes=["enhancement-panel"]):
                        gr.Markdown("### Voice Options")
                        voice_method = gr.Dropdown(
                            choices=["XTTS (Best Quality)", "pyttsx3 (Fast)", "System TTS (Fallback)"],
                            value="XTTS (Best Quality)" if TTS_AVAILABLE else "System TTS (Fallback)",
                            label="🔊 Voice Engine"
                        )
                        voice_speed = gr.Slider(
                            minimum=0.5, maximum=2.0, value=1.0,
                            label="⚡ Speech Speed"
                        )
                
                with gr.Column():
                    voice_btn = gr.Button(
                        "🎤 Generate Enhanced Voice", 
                        variant="primary",
                        elem_classes=["framework-btn"],
                        size="lg"
                    )
                    
                    output_audio = gr.Audio(
                        label="🎵 Generated Voice Output",
                        show_download_button=True
                    )
                    
                    if WHISPER_AVAILABLE:
                        audio_analysis = gr.JSON(
                            label="👂 Audio Analysis (Whisper)",
                            elem_classes=["quality-metrics"]
                        )
        
        with gr.Tab("📊 System Status & Models"):
            with gr.Row():
                with gr.Column():
                    system_info = gr.Textbox(
                        label="🖥️ Comprehensive System Information",
                        value=get_comprehensive_system_info(),
                        interactive=False,
                        lines=15,
                        elem_classes=["status-info"]
                    )
                    
                    refresh_btn = gr.Button(
                        "🔄 Refresh System Status", 
                        variant="secondary"
                    )
                
                with gr.Column():
                    model_status = gr.JSON(
                        label="🤖 Enhanced Models Status",
                        value=system_status,
                        elem_classes=["model-status"]
                    )
                    
                    gr.Markdown("""
                    ### 🎯 Performance Tips
                    - **Real-ESRGAN**: Best for upscaling and detail enhancement
                    - **GFPGAN**: Professional face restoration and enhancement  
                    - **XTTS**: High-quality voice cloning with reference audio
                    - **Whisper**: Advanced audio transcription and analysis
                    - **MPS/CUDA**: Hardware acceleration for faster processing
                    """)
        
        with gr.Tab("⚙️ Advanced Settings"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🔧 Processing Settings")
                    
                    batch_size = gr.Slider(
                        minimum=1, maximum=32, value=4,
                        label="📦 Batch Size"
                    )
                    
                    memory_optimization = gr.Checkbox(
                        label="🧠 Memory Optimization", 
                        value=True
                    )
                    
                    temporal_consistency = gr.Checkbox(
                        label="🎬 Temporal Consistency", 
                        value=True
                    )
                    
                    output_format = gr.Dropdown(
                        choices=["MP4", "AVI", "MOV", "MKV"],
                        value="MP4",
                        label="📁 Output Format"
                    )
                
                with gr.Column():
                    gr.Markdown("### 📈 Quality Settings")
                    
                    enhancement_strength = gr.Slider(
                        minimum=0.1, maximum=2.0, value=1.0,
                        label="💪 Enhancement Strength"
                    )
                    
                    noise_reduction = gr.Slider(
                        minimum=0, maximum=10, value=5,
                        label="🔇 Noise Reduction"
                    )
                    
                    color_correction = gr.Checkbox(
                        label="🎨 Color Correction", 
                        value=True
                    )
                    
                    sharpening = gr.Slider(
                        minimum=0, maximum=5, value=2,
                        label="🔍 Sharpening"
                    )
        
        # Event handlers with enhanced functionality
        def process_enhanced_face_swap(source, target, enhance, upscale, resolution, preset, preview_enabled, thermal_enabled, method):
            if not source or not target:
                return None, {"error": "Missing input files"}
            
            # Validate file paths
            import os
            if not os.path.isfile(source) or not os.path.isfile(target):
                return None, {"error": "Invalid file paths provided"}
            
            # Reset frame tracker
            frame_tracker.reset_status()
            
            output_path = tempfile.mktemp(suffix=".mp4")
            
            # Choose processing method
            if method == "chunked_parallel":
                result_path, metrics = processor.chunked_processor.process_video_chunked(source, target, output_path)
            elif method == "streaming":
                result_path, metrics = processor.streaming_processor.process_streaming(source, target, output_path)
            elif method == "m3_optimized":
                result_path, metrics = processor.enhanced_face_swap(source, target, output_path, enhance, preset)
            else:  # professional_fallback
                result_path, metrics = processor.professional_engine_fallback(source, target, output_path, enhance)
            
            # Add processing info to metrics
            if isinstance(metrics, dict) and 'error' not in metrics:
                metrics.update({
                    'enhancement_enabled': enhance,
                    'upscaling_enabled': upscale,
                    'quality_preset': preset,
                    'preview_enabled': preview_enabled,
                    'thermal_management': thermal_enabled,
                    'processing_method': method,
                    'processing_time': time.time(),
                    'm3_optimized': method == "m3_optimized"
                })
            
            return result_path, metrics
        
        def get_real_time_status():
            """Get real-time processing status for UI updates"""
            try:
                status = frame_tracker.load_status()
                
                if status["processing"]:
                    progress_text = f"{status['current_frame']}/{status['total_frames']} frames"
                    speed_text = f"{status['frames_per_second']:.1f} FPS"
                    
                    remaining_mins = status['estimated_remaining'] / 60
                    time_text = f"{remaining_mins:.1f} minutes remaining"
                    
                    memory_gb = status['memory_usage'] / 1024 if status['memory_usage'] > 0 else 0
                    stats_text = f"Memory: {memory_gb:.1f}GB | GPU: {status['gpu_usage']}% | Thermal: {status['thermal_state']}"
                else:
                    progress_text = "Ready to process..."
                    speed_text = "0 FPS"
                    time_text = "--:--"
                    stats_text = "Memory: -- | GPU: -- | Thermal: Normal"
                
                return progress_text, speed_text, time_text, stats_text
                
            except Exception as e:
                return "Status unavailable", "0 FPS", "--:--", "Error loading stats"
        
        def process_enhanced_voice_clone(text, reference, method, speed):
            if not text.strip():
                return None, None
            
            output_path = tempfile.mktemp(suffix=".wav")
            result_path = processor.enhanced_voice_clone(text, reference, output_path)
            
            # Analyze audio if Whisper is available
            analysis = None
            if WHISPER_AVAILABLE and result_path and os.path.exists(result_path):
                analysis = processor.process_audio_with_whisper(result_path)
            
            return result_path, analysis
        
        # Connect event handlers
        swap_btn.click(
            process_enhanced_face_swap,
            inputs=[source_img, target_vid, enhance_quality, upscale_enabled, target_res, quality_preset, enable_preview, thermal_management, processing_method],
            outputs=[output_video, quality_metrics]
        )
        
        # Real-time status updates
        refresh_timer.tick(
            get_real_time_status,
            outputs=[frame_progress, processing_speed, time_remaining, system_stats]
        )
        
        voice_btn.click(
            process_enhanced_voice_clone,
            inputs=[input_text, ref_audio, voice_method, voice_speed],
            outputs=[output_audio, audio_analysis if WHISPER_AVAILABLE else gr.JSON(visible=False)]
        )
        
        refresh_btn.click(
            lambda: [get_comprehensive_system_info(), check_enhanced_system_status()],
            outputs=[system_info, model_status]
        )
    
    # Launch with enhanced settings
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True,
        quiet=False
    )

if __name__ == "__main__":
    logger.info("Starting Mac NetNavi Enhanced Deepfake Studio...")
    logger.info(f"Available enhanced features: {sum([REALESRGAN_AVAILABLE, GFPGAN_AVAILABLE, TTS_AVAILABLE, WHISPER_AVAILABLE, FACE_RECOGNITION_AVAILABLE])}/5")
    main()
