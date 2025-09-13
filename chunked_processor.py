#!/usr/bin/env python3

import cv2
import numpy as np
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import logging
from real_time_tracker import frame_tracker

logger = logging.getLogger(__name__)

class ChunkedVideoProcessor:
    def __init__(self, chunk_size=30, max_workers=4):
        self.chunk_size = chunk_size  # Process 30 frames at a time
        self.max_workers = max_workers
        self.processing_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
    def split_video_into_chunks(self, video_path: str) -> list:
        """Split video into manageable chunks"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        chunks = []
        for start_frame in range(0, total_frames, self.chunk_size):
            end_frame = min(start_frame + self.chunk_size, total_frames)
            chunks.append({
                'start': start_frame,
                'end': end_frame,
                'frames': end_frame - start_frame,
                'duration': (end_frame - start_frame) / fps
            })
        
        cap.release()
        logger.info(f"Split video into {len(chunks)} chunks of ~{self.chunk_size} frames")
        return chunks
    
    def process_chunk(self, video_path: str, source_image: np.ndarray, chunk_info: dict, chunk_id: int) -> list:
        """Process a single chunk of frames"""
        try:
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, chunk_info['start'])
            
            processed_frames = []
            
            for frame_idx in range(chunk_info['frames']):
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process frame with face swap
                processed_frame = self.process_single_frame(source_image, frame_rgb)
                processed_frames.append(processed_frame)
                
                # Update real-time tracking
                global_frame_num = chunk_info['start'] + frame_idx
                frame_tracker.update_frame(global_frame_num)
                
                # Log progress every 10 frames
                if frame_idx % 10 == 0:
                    logger.info(f"Chunk {chunk_id}: Frame {frame_idx}/{chunk_info['frames']}")
            
            cap.release()
            logger.info(f"Completed chunk {chunk_id}: {len(processed_frames)} frames")
            return processed_frames
            
        except Exception as e:
            logger.error(f"Chunk {chunk_id} processing failed: {e}")
            frame_tracker.add_error()
            return []
    
    def process_single_frame(self, source_image: np.ndarray, target_frame: np.ndarray) -> np.ndarray:
        """Process single frame - simplified for speed"""
        try:
            # Use professional engine for face swapping
            from professional_deepfake_engine import ProfessionalDeepfakeEngine
            engine = ProfessionalDeepfakeEngine()
            return engine.professional_face_swap(source_image, target_frame)
        except Exception as e:
            logger.error(f"Frame processing failed: {e}")
            return target_frame
    
    def merge_chunks(self, chunk_results: list, output_path: str, fps: float, resolution: tuple):
        """Merge processed chunks back into video"""
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, resolution)
            
            total_frames_written = 0
            
            for chunk_frames in chunk_results:
                for frame in chunk_frames:
                    if frame.shape[:2] != (resolution[1], resolution[0]):
                        frame = cv2.resize(frame, resolution)
                    
                    out_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(out_frame)
                    total_frames_written += 1
            
            out.release()
            logger.info(f"Merged {total_frames_written} frames into {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Chunk merging failed: {e}")
            return ""
    
    def process_video_chunked(self, source_image_path: str, target_video_path: str, output_path: str) -> tuple:
        """Process video using chunked approach"""
        try:
            # Load source image
            source_img = cv2.imread(source_image_path)
            source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
            
            # Get video info
            cap = cv2.VideoCapture(target_video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            # Start tracking
            frame_tracker.start_processing(total_frames, "Chunked Face Swap")
            
            # Split into chunks
            chunks = self.split_video_into_chunks(target_video_path)
            
            # Process chunks in parallel
            chunk_results = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                
                for i, chunk in enumerate(chunks):
                    future = executor.submit(
                        self.process_chunk, 
                        target_video_path, 
                        source_img, 
                        chunk, 
                        i
                    )
                    futures.append(future)
                
                # Collect results in order
                for i, future in enumerate(futures):
                    try:
                        result = future.result(timeout=300)  # 5 minute timeout per chunk
                        chunk_results.append(result)
                        logger.info(f"Chunk {i} completed successfully")
                    except Exception as e:
                        logger.error(f"Chunk {i} failed: {e}")
                        chunk_results.append([])  # Empty chunk on failure
            
            # Merge chunks
            final_output = self.merge_chunks(chunk_results, output_path, fps, (width, height))
            
            # Finish tracking
            frame_tracker.finish_processing()
            
            metrics = {
                'total_frames': total_frames,
                'chunks_processed': len(chunks),
                'fps': fps,
                'resolution': f"{width}x{height}",
                'processing_method': 'chunked_parallel'
            }
            
            return final_output, metrics
            
        except Exception as e:
            logger.error(f"Chunked processing failed: {e}")
            frame_tracker.add_error()
            return "", {"error": str(e)}

# Memory-efficient streaming processor
class StreamingProcessor:
    def __init__(self, buffer_size=10):
        self.buffer_size = buffer_size
        
    def process_streaming(self, source_image_path: str, target_video_path: str, output_path: str) -> tuple:
        """Process video with minimal memory footprint"""
        try:
            # Load source image
            source_img = cv2.imread(source_image_path)
            source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
            
            # Setup video processing
            cap = cv2.VideoCapture(target_video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Start tracking
            frame_tracker.start_processing(total_frames, "Streaming Face Swap")
            
            frame_buffer = []
            processed_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    # Process remaining frames
                    if frame_buffer:
                        processed_frames = self.process_frame_batch(source_img, frame_buffer)
                        for pframe in processed_frames:
                            out_frame = cv2.cvtColor(pframe, cv2.COLOR_RGB2BGR)
                            out.write(out_frame)
                        processed_count += len(frame_buffer)
                    break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_buffer.append(frame_rgb)
                
                # Process when buffer is full
                if len(frame_buffer) >= self.buffer_size:
                    processed_frames = self.process_frame_batch(source_img, frame_buffer)
                    
                    for pframe in processed_frames:
                        out_frame = cv2.cvtColor(pframe, cv2.COLOR_RGB2BGR)
                        out.write(out_frame)
                    
                    processed_count += len(frame_buffer)
                    frame_tracker.update_frame(processed_count)
                    
                    frame_buffer = []  # Clear buffer
            
            cap.release()
            out.release()
            frame_tracker.finish_processing()
            
            metrics = {
                'total_frames': processed_count,
                'fps': fps,
                'resolution': f"{width}x{height}",
                'processing_method': 'streaming'
            }
            
            return output_path, metrics
            
        except Exception as e:
            logger.error(f"Streaming processing failed: {e}")
            return "", {"error": str(e)}
    
    def process_frame_batch(self, source_image: np.ndarray, frame_batch: list) -> list:
        """Process a batch of frames"""
        processed = []
        for frame in frame_batch:
            try:
                from professional_deepfake_engine import ProfessionalDeepfakeEngine
                engine = ProfessionalDeepfakeEngine()
                processed_frame = engine.professional_face_swap(source_image, frame)
                processed.append(processed_frame)
            except Exception as e:
                logger.error(f"Batch frame processing failed: {e}")
                processed.append(frame)  # Use original on failure
        return processed
