#!/usr/bin/env python3

import json
import time
import threading
from datetime import datetime
from pathlib import Path

class RealTimeFrameTracker:
    def __init__(self):
        self.status_file = Path("frame_status.json")
        self.lock = threading.Lock()
        self.reset_status()
        
    def reset_status(self):
        """Reset tracking status"""
        status = {
            "processing": False,
            "current_frame": 0,
            "total_frames": 0,
            "fps": 0,
            "elapsed_time": 0,
            "estimated_remaining": 0,
            "frames_per_second": 0,
            "last_frame_time": 0,
            "quality_metrics": {"ssim": 0, "psnr": 0},
            "memory_usage": 0,
            "gpu_usage": 0,
            "thermal_state": "normal",
            "current_operation": "idle",
            "preview_path": "",
            "error_count": 0,
            "last_update": datetime.now().isoformat()
        }
        self.save_status(status)
        
    def save_status(self, status):
        """Save status to JSON file"""
        with self.lock:
            with open(self.status_file, 'w') as f:
                json.dump(status, f, indent=2)
    
    def load_status(self):
        """Load current status"""
        try:
            with open(self.status_file, 'r') as f:
                return json.load(f)
        except:
            self.reset_status()
            return self.load_status()
    
    def start_processing(self, total_frames, operation="Face Swap"):
        """Start processing tracking"""
        status = self.load_status()
        status.update({
            "processing": True,
            "current_frame": 0,
            "total_frames": total_frames,
            "start_time": time.time(),
            "current_operation": operation,
            "error_count": 0,
            "last_update": datetime.now().isoformat()
        })
        self.save_status(status)
    
    def update_frame(self, frame_num, quality_metrics=None):
        """Update frame progress"""
        status = self.load_status()
        current_time = time.time()
        
        if status["processing"]:
            elapsed = current_time - status.get("start_time", current_time)
            frames_processed = frame_num - status.get("last_frame", 0)
            time_diff = current_time - status.get("last_frame_time", current_time)
            
            if time_diff > 0:
                fps = frames_processed / time_diff
            else:
                fps = 0
            
            remaining_frames = status["total_frames"] - frame_num
            estimated_remaining = remaining_frames / fps if fps > 0 else 0
            
            status.update({
                "current_frame": frame_num,
                "elapsed_time": elapsed,
                "estimated_remaining": estimated_remaining,
                "frames_per_second": fps,
                "last_frame_time": current_time,
                "last_frame": frame_num,
                "last_update": datetime.now().isoformat()
            })
            
            if quality_metrics:
                status["quality_metrics"] = quality_metrics
                
            self.save_status(status)
    
    def update_system_stats(self, memory_mb, gpu_percent, thermal_state):
        """Update system statistics"""
        status = self.load_status()
        status.update({
            "memory_usage": memory_mb,
            "gpu_usage": gpu_percent,
            "thermal_state": thermal_state,
            "last_update": datetime.now().isoformat()
        })
        self.save_status(status)
    
    def set_preview(self, preview_path):
        """Set preview file path"""
        status = self.load_status()
        status["preview_path"] = preview_path
        status["last_update"] = datetime.now().isoformat()
        self.save_status(status)
    
    def add_error(self):
        """Increment error count"""
        status = self.load_status()
        status["error_count"] += 1
        status["last_update"] = datetime.now().isoformat()
        self.save_status(status)
    
    def finish_processing(self):
        """Mark processing as complete"""
        status = self.load_status()
        status.update({
            "processing": False,
            "current_operation": "completed",
            "last_update": datetime.now().isoformat()
        })
        self.save_status(status)

# Global tracker instance
frame_tracker = RealTimeFrameTracker()
