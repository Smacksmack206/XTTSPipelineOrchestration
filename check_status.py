#!/usr/bin/env python3

import os
import time
from datetime import datetime

def get_processing_status():
    """Get current processing status from logs"""
    log_file = "deepfake_processing.log"
    
    if not os.path.exists(log_file):
        return "❌ No processing log found - app may not be running"
    
    try:
        # Get last 20 lines of log
        with open(log_file, 'r') as f:
            lines = f.readlines()
            recent_lines = lines[-20:] if len(lines) > 20 else lines
        
        if not recent_lines:
            return "📝 Log file is empty"
        
        # Parse recent activity
        status_info = []
        current_operation = None
        latest_progress = None
        latest_status = None
        errors = []
        
        for line in recent_lines:
            line = line.strip()
            if "STARTED:" in line:
                current_operation = line.split("STARTED: ")[1]
            elif "PROGRESS:" in line:
                latest_progress = line.split("PROGRESS: ")[1]
            elif "STATUS:" in line:
                latest_status = line.split("STATUS: ")[1]
            elif "ERROR:" in line:
                errors.append(line.split("ERROR: ")[1])
            elif "COMPLETED:" in line:
                current_operation = "✅ " + line.split("COMPLETED: ")[1]
        
        # Build status report
        status_info.append(f"📊 **Current Processing Status**")
        status_info.append(f"⏰ Last updated: {datetime.now().strftime('%H:%M:%S')}")
        
        if current_operation:
            status_info.append(f"🔄 Operation: {current_operation}")
        
        if latest_progress:
            status_info.append(f"📈 Progress: {latest_progress}")
            
        if latest_status:
            status_info.append(f"ℹ️  Status: {latest_status}")
        
        if errors:
            status_info.append(f"❌ Recent errors: {len(errors)}")
            for error in errors[-3:]:  # Show last 3 errors
                status_info.append(f"   • {error}")
        
        # Check if process is still running
        try:
            import psutil
            python_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                if proc.info['name'] and 'python' in proc.info['name'].lower():
                    if proc.info['cmdline'] and any('app_upgraded.py' in cmd for cmd in proc.info['cmdline']):
                        python_processes.append(proc.info['pid'])
            
            if python_processes:
                status_info.append(f"🟢 App running (PID: {python_processes[0]})")
            else:
                status_info.append(f"🔴 App not running")
                
        except ImportError:
            status_info.append("⚠️  Cannot check process status (psutil not available)")
        
        return "\n".join(status_info)
        
    except Exception as e:
        return f"❌ Error reading log: {e}"

def tail_log(lines=10):
    """Show last N lines of processing log"""
    log_file = "deepfake_processing.log"
    
    if not os.path.exists(log_file):
        return "❌ No processing log found"
    
    try:
        with open(log_file, 'r') as f:
            all_lines = f.readlines()
            recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
        
        return "📝 **Recent Log Entries:**\n" + "".join(recent_lines)
        
    except Exception as e:
        return f"❌ Error reading log: {e}"

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "tail":
        lines = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        print(tail_log(lines))
    else:
        print(get_processing_status())
