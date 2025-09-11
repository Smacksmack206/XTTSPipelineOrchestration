#!/usr/bin/env python3
import psutil
import time
import sys

def monitor_memory():
    """Monitor system memory usage"""
    while True:
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=1)
        
        print(f"\rRAM: {memory.percent:.1f}% | CPU: {cpu:.1f}% | Available: {memory.available/1024/1024/1024:.1f}GB", end="")
        
        if memory.percent > 80:
            print(f"\n⚠️  HIGH MEMORY USAGE: {memory.percent:.1f}%")
        
        time.sleep(2)

if __name__ == "__main__":
    try:
        monitor_memory()
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
        sys.exit(0)
