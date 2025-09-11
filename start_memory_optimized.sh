#!/bin/bash

echo "🚀 Starting Mac NetNavi (Memory Optimized)"

# Install all requirements
pip install -r requirements_memory_optimized.txt

# Check available memory
AVAILABLE_RAM=$(python3 -c "import psutil; print(int(psutil.virtual_memory().available/1024/1024/1024))")
echo "💾 Available RAM: ${AVAILABLE_RAM}GB"

if [ $AVAILABLE_RAM -lt 4 ]; then
    echo "⚠️  Low memory detected. Consider closing other applications."
fi

# Start memory monitor in background
python3 memory_monitor.py &
MONITOR_PID=$!

# Function to cleanup on exit
cleanup() {
    echo "🧹 Cleaning up..."
    kill $MONITOR_PID 2>/dev/null
    exit
}

# Set trap to cleanup on script exit
trap cleanup EXIT INT TERM

# Start the optimized app
echo "🎭 Launching Mac NetNavi..."
python3 app_memory_optimized.py

# Cleanup will be called automatically by trap
