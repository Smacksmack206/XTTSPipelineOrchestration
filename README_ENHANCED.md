# 🚀 Mac NetNavi Enhanced

Advanced AI Digital Twin with dual optimization paths for M3 MacBook Air.

## 🎯 Two Optimized Paths

### 🔥 Basic Path
- **Speed**: Ultra-fast generation (30-60s)
- **AI**: Ollama with lightweight models
- **Voice**: XTTS high-quality cloning
- **Video**: Static image with audio
- **Use Case**: Quick prototypes, testing

### ⚡ Advanced Path  
- **Quality**: Maximum fidelity output
- **AI**: Multiple backends (Ollama + Transformers)
- **Voice**: XTTS + Coqui options
- **Video**: Animated with mouth movement
- **Features**: Driving video support
- **Use Case**: Production-quality twins

## 🧠 Local AI Models

### Ollama Models (Recommended)
- **llama3.2:3b** - Fastest, 2GB RAM
- **llama3.2** - Balanced, 4GB RAM  
- **mistral** - Creative responses, 4GB RAM
- **phi3** - Efficient reasoning, 2GB RAM
- **codellama** - Technical content

### Transformers Models (Offline)
- **DialoGPT-small** - Fast chat responses
- **DialoGPT-medium** - Better conversations
- **BlenderBot-400M** - Optimized chatbot

## 🎤 Voice Options

### High Quality (XTTS)
- Perfect voice cloning
- Any language support
- 10-15s generation time
- Requires voice sample

### Fast (Coqui TTS)
- Quick generation (3-5s)
- Good quality synthetic voice
- No voice sample needed
- Multiple voice options

## 🎭 Video Generation

### Static Video
- Image + audio combination
- Instant generation
- Perfect lip-sync timing
- Professional quality

### Animated Video
- Face detection + mouth animation
- Audio-driven movement
- Driving video support
- Realistic expressions

## 🚀 Quick Start

```bash
# Enhanced setup
./setup_enhanced.sh

# Run enhanced version
python app_enhanced.py
```

## Cloning the Repository

To get a local copy up and running, follow these simple steps.

### Prerequisites

You need to have `git` installed on your system.

### Installation

1.  Clone the repo
    ```sh
    git clone git@github.com:Smacksmack206/XTTSPipelineOrchestration.git
    ```
2.  Navigate to the repository directory
    ```sh
    cd XTTSPipelineOrchestration
    ```
3.  Initialize and update submodules
    ```sh
    git submodule update --init --recursive
    ```

## 📊 Performance Comparison

| Feature | Basic Path | Advanced Path |
|---------|------------|---------------|
| Generation Time | 30-60s | 60-120s |
| AI Quality | Good | Excellent |
| Voice Quality | High | Customizable |
| Video Quality | Static | Animated |
| RAM Usage | 2-4GB | 4-8GB |
| CPU Usage | Medium | High |

## 🛠️ Advanced Features

### Multi-Modal Input
- Portrait images (any format)
- Voice samples (wav/mp3/m4a)
- Driving videos (mp4/mov)
- Text prompts

### AI Customization
- Model selection per use case
- Temperature/creativity control
- Response length limits
- Context awareness

### Output Options
- Multiple video formats
- Audio export (wav/mp3)
- Batch processing
- History tracking

## 🔧 Optimization Tips

### For Speed (Basic Path)
```python
# Use lightweight model
model = "llama3.2:3b"
tts_quality = "Fast (Coqui)"
video_type = "Static"
```

### For Quality (Advanced Path)
```python
# Use full model
model = "llama3.2"
tts_quality = "High Quality (XTTS)"
video_type = "Animated"
```

## 🎯 Use Cases

### Basic Path
- Rapid prototyping
- Voice message creation
- Simple avatars
- Testing concepts

### Advanced Path
- Professional presentations
- Marketing content
- Educational videos
- Entertainment

## 🚨 Troubleshooting

### Ollama Issues
```bash
# Check service
ollama list
ollama serve

# Reinstall model
ollama rm llama3.2:3b
ollama pull llama3.2:3b
```

### XTTS Issues
```bash
# Reinstall TTS
pip uninstall TTS
pip install TTS --no-cache-dir
```

### Performance Issues
```bash
# Monitor resources
top -pid $(pgrep -f "python app_enhanced.py")

# Reduce model size
# Use llama3.2:3b instead of llama3.2
```

## 📈 Roadmap

- [ ] Real-time generation
- [ ] Custom model training
- [ ] Multi-language support
- [ ] API endpoints
- [ ] Mobile optimization