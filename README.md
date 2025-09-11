# 🚀 Mac NetNavi

An AI-powered tool to create digital twins with animated faces from an image and audio.

## 🚀 Quick Start

1.  **Provide your media:**
    *   Place your source image (e.g., `my_face.png`) and source audio (e.g., `my_voice.wav`) in the `results` directory.
    *   (Optional) If you want to transfer the face onto a video, also place your source video (e.g., `my_video.mp4`) in the `results` directory.

2.  **Run the application:**
    The application will automatically pick up the files from the `results` directory.

    ```bash
    python app_memory_optimized.py
    ```

    *   If a source video is provided, the face from the source image will be swapped onto the video.
    *   If no source video is provided, a new video will be generated with the source image's face animated according to the audio.

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

## 🧠 AI Models

This project can use various local AI models for different tasks.

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

## 📈 Roadmap

- [ ] Real-time generation
- [ ] Custom model training
- [ ] Multi-language support
- [ ] API endpoints
- [ ] Mobile optimization
