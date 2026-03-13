# Qwen3-TTS for Apple Silicon

**Run Qwen3-TTS text-to-speech locally on Apple Silicon. Voice cloning, voice design, and 9 built-in speakers — completely offline, no cloud APIs.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-Native-000000?logo=apple&logoColor=white)](https://support.apple.com/en-us/116943)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![MLX](https://img.shields.io/badge/MLX-Framework-FF6B00?logo=apple&logoColor=white)](https://ml-explore.github.io/mlx/)

## Features

- **Voice Cloning** — Clone any voice from a 5-second audio sample
- **Voice Design** — Create voices by describing them ("deep narrator", "excited child")
- **9 Built-in Speakers** — With emotion and speed control
- **100% Local** — Runs entirely on your Mac, no internet after model download
- **MLX Optimized** — Native GPU inference on M-series chips

## Why MLX?

| Metric | PyTorch | MLX |
|--------|---------|-----|
| RAM Usage | 10+ GB | 2-3 GB |
| CPU Temperature | 80-90C | 40-50C |

*Benchmarked on M4 MacBook Air (fanless) with 1.7B 8-bit models*

## Quick Start

```bash
git clone https://github.com/ExpertVagabond/qwen3-tts-apple-silicon.git
cd qwen3-tts-apple-silicon
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg
```

## Models

### Pro (1.7B — Best Quality)

| Model | Use Case | HuggingFace |
|-------|----------|-------------|
| CustomVoice | Preset speakers + emotion control | `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit` |
| VoiceDesign | Create voices from text description | `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit` |
| Base | Voice cloning from reference audio | `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit` |

### Lite (0.6B — Faster, Less RAM)

| Model | Use Case | HuggingFace |
|-------|----------|-------------|
| CustomVoice | Preset speakers + emotion | `mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit` |
| VoiceDesign | Voice from description | `mlx-community/Qwen3-TTS-12Hz-0.6B-VoiceDesign-8bit` |
| Base | Voice cloning | `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit` |

Download model folders from HuggingFace and place in `models/`.

## Usage

```bash
source .venv/bin/activate
python main.py
```

Interactive menu with 6 modes:
1. **Custom Voice** — Pick speaker, set emotion and speed, generate
2. **Voice Design** — Describe a voice, generate speech matching it
3. **Voice Cloning** — Clone from reference audio (saved voices, enroll new, quick clone)
4-6. Same modes with Lite models (faster)

## Tips

- Drag `.txt` files into terminal for long-form text
- Voice cloning works best with clean 5-10 second clips
- Audio auto-plays on macOS via `afplay`
- All output saved to `outputs/` organized by mode
- RAM: ~3 GB for Lite, ~6 GB for Pro

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- ffmpeg (`brew install ffmpeg`)

## Related

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) — Original model by Alibaba
- [MLX Audio](https://github.com/Blaizzy/mlx-audio) — MLX framework for audio
- [ai-music-mcp](https://github.com/ExpertVagabond/ai-music-mcp) — MCP server with TTS integration

## License

[MIT](LICENSE)

## Author

Built by [Purple Squirrel Media](https://purplesquirrelmedia.io)
