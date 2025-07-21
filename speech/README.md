# Speech Processing with GenAI

This directory contains implementations for speech-to-text, text-to-speech, and audio generation using generative AI models.

## Contents

### 1. Speech-to-Text
- **Whisper Integration**: OpenAI Whisper for accurate transcription
- **Real-time Transcription**: Live audio processing
- **Multi-language Support**: Transcription in 100+ languages
- **Custom Fine-tuning**: Domain-specific speech recognition

### 2. Text-to-Speech
- **Bark Integration**: Realistic voice synthesis
- **ElevenLabs API**: High-quality voice cloning
- **Voice Customization**: Create custom voice profiles
- **Emotional Speech**: Control tone and emotion

### 3. Audio Generation
- **Music Generation**: AI-powered music creation
- **Sound Effects**: Generate custom audio effects
- **Voice Conversion**: Transform voice characteristics
- **Audio Enhancement**: Noise reduction and quality improvement

## Quick Start

```python
# Speech-to-Text with Whisper
import whisper
model = whisper.load_model("base")
result = model.transcribe("audio.wav")
print(result["text"])

# Text-to-Speech with Bark
from bark import SAMPLE_RATE, generate_audio, preload_models
audio_array = generate_audio("Hello, this is a test")
```

## Examples

- [`voice_assistant.ipynb`](voice_assistant.ipynb) - Complete voice assistant implementation
- [`speech_translation.py`](speech_translation.py) - Real-time speech translation
- [`audio_generation.py`](audio_generation.py) - Music and sound generation
- [`voice_cloning.ipynb`](voice_cloning.ipynb) - Custom voice creation

## Requirements

- whisper-openai
- bark
- elevenlabs
- torchaudio
- librosa
- soundfile
