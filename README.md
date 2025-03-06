# Video Transcriber

A Python tool for transcribing videos and audio files with speaker diarization. This tool processes video or audio files, transcribes the speech to text, and identifies different speakers in the conversation.

## Features

- Support for both video and audio files
- Direct WAV file processing (no conversion needed)
- Video to audio extraction
- Speech-to-text transcription using Whisper
- Speaker diarization
- Multiple output formats (txt, srt, vtt)
- Progress tracking and timeout handling
- Hardware acceleration support (CUDA, MPS)
- Optimized parameters for different model sizes
- **NEW: Modular architecture for better maintainability**
- **NEW: Caching system for improved performance**

## Architecture

The project has been restructured into a modular architecture:

```
video_transcriber/
├── src/
│   ├── audio/         # Audio processing components
│   ├── transcription/ # Transcription engine
│   ├── diarization/   # Speaker diarization
│   ├── output/        # Output formatting
│   ├── cache/         # Caching system
│   ├── config.py      # Configuration handling
│   ├── transcriber.py # Main orchestrator
├── tests/
│   ├── unit/          # Unit tests
│   ├── integration/   # Integration tests
│   ├── fixtures/      # Test fixtures
```

### Key Components

- **AudioProcessor**: Handles audio extraction and processing
- **TranscriptionEngine**: Manages speech-to-text transcription
- **DiarizationEngine**: Handles speaker identification
- **OutputFormatter**: Formats transcripts in various output formats
- **CacheManager**: Manages caching of audio, transcription, and diarization results

## Caching System

The new caching system improves performance by:

- Caching extracted audio files to avoid repeated extraction
- Caching transcription results for previously processed files
- Caching diarization results for previously processed files
- Automatically managing cache expiration and size limits

Configure caching in the `.env` file:
```bash
CACHE_ENABLED=true       # Enable/disable caching
CACHE_EXPIRATION=604800  # Cache expiration in seconds (default: 7 days)
MAX_CACHE_SIZE=10737418240  # Maximum cache size in bytes (default: 10GB)
```

## Supported Formats

### Input Formats
- Video: mov, mp4, etc. (any format supported by MoviePy)
- Audio: wav (direct processing), mp3, m4a, aac (auto-converted to wav)

### Output Formats
- `txt`: Simple text format with speaker labels
- `srt`: SubRip subtitle format with timestamps
- `vtt`: WebVTT format for web video subtitles

## Whisper Models

The system supports different Whisper model sizes, each with its own trade-offs:

| Model | Size | Memory | Speed | Accuracy | Use Case |
|-------|------|---------|--------|-----------|-----------|
| tiny | ~75MB | Minimal | Fastest | Basic | Quick tests, simple audio |
| base | ~150MB | Low | Fast | Good | General use, clear audio |
| small | ~500MB | Medium | Moderate | Better | Professional use |
| medium | ~1.5GB | High | Slower | Very Good | Complex audio |
| large-v3 | ~3GB | Very High | Slowest | Best | Critical accuracy needs |

### Model-Specific Optimizations

- **Base Model**: Optimized for general use with balanced parameters
  - Default VAD settings
  - Standard beam size (5)
  - Good for most use cases

- **Medium/Large Models**: Enhanced parameters for better accuracy
  - Increased beam size (6)
  - Adjusted VAD parameters for better word boundary detection
  - Added speech padding to prevent word cutting
  - Context-aware processing with previous text conditioning
  - Optimized for conversation transcription

Choose your model in the `.env` file:
```bash
WHISPER_MODEL=base  # Options: tiny, base, small, medium, large-v3
```

## Requirements

- Python 3.8+
- FFmpeg (for video/audio processing)
- PyTorch
- Other dependencies listed in requirements.txt

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/video_transcriber.git
cd video_transcriber
```

2. Set up the environment and install dependencies:
```bash
make setup  # Creates venv and installs all dependencies
```
   Or manually:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Copy the example environment file and configure your settings:
```bash
cp .env.example .env
```

## Configuration

Edit the `.env` file to configure:

- `HF_TOKEN`: Your HuggingFace token for accessing models
- `WHISPER_MODEL`: Whisper model size (tiny, base, small, medium, large)
- `LANGUAGE`: Target language for transcription (default: en)
- `OUTPUT_FORMAT`: Transcript format (txt, srt, vtt)
- `INCLUDE_DIARIZATION`: Enable/disable speaker diarization
- `CACHE_ENABLED`: Enable/disable caching system
- `CACHE_EXPIRATION`: Cache expiration time in seconds
- `MAX_CACHE_SIZE`: Maximum cache size in bytes
- Various timeout settings

## Usage

### Basic Usage

Process a video file:
```bash
python -m scripts.transcribe_video path/to/your/video.mp4
```

Process an audio file (WAV files are processed directly):
```bash
python -m scripts.transcribe_video path/to/your/audio.wav
```

### Specify Output Location

```bash
python -m scripts.transcribe_video path/to/your/file.mp4 -o path/to/output.txt
```

### Resume Partial Processing

If you've already extracted the audio:
```bash
python -m scripts.transcribe_video path/to/your/audio.wav
```

## Development

### Using Make Commands

The project includes several make commands to simplify common operations:

```bash
make help     # Show all available commands
make setup    # Create virtual environment and install dependencies
make venv     # Create virtual environment only
make install  # Install dependencies into existing virtual environment
make test     # Run tests
make clean    # Remove Python cache files and temporary files
```

### Manual Development Setup

For development work without using Make:
```bash
pip install -r requirements-dev.txt
python -m pytest tests/
```

## Performance Considerations

- **Caching**: Enable caching for improved performance when processing the same files multiple times
- **Model Selection**: Choose the appropriate model size based on your accuracy needs and hardware capabilities
- **Hardware Acceleration**: Use CUDA (NVIDIA) or MPS (Apple Silicon) for faster processing
- **Memory Usage**: Large files may require significant memory, especially with larger models

## Known Issues

- Large video files may require significant memory
- Some hardware acceleration features require specific hardware/drivers
- Non-WAV audio files will be converted to WAV before processing
- You may see warnings about pyannote.audio and PyTorch version mismatches. Options to address this:
  - Use a newer diarization model: Set `DIARIZATION_MODEL=pyannote/speaker-diarization@2.1.1` in your `.env` file
  - Downgrade libraries: `pip install pyannote.audio==0.0.1 torch==1.10.0`
  - Ignore the warnings if diarization is working correctly

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

MIT License

Copyright (c) 2025 James Schindler

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Acknowledgments

- OpenAI's Whisper for transcription
- Pyannote.audio for speaker diarization
- MoviePy for video/audio processing
