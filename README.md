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
- Various timeout settings

## Usage

### Basic Usage

Process a video file:
```bash
python transcribe_video.py path/to/your/video.mp4
```

Process an audio file (WAV files are processed directly):
```bash
python transcribe_video.py path/to/your/audio.wav
```

### Specify Output Location

```bash
python transcribe_video.py path/to/your/file.mp4 -o path/to/output.txt
```

### Resume Partial Processing

If you've already extracted the audio:
```bash
python transcribe_video.py path/to/your/audio.wav
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

## Known Issues

- Large video files may require significant memory
- Some hardware acceleration features require specific hardware/drivers
- Non-WAV audio files will be converted to WAV before processing

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
