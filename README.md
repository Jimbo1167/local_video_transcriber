# Video Transcriber

A Python tool for local video transcription with speaker diarization, optimized for Apple Silicon.

## Features
- Extract audio from video files
- Transcribe speech to text using Whisper
- Identify different speakers using pyannote.audio
- Fully local processing - no cloud services required
- Optimized for Apple Silicon with MPS acceleration
- Memory-efficient batch processing for long videos

## Performance
- **Apple Silicon (M1/M2/M3)**: ~35-50 minutes for a 1-hour video
  - Uses MPS acceleration for diarization
  - Optimized CPU processing for Whisper
  - Efficient memory management
- **Other Systems**: ~55-80 minutes for a 1-hour video

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/video-transcriber.git
cd video-transcriber
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Copy .env.example to .env and add your HuggingFace token:
```bash
cp .env.example .env
```

## Usage

Basic transcription:
```python
from src.transcriber import Transcriber

transcriber = Transcriber()
segments = transcriber.transcribe("path/to/video.mp4")

for start, end, text, speaker in segments:
    print(f"[{speaker}] [{start:.2f}s -> {end:.2f}s] {text}")
```

## Optimization Details

The transcriber automatically detects and utilizes the best available hardware:
- On Apple Silicon: Uses MPS (Metal Performance Shaders) for diarization and optimized CPU processing for Whisper
- Memory-efficient batch processing (50 segments at a time)
- Voice Activity Detection (VAD) for improved accuracy and speed
- Int8 quantization for better performance
- Automatic memory cleanup during batch processing

## Development

For development and testing:
```bash
pip install -r requirements-dev.txt
pytest tests/
```

For coverage reporting:
```bash
pytest tests/ --cov=src/
```

## Future Enhancements

We maintain a comprehensive roadmap of planned features and improvements. See [FUTURE.md](FUTURE.md) for details about upcoming enhancements, including:
- Batch processing support
- Real-time transcription
- GUI interface
- Cloud integrations
- Performance optimizations
- And much more!

See examples/basic_transcription.py for more detailed usage.