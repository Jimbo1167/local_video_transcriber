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

## Supported Formats

### Input Formats
- Video: mov, mp4, etc. (any format supported by MoviePy)
- Audio: wav (direct processing), mp3, m4a, aac (auto-converted to wav)

### Output Formats
- `txt`: Simple text format with speaker labels
- `srt`: SubRip subtitle format with timestamps
- `vtt`: WebVTT format for web video subtitles

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

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Copy the example environment file and configure your settings:
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

For development work, install additional dependencies:
```bash
pip install -r requirements-dev.txt
```

Run tests:
```bash
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

[Your chosen license]

## Acknowledgments

- OpenAI's Whisper for transcription
- Pyannote.audio for speaker diarization
- MoviePy for video/audio processing