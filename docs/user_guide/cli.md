# Command Line Interface Guide

This guide provides detailed information about the Video Transcriber command-line interface (CLI) commands, options, and usage examples.

## Overview

The Video Transcriber provides several command-line scripts for different use cases:

1. `transcribe.py` - Unified CLI with subcommands for transcription
2. `model_server.py` - Server for persistent model instances
3. `model_client.py` - Client for interacting with the model server
4. `batch_transcribe.py` - Process multiple files in batch
5. `stream_transcribe.py` - Process files in streaming mode
6. `transcribe_video.py` - Legacy script for basic transcription

## Unified CLI: `transcribe.py`

The `transcribe.py` script provides a unified interface with subcommands for different transcription modes.

### Global Options

- `--help`: Show help message and exit
- `--version`: Show version information and exit

### Transcribe Command

The `transcribe` command processes a single video or audio file.

```bash
./scripts/transcribe.py transcribe [OPTIONS] INPUT_PATH
```

#### Options

- `--output, -o TEXT`: Output file path
- `--format, -f [txt|srt|vtt|json]`: Output format (default: txt)
- `--model, -m [tiny|base|small|medium|large-v3]`: Whisper model size (default: base)
- `--language, -l TEXT`: Language code (default: en)
- `--diarize / --no-diarize`: Enable/disable speaker diarization (default: enabled)
- `--help`: Show help message and exit

#### Examples

Basic transcription:
```bash
./scripts/transcribe.py transcribe path/to/video.mp4
```

Specify output format and location:
```bash
./scripts/transcribe.py transcribe path/to/video.mp4 -f srt -o path/to/output.srt
```

Disable speaker diarization:
```bash
./scripts/transcribe.py transcribe path/to/video.mp4 --no-diarize
```

Use a different model:
```bash
./scripts/transcribe.py transcribe path/to/video.mp4 -m medium
```

### Stream Command

The `stream` command processes a file in streaming mode to reduce memory usage.

```bash
./scripts/transcribe.py stream [OPTIONS] INPUT_PATH
```

#### Options

Same as the `transcribe` command.

#### Examples

Basic streaming transcription:
```bash
./scripts/transcribe.py stream path/to/video.mp4
```

Streaming with specific options:
```bash
./scripts/transcribe.py stream path/to/video.mp4 -f vtt -m small --no-diarize
```

### Batch Command

The `batch` command processes multiple files in batch.

```bash
./scripts/transcribe.py batch [OPTIONS] INPUT_PATHS...
```

#### Options

- `--output-dir, -o TEXT`: Output directory (default: transcripts)
- `--format, -f [txt|srt|vtt|json]`: Output format (default: txt)
- `--model, -m [tiny|base|small|medium|large-v3]`: Whisper model size (default: base)
- `--language, -l TEXT`: Language code (default: en)
- `--diarize / --no-diarize`: Enable/disable speaker diarization (default: enabled)
- `--workers, -w INTEGER`: Number of worker processes (default: auto)
- `--help`: Show help message and exit

#### Examples

Process multiple files:
```bash
./scripts/transcribe.py batch path/to/video1.mp4 path/to/video2.mp4
```

Process all MP4 files in a directory:
```bash
./scripts/transcribe.py batch path/to/directory/*.mp4
```

Specify output directory and format:
```bash
./scripts/transcribe.py batch path/to/directory/*.mp4 -o path/to/output -f srt
```

Limit the number of worker processes:
```bash
./scripts/transcribe.py batch path/to/directory/*.mp4 -w 2
```

## Model Server: `model_server.py`

The `model_server.py` script runs a persistent model server for faster processing.

```bash
./scripts/model_server.py [OPTIONS] COMMAND [ARGS]...
```

### Commands

- `start`: Start the model server
- `stop`: Stop the model server
- `restart`: Restart the model server
- `status`: Check the server status

### Start Command

```bash
./scripts/model_server.py start [OPTIONS]
```

#### Options

- `--host TEXT`: Host to bind the server (default: localhost)
- `--port INTEGER`: Port to bind the server (default: 5000)
- `--model [tiny|base|small|medium|large-v3]`: Whisper model size (default: base)
- `--diarize / --no-diarize`: Enable/disable speaker diarization (default: enabled)
- `--daemon / --no-daemon`: Run as a daemon process (default: no-daemon)
- `--help`: Show help message and exit

#### Examples

Start the server with default settings:
```bash
./scripts/model_server.py start
```

Start the server with a specific model and port:
```bash
./scripts/model_server.py start --model medium --port 5001
```

Start the server as a daemon process:
```bash
./scripts/model_server.py start --daemon
```

### Stop Command

```bash
./scripts/model_server.py stop
```

### Restart Command

```bash
./scripts/model_server.py restart [OPTIONS]
```

Options are the same as the `start` command.

### Status Command

```bash
./scripts/model_server.py status
```

## Model Client: `model_client.py`

The `model_client.py` script interacts with the model server.

```bash
./scripts/model_client.py [OPTIONS] COMMAND [ARGS]...
```

### Commands

- `status`: Check the server status
- `transcribe`: Transcribe a file using the server

### Status Command

```bash
./scripts/model_client.py status [OPTIONS]
```

#### Options

- `--server-url TEXT`: URL of the model server (default: http://localhost:5000)
- `--help`: Show help message and exit

#### Examples

Check the status of the default server:
```bash
./scripts/model_client.py status
```

Check the status of a specific server:
```bash
./scripts/model_client.py status --server-url http://example.com:5000
```

### Transcribe Command

```bash
./scripts/model_client.py transcribe [OPTIONS] INPUT_PATH
```

#### Options

- `--server-url TEXT`: URL of the model server (default: http://localhost:5000)
- `--output, -o TEXT`: Output file path
- `--format, -f [txt|srt|vtt|json]`: Output format (default: txt)
- `--language, -l TEXT`: Language code (default: en)
- `--diarize / --no-diarize`: Enable/disable speaker diarization (default: enabled)
- `--help`: Show help message and exit

#### Examples

Transcribe a file using the default server:
```bash
./scripts/model_client.py transcribe path/to/video.mp4
```

Specify output format and location:
```bash
./scripts/model_client.py transcribe path/to/video.mp4 -f srt -o path/to/output.srt
```

Use a specific server:
```bash
./scripts/model_client.py transcribe path/to/video.mp4 --server-url http://example.com:5000
```

## Batch Transcription: `batch_transcribe.py`

The `batch_transcribe.py` script processes multiple files in batch.

```bash
./scripts/batch_transcribe.py [OPTIONS] INPUT_PATHS...
```

### Options

- `--output-dir, -o TEXT`: Output directory (default: transcripts)
- `--format, -f [txt|srt|vtt|json]`: Output format (default: txt)
- `--model, -m [tiny|base|small|medium|large-v3]`: Whisper model size (default: base)
- `--language, -l TEXT`: Language code (default: en)
- `--diarize / --no-diarize`: Enable/disable speaker diarization (default: enabled)
- `--workers, -w INTEGER`: Number of worker processes (default: auto)
- `--help`: Show help message and exit

### Examples

Process multiple files:
```bash
./scripts/batch_transcribe.py path/to/video1.mp4 path/to/video2.mp4
```

Process all MP4 files in a directory:
```bash
./scripts/batch_transcribe.py path/to/directory/*.mp4
```

Specify output directory and format:
```bash
./scripts/batch_transcribe.py path/to/directory/*.mp4 -o path/to/output -f srt
```

## Streaming Transcription: `stream_transcribe.py`

The `stream_transcribe.py` script processes a file in streaming mode to reduce memory usage.

```bash
./scripts/stream_transcribe.py [OPTIONS] INPUT_PATH
```

### Options

- `--output, -o TEXT`: Output file path
- `--format, -f [txt|srt|vtt|json]`: Output format (default: txt)
- `--model, -m [tiny|base|small|medium|large-v3]`: Whisper model size (default: base)
- `--language, -l TEXT`: Language code (default: en)
- `--diarize / --no-diarize`: Enable/disable speaker diarization (default: enabled)
- `--help`: Show help message and exit

### Examples

Basic streaming transcription:
```bash
./scripts/stream_transcribe.py path/to/video.mp4
```

Specify output format and location:
```bash
./scripts/stream_transcribe.py path/to/video.mp4 -f srt -o path/to/output.srt
```

## Legacy Script: `transcribe_video.py`

The `transcribe_video.py` script is the original script for basic transcription.

```bash
python -m scripts.transcribe_video [OPTIONS] INPUT_PATH
```

### Options

- `--output, -o TEXT`: Output file path
- `--format, -f [txt|srt|vtt|json]`: Output format (default: txt)
- `--diarize / --no-diarize`: Enable/disable speaker diarization (default: enabled)
- `--help`: Show help message and exit

### Examples

Basic transcription:
```bash
python -m scripts.transcribe_video path/to/video.mp4
```

Specify output format and location:
```bash
python -m scripts.transcribe_video path/to/video.mp4 -f srt -o path/to/output.srt
``` 