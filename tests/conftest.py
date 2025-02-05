"""
Pytest configuration and fixtures for testing the video transcriber.
"""

import os
import pytest
from pathlib import Path
import numpy as np
from src.transcriber import Transcriber

@pytest.fixture
def sample_audio():
    """Generate a simple sine wave audio sample for testing."""
    sample_rate = 16000
    duration = 2.0  # 2 seconds
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    return audio

@pytest.fixture
def mock_video(tmp_path):
    """Create a mock video file for testing."""
    video_path = tmp_path / "test_video.mp4"
    # Create an empty file
    video_path.touch()
    return str(video_path)

@pytest.fixture
def test_transcriber():
    """Create a transcriber instance with test configuration."""
    # Set test environment variables
    os.environ["WHISPER_MODEL"] = "tiny"  # Use smallest model for faster tests
    os.environ["INCLUDE_DIARIZATION"] = "true"
    os.environ["OUTPUT_FORMAT"] = "txt"
    return Transcriber()

@pytest.fixture
def output_dir(tmp_path):
    """Create and return a temporary directory for test outputs."""
    output_path = tmp_path / "transcripts"
    output_path.mkdir(exist_ok=True)
    return output_path 