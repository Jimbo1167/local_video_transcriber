"""
Tests for the video transcriber functionality.
"""

import os
import pytest
from src.transcriber import Transcriber
import json
from pathlib import Path

def test_transcriber_initialization(test_transcriber):
    """Test that the transcriber initializes correctly."""
    assert test_transcriber.whisper is not None
    assert test_transcriber.whisper_model_size == "tiny"
    assert test_transcriber.language == "en"
    assert test_transcriber.include_diarization is True

def test_output_formats(test_transcriber, mock_video, output_dir):
    """Test different output formats (txt, srt, vtt)."""
    # Mock segments for testing
    segments = [
        (0.0, 2.0, "Hello world", "SPEAKER_01"),
        (2.0, 4.0, "This is a test", "SPEAKER_02")
    ]
    
    formats = ["txt", "srt", "vtt"]
    for fmt in formats:
        # Set output format
        test_transcriber.output_format = fmt
        output_path = output_dir / f"test.{fmt}"
        
        # Save transcript
        test_transcriber.save_transcript(segments, str(output_path))
        
        # Verify file exists and is not empty
        assert output_path.exists()
        assert output_path.stat().st_size > 0
        
        # Read and verify content based on format
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Basic content checks
            assert "Hello world" in content
            assert "This is a test" in content
            
            # Format-specific checks
            if fmt == "txt":
                assert "[SPEAKER_01]" in content
                assert "[SPEAKER_02]" in content
            elif fmt == "srt":
                assert "00:00:00" in content
                assert "-->" in content
            elif fmt == "vtt":
                assert "WEBVTT" in content
                assert "-->" in content

def test_diarization_toggle(mock_video, output_dir):
    """Test transcriber with and without diarization."""
    # Test with diarization enabled
    os.environ["INCLUDE_DIARIZATION"] = "true"
    transcriber_with = Transcriber()
    assert transcriber_with.include_diarization is True
    
    # Test with diarization disabled
    os.environ["INCLUDE_DIARIZATION"] = "false"
    transcriber_without = Transcriber()
    assert transcriber_without.include_diarization is False

def test_timestamp_formatting(test_transcriber):
    """Test timestamp formatting for different output formats."""
    # Test regular SRT format
    srt_timestamp = test_transcriber._format_timestamp(3661.5)  # 1h 1m 1.5s
    assert srt_timestamp == "01:01:01,500"
    
    # Test VTT format
    vtt_timestamp = test_transcriber._format_timestamp(3661.5, vtt=True)
    assert vtt_timestamp == "01:01:01.500"

def test_error_handling(test_transcriber, tmp_path):
    """Test error handling for invalid inputs."""
    # Test invalid video path
    with pytest.raises(Exception):
        test_transcriber.transcribe("nonexistent_video.mp4")
    
    # Test invalid output directory
    segments = [(0, 1, "test", "SPEAKER")]
    invalid_path = "/nonexistent/directory/transcript.txt"
    with pytest.raises(Exception):
        test_transcriber.save_transcript(segments, invalid_path)
    
    # Test invalid output format
    test_transcriber.output_format = "invalid"
    valid_path = tmp_path / "test.txt"
    with pytest.raises(ValueError):
        test_transcriber.save_transcript(segments, str(valid_path))

def test_device_selection(monkeypatch):
    """Test device selection logic."""
    # Test MPS (Apple Silicon) selection
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    transcriber = Transcriber()
    assert transcriber.device == "mps"
    
    # Test CUDA selection
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    transcriber = Transcriber()
    assert transcriber.device == "cuda"
    
    # Test CPU fallback
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    transcriber = Transcriber()
    assert transcriber.device == "cpu" 