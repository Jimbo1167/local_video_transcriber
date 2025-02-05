"""
Tests for the video transcriber functionality.
"""

import os
import pytest
from src.transcriber import Transcriber
import json
from pathlib import Path
import tempfile
import unittest
import torch

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
    # Mock device-related functions
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    # Mock the model loading to avoid actual device operations
    class MockWhisperModel:
        def __init__(self, *args, **kwargs):
            pass

    class MockPipeline:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return MockDiarizer()

    class MockDiarizer:
        def __init__(self):
            self.device = None

        def to(self, device):
            self.device = device
            return self

        def __call__(self, *args, **kwargs):
            return self

    # Mock the model loading functions
    monkeypatch.setattr("src.transcriber.WhisperModel", MockWhisperModel)
    monkeypatch.setattr("pyannote.audio.Pipeline", MockPipeline)

    # Test MPS selection
    transcriber = Transcriber()
    assert transcriber._get_device() == "mps"

    # Test CUDA selection
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    transcriber = Transcriber()
    assert transcriber._get_device() == "cuda"

    # Test CPU fallback
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    transcriber = Transcriber()
    assert transcriber._get_device() == "cpu"

def test_env_variable_parsing(monkeypatch):
    """Test that environment variables are properly parsed, handling comments and whitespace."""
    test_cases = [
        ("txt", "txt"),  # Simple case
        ("txt  ", "txt"),  # Trailing whitespace
        ("  txt", "txt"),  # Leading whitespace
        ("txt # some comment", "txt"),  # Comment at end
        ("txt#comment", "txt"),  # Comment without space
        ("  txt  # comment with spaces  ", "txt"),  # Complex case with whitespace and comment
        (" srt #Options: txt, srt, vtt", "srt"),  # Real world case from bug
    ]
    
    for env_value, expected in test_cases:
        monkeypatch.setenv("OUTPUT_FORMAT", env_value)
        transcriber = Transcriber()
        assert transcriber.output_format == expected, \
            f"Failed to parse '{env_value}' correctly. Expected '{expected}' but got '{transcriber.output_format}'"

def test_audio_file_detection(test_transcriber):
    """Test detection of different audio file types."""
    assert test_transcriber._is_audio_file("test.wav") is True
    assert test_transcriber._is_audio_file("test.mp3") is True
    assert test_transcriber._is_audio_file("test.m4a") is True
    assert test_transcriber._is_audio_file("test.aac") is True
    assert test_transcriber._is_audio_file("test.WAV") is True  # Test case insensitivity
    assert test_transcriber._is_audio_file("test.mp4") is False
    assert test_transcriber._is_audio_file("test.mov") is False
    assert test_transcriber._is_audio_file("test.txt") is False

def test_get_audio_path(test_transcriber, tmp_path, monkeypatch):
    """Test audio path resolution for different input types."""
    # Mock the entire moviepy module
    class MockMoviePy:
        class VideoFileClip:
            def __init__(self, path):
                self.path = path
                self.audio = self
                self.duration = 10.0

            def write_audiofile(self, path, logger=None):
                with open(path, 'wb') as f:
                    f.write(b'RIFF' + b'\x00' * 36 + b'data' + b'\x00' * 4)

            def close(self):
                pass

    # Create a mock module
    mock_moviepy = type('moviepy', (), {'VideoFileClip': MockMoviePy.VideoFileClip})
    monkeypatch.setattr("src.transcriber.VideoFileClip", mock_moviepy.VideoFileClip)

    # Test WAV file (should be used directly)
    wav_file = tmp_path / "test.wav"
    with open(wav_file, 'wb') as f:
        f.write(b'RIFF' + b'\x00' * 36 + b'data' + b'\x00' * 4)

    audio_path, needs_cleanup = test_transcriber._get_audio_path(str(wav_file))
    assert audio_path == str(wav_file)
    assert needs_cleanup is False

    # Test MP3 file (should be converted)
    mp3_file = tmp_path / "test.mp3"
    with open(mp3_file, 'wb') as f:
        f.write(b'ID3' + b'\x00' * 100)  # Mock MP3 content

    audio_path, needs_cleanup = test_transcriber._get_audio_path(str(mp3_file))
    assert audio_path == str(mp3_file).rsplit(".", 1)[0] + ".wav"
    assert needs_cleanup is True
    assert os.path.exists(audio_path)

    # Cleanup
    if os.path.exists(audio_path):
        os.remove(audio_path)

def test_transcribe_with_different_inputs(test_transcriber, tmp_path, monkeypatch):
    """Test transcription with different input types."""
    # Mock dependencies to avoid actual processing
    class MockWhisperResult:
        def __init__(self, start, end, text):
            self.start = start
            self.end = end
            self.text = text

    def mock_transcribe(*args, **kwargs):
        segments = [
            MockWhisperResult(0, 1, "Test segment one"),
            MockWhisperResult(1, 2, "Test segment two")
        ]
        return segments, None

    class MockVideoFileClip:
        def __init__(self, path):
            self.audio = self
            self.duration = 10.0

        def write_audiofile(self, path, logger=None):
            with open(path, 'wb') as f:
                f.write(b'RIFF' + b'\x00' * 36 + b'data' + b'\x00' * 4)

        def close(self):
            pass

    class MockDiarization:
        def itertracks(self, yield_label=True):
            from pyannote.core import Segment
            yield Segment(0, 1), None, "SPEAKER_1"
            yield Segment(1, 2), None, "SPEAKER_2"

    class MockDiarizer:
        def __call__(self, audio_path):
            return MockDiarization()

        def to(self, device):
            return self

    monkeypatch.setattr("moviepy.VideoFileClip", MockVideoFileClip)
    monkeypatch.setattr(test_transcriber.whisper, "transcribe", mock_transcribe)
    test_transcriber.diarizer = MockDiarizer()

    # Test with WAV file
    wav_file = tmp_path / "test.wav"
    with open(wav_file, 'wb') as f:
        f.write(b'RIFF' + b'\x00' * 36 + b'data' + b'\x00' * 4)

    segments = test_transcriber.transcribe(str(wav_file))
    assert len(segments) == 2
    assert segments[0] == (0, 1, "Test segment one", "SPEAKER_1")
    assert segments[1] == (1, 2, "Test segment two", "SPEAKER_2")

    # Test without diarization
    test_transcriber.include_diarization = False
    segments = test_transcriber.transcribe(str(wav_file))
    assert len(segments) == 2
    assert segments[0] == (0, 1, "Test segment one", "")
    assert segments[1] == (1, 2, "Test segment two", "")

def test_error_handling_for_audio_files(test_transcriber, tmp_path):
    """Test error handling for audio file processing."""
    # Test with non-existent file
    with pytest.raises(Exception):
        test_transcriber.transcribe("nonexistent.wav")
    
    # Test with invalid audio file
    invalid_wav = tmp_path / "invalid.wav"
    invalid_wav.write_text("not a real wav file")
    with pytest.raises(Exception):
        test_transcriber.transcribe(str(invalid_wav))
    
    # Test with unsupported audio format
    unsupported = tmp_path / "test.ogg"
    unsupported.touch()
    with pytest.raises(Exception):
        test_transcriber.transcribe(str(unsupported))

class TestTranscriber(unittest.TestCase):
    def setUp(self):
        self.transcriber = Transcriber()
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Sample segments for testing
        self.test_segments = [
            (0.0, 2.5, "Hello world", "SPEAKER_00"),
            (2.5, 5.0, "This is a test", "SPEAKER_01"),
            (5.0, 7.5, "Of the transcription", "SPEAKER_00")
        ]
    
    def tearDown(self):
        # Clean up temporary files
        if os.path.exists(self.test_dir):
            for file in os.listdir(self.test_dir):
                os.remove(os.path.join(self.test_dir, file))
            os.rmdir(self.test_dir)
    
    def test_save_transcript_txt_format(self):
        """Test saving transcript in txt format with diarization"""
        output_path = os.path.join(self.test_dir, "test_output.txt")
        self.transcriber.output_format = "txt"
        self.transcriber.include_diarization = True
        
        # Save the transcript
        self.transcriber.save_transcript(self.test_segments, output_path)
        
        # Verify the file exists and has content
        self.assertTrue(os.path.exists(output_path))
        self.assertTrue(os.path.getsize(output_path) > 0)
        
        # Read and verify content
        with open(output_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        self.assertEqual(len(lines), 3)  # Should have 3 lines for 3 segments
        self.assertEqual(lines[0], "[SPEAKER_00] Hello world\n")
        self.assertEqual(lines[1], "[SPEAKER_01] This is a test\n")
        self.assertEqual(lines[2], "[SPEAKER_00] Of the transcription\n")
    
    def test_save_transcript_txt_format_no_diarization(self):
        """Test saving transcript in txt format without diarization"""
        output_path = os.path.join(self.test_dir, "test_output.txt")
        self.transcriber.output_format = "txt"
        self.transcriber.include_diarization = False
        
        self.transcriber.save_transcript(self.test_segments, output_path)
        
        with open(output_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        self.assertEqual(len(lines), 3)
        self.assertEqual(lines[0], "Hello world\n")
        self.assertEqual(lines[1], "This is a test\n")
        self.assertEqual(lines[2], "Of the transcription\n")
    
    def test_save_transcript_srt_format(self):
        """Test saving transcript in SRT format"""
        output_path = os.path.join(self.test_dir, "test_output.srt")
        self.transcriber.output_format = "srt"
        self.transcriber.include_diarization = True
        
        self.transcriber.save_transcript(self.test_segments, output_path)
        
        with open(output_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # SRT format should have 4 lines per segment (number, timestamp, text, blank line)
        self.assertEqual(len(lines), 12)
        
        # Check first segment format
        self.assertEqual(lines[0], "1\n")
        self.assertEqual(lines[1], "00:00:00,000 --> 00:00:02,500\n")
        self.assertEqual(lines[2], "[SPEAKER_00] Hello world\n")
        self.assertEqual(lines[3], "\n")
    
    def test_save_transcript_vtt_format(self):
        """Test saving transcript in VTT format"""
        output_path = os.path.join(self.test_dir, "test_output.vtt")
        self.transcriber.output_format = "vtt"
        self.transcriber.include_diarization = True
        
        self.transcriber.save_transcript(self.test_segments, output_path)
        
        with open(output_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # VTT format should have WEBVTT header + blank line + 4 lines per segment
        self.assertEqual(len(lines), 14)
        self.assertEqual(lines[0], "WEBVTT\n")
        self.assertEqual(lines[1], "\n")
        
        # Check first segment format
        self.assertEqual(lines[2], "1\n")
        self.assertEqual(lines[3], "00:00:00.000 --> 00:00:02.500\n")
        self.assertEqual(lines[4], "[SPEAKER_00] Hello world\n")
        self.assertEqual(lines[5], "\n")
    
    def test_save_transcript_empty_segments(self):
        """Test saving transcript with empty segments list"""
        output_path = os.path.join(self.test_dir, "test_output.txt")
        self.transcriber.output_format = "txt"
        
        self.transcriber.save_transcript([], output_path)
        
        self.assertTrue(os.path.exists(output_path))
        self.assertEqual(os.path.getsize(output_path), 0)
    
    def test_save_transcript_invalid_format(self):
        """Test saving transcript with invalid format"""
        output_path = os.path.join(self.test_dir, "test_output.txt")
        self.transcriber.output_format = "invalid"
        
        with self.assertRaises(ValueError):
            self.transcriber.save_transcript(self.test_segments, output_path)

if __name__ == '__main__':
    unittest.main() 