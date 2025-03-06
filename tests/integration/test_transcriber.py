import os
import sys
import pytest
import tempfile
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import the src package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.config import Config
from src.transcriber import Transcriber
from src.audio.processor import AudioProcessor
from src.transcription.engine import TranscriptionEngine
from src.diarization.engine import DiarizationEngine
from src.output.formatter import OutputFormatter

@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    config = Config()
    config.whisper_model_size = "base"
    config.include_diarization = True
    config.output_format = "txt"
    return config

@pytest.fixture
def mock_audio_processor():
    """Create a mock audio processor."""
    mock = MagicMock(spec=AudioProcessor)
    mock.get_audio_path.return_value = ("test.wav", False)
    return mock

@pytest.fixture
def mock_transcription_engine():
    """Create a mock transcription engine."""
    mock = MagicMock(spec=TranscriptionEngine)
    mock.transcribe.return_value = [
        {"start": 0.0, "end": 2.0, "text": "Test segment one"},
        {"start": 2.0, "end": 4.0, "text": "Test segment two"}
    ]
    return mock

@pytest.fixture
def mock_diarization_engine():
    """Create a mock diarization engine."""
    mock = MagicMock(spec=DiarizationEngine)
    mock.diarize.return_value = [
        {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_01"},
        {"start": 2.0, "end": 4.0, "speaker": "SPEAKER_02"}
    ]
    return mock

@pytest.fixture
def mock_output_formatter():
    """Create a mock output formatter."""
    mock = MagicMock(spec=OutputFormatter)
    return mock

@pytest.fixture
def transcriber(mock_config, mock_audio_processor, mock_transcription_engine, 
               mock_diarization_engine, mock_output_formatter):
    """Create a transcriber with mock components."""
    transcriber = Transcriber(mock_config)
    transcriber.audio_processor = mock_audio_processor
    transcriber.transcription_engine = mock_transcription_engine
    transcriber.diarization_engine = mock_diarization_engine
    transcriber.output_formatter = mock_output_formatter
    return transcriber

def test_transcribe_with_diarization(transcriber, mock_audio_processor, 
                                   mock_transcription_engine, mock_diarization_engine):
    """Test transcribe method with diarization enabled."""
    # Setup mock return values
    mock_audio_processor.get_audio_path.return_value = ("test.wav", False)
    
    # Run the transcribe method
    result = transcriber.transcribe("test.mp4")
    
    # Check that the audio processor was called
    mock_audio_processor.get_audio_path.assert_called_once_with("test.mp4")
    
    # Check that the transcription engine was called
    mock_transcription_engine.transcribe.assert_called_once_with("test.wav")
    
    # Check that the diarization engine was called
    mock_diarization_engine.diarize.assert_called_once_with("test.wav")
    
    # Check the result
    assert len(result) == 2
    assert result[0] == (0.0, 2.0, "Test segment one", "SPEAKER_01")
    assert result[1] == (2.0, 4.0, "Test segment two", "SPEAKER_02")

def test_transcribe_without_diarization(transcriber, mock_audio_processor, 
                                      mock_transcription_engine, mock_diarization_engine):
    """Test transcribe method with diarization disabled."""
    # Setup mock return values
    mock_audio_processor.get_audio_path.return_value = ("test.wav", False)
    
    # Disable diarization
    transcriber.include_diarization = False
    transcriber.config.include_diarization = False
    
    # Run the transcribe method
    result = transcriber.transcribe("test.mp4")
    
    # Check that the audio processor was called
    mock_audio_processor.get_audio_path.assert_called_once_with("test.mp4")
    
    # Check that the transcription engine was called
    mock_transcription_engine.transcribe.assert_called_once_with("test.wav")
    
    # Check that the diarization engine was not called
    mock_diarization_engine.diarize.assert_not_called()
    
    # Check the result
    assert len(result) == 2
    assert result[0][3] == ""  # No speaker information
    assert result[1][3] == ""  # No speaker information

def test_save_transcript(transcriber, mock_output_formatter):
    """Test save_transcript method."""
    # Create test segments
    segments = [
        (0.0, 2.0, "Test segment one", "SPEAKER_01"),
        (2.0, 4.0, "Test segment two", "SPEAKER_02")
    ]
    
    # Run the save_transcript method
    transcriber.save_transcript(segments, "test.txt")
    
    # Check that the output formatter was called
    mock_output_formatter.save_transcript.assert_called_once_with(segments, "test.txt")

def test_combine_segments_with_speakers(transcriber):
    """Test _combine_segments_with_speakers method."""
    # Create test segments
    transcription_segments = [
        {"start": 0.0, "end": 2.0, "text": "Test segment one"},
        {"start": 2.0, "end": 4.0, "text": "Test segment two"}
    ]
    
    diarization_segments = [
        {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_01"},
        {"start": 2.0, "end": 4.0, "speaker": "SPEAKER_02"}
    ]
    
    # Run the _combine_segments_with_speakers method
    result = transcriber._combine_segments_with_speakers(transcription_segments, diarization_segments)
    
    # Check the result
    assert len(result) == 2
    assert result[0] == (0.0, 2.0, "Test segment one", "SPEAKER_01")
    assert result[1] == (2.0, 4.0, "Test segment two", "SPEAKER_02")
    
    # Test with no diarization segments
    result = transcriber._combine_segments_with_speakers(transcription_segments, None)
    
    # Check the result
    assert len(result) == 2
    assert result[0] == (0.0, 2.0, "Test segment one", "")
    assert result[1] == (2.0, 4.0, "Test segment two", "") 