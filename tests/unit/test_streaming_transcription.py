import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from typing import List, Dict, Any, Iterator

from src.config import Config
from src.transcription.streaming import StreamingTranscriber, AsyncStreamingTranscriber
from src.transcription.engine import TranscriptionEngine

@pytest.fixture
def mock_whisper_model():
    """Create a mock WhisperModel for testing."""
    mock = MagicMock()
    
    # Mock the transcribe method
    def mock_transcribe(audio_data, **kwargs):
        # Create mock segments based on the length of audio data
        class MockSegment:
            def __init__(self, start, end, text):
                self.start = start
                self.end = end
                self.text = text
                self.words = []
        
        # Generate segments based on audio length
        # For testing, create a segment for each 1 second of audio
        segments = []
        
        # Handle both numpy arrays and lists
        if isinstance(audio_data, np.ndarray):
            audio_length = len(audio_data) / 16000  # Assuming 16kHz audio
        else:
            audio_length = 1  # Default for testing
        
        for i in range(int(audio_length) or 1):  # Ensure at least one segment
            segments.append(MockSegment(
                start=float(i),
                end=float(i + 1),
                text=f"Test segment {i+1}"
            ))
        
        return segments, {"language": "en"}
    
    mock.transcribe.side_effect = mock_transcribe
    return mock

@pytest.fixture
def config():
    """Create a test configuration."""
    config = Config()
    config.language = "en"
    config.whisper_model_size = "tiny"
    return config

def create_audio_chunks():
    """Create audio chunks for testing."""
    # Create 3 chunks of 1 second each at 16kHz
    sample_rate = 16000
    return [
        np.zeros(sample_rate, dtype=np.float32),
        np.zeros(sample_rate, dtype=np.float32),
        np.zeros(sample_rate, dtype=np.float32)
    ]

def test_streaming_transcriber_init(mock_whisper_model, config):
    """Test initialization of StreamingTranscriber."""
    transcriber = StreamingTranscriber(mock_whisper_model, config)
    
    assert transcriber.whisper == mock_whisper_model
    assert transcriber.config == config
    assert transcriber.language == config.language
    assert transcriber.sample_rate == 16000
    assert transcriber.buffer_size == 30 * 16000  # 30 seconds buffer
    assert transcriber.min_chunk_size == 5 * 16000  # 5 seconds minimum

def test_streaming_transcriber_process_stream(mock_whisper_model, config):
    """Test processing an audio stream."""
    transcriber = StreamingTranscriber(mock_whisper_model, config)
    
    # Create audio chunks
    audio_chunks = create_audio_chunks()
    
    # Process the stream
    segments = list(transcriber.process_stream(iter(audio_chunks)))
    
    # We should get at least one segment
    assert len(segments) > 0
    
    # Check the structure of the segments
    for segment in segments:
        assert "start" in segment
        assert "end" in segment
        assert "text" in segment
        assert "words" in segment
        assert isinstance(segment["start"], float)
        assert isinstance(segment["end"], float)
        assert isinstance(segment["text"], str)
        assert isinstance(segment["words"], list)

def test_async_streaming_transcriber(mock_whisper_model, config):
    """Test the AsyncStreamingTranscriber."""
    async_transcriber = AsyncStreamingTranscriber(mock_whisper_model, config)
    
    # Create audio chunks
    audio_chunks = create_audio_chunks()
    
    # Start processing
    async_transcriber.start_processing(iter(audio_chunks))
    
    # Get results
    segments = list(async_transcriber.get_results())
    
    # We should get at least one segment
    assert len(segments) > 0
    
    # Check the structure of the segments
    for segment in segments:
        assert "start" in segment
        assert "end" in segment
        assert "text" in segment
        assert "words" in segment
        assert isinstance(segment["start"], float)
        assert isinstance(segment["end"], float)
        assert isinstance(segment["text"], str)
        assert isinstance(segment["words"], list)
    
    # Stop the transcriber
    async_transcriber.stop()
    assert not async_transcriber.is_running

@patch('src.transcription.engine.WhisperModel')
def test_transcription_engine_stream(mock_whisper_class, config):
    """Test the transcribe_stream method of TranscriptionEngine."""
    # Create a mock WhisperModel instance
    mock_whisper = MagicMock()
    mock_whisper_class.return_value = mock_whisper
    
    # Mock the transcribe method
    def mock_transcribe(audio_data, **kwargs):
        class MockSegment:
            def __init__(self, start, end, text):
                self.start = start
                self.end = end
                self.text = text
                self.words = []
        
        segments = [MockSegment(0.0, 1.0, "Test segment")]
        return segments, {"language": "en"}
    
    mock_whisper.transcribe.side_effect = mock_transcribe
    
    # Create the transcription engine
    engine = TranscriptionEngine(config)
    engine.whisper = mock_whisper
    
    # Create audio chunks
    audio_chunks = create_audio_chunks()
    
    # Process the stream
    segments = list(engine.transcribe_stream(iter(audio_chunks)))
    
    # We should get at least one segment
    assert len(segments) > 0
    
    # Check the structure of the segments
    for segment in segments:
        assert "start" in segment
        assert "end" in segment
        assert "text" in segment
        assert "words" in segment

@patch('src.transcription.engine.WhisperModel')
def test_transcription_engine_async_stream(mock_whisper_class, config):
    """Test the start_async_transcription method of TranscriptionEngine."""
    # Create a mock WhisperModel instance
    mock_whisper = MagicMock()
    mock_whisper_class.return_value = mock_whisper
    
    # Mock the transcribe method
    def mock_transcribe(audio_data, **kwargs):
        class MockSegment:
            def __init__(self, start, end, text):
                self.start = start
                self.end = end
                self.text = text
                self.words = []
        
        segments = [MockSegment(0.0, 1.0, "Test segment")]
        return segments, {"language": "en"}
    
    mock_whisper.transcribe.side_effect = mock_transcribe
    
    # Create the transcription engine
    engine = TranscriptionEngine(config)
    engine.whisper = mock_whisper
    
    # Create audio chunks
    audio_chunks = create_audio_chunks()
    
    # Start async processing
    async_transcriber = engine.start_async_transcription(iter(audio_chunks))
    
    # Get results
    segments = list(async_transcriber.get_results())
    
    # We should get at least one segment
    assert len(segments) > 0
    
    # Check the structure of the segments
    for segment in segments:
        assert "start" in segment
        assert "end" in segment
        assert "text" in segment
        assert "words" in segment
    
    # Stop the transcriber
    async_transcriber.stop() 