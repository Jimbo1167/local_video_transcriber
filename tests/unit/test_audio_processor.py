import os
import sys
import pytest
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import the src package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.config import Config
from src.audio.processor import AudioProcessor, TimeoutException

@pytest.fixture
def config():
    """Create a test configuration."""
    config = Config()
    config.audio_timeout = 10
    return config

@pytest.fixture
def audio_processor(config):
    """Create a test audio processor."""
    return AudioProcessor(config)

def test_is_audio_file(audio_processor):
    """Test the is_audio_file method."""
    assert audio_processor.is_audio_file("test.wav") is True
    assert audio_processor.is_audio_file("test.mp3") is True
    assert audio_processor.is_audio_file("test.m4a") is True
    assert audio_processor.is_audio_file("test.aac") is True
    assert audio_processor.is_audio_file("test.flac") is True
    assert audio_processor.is_audio_file("test.ogg") is True
    assert audio_processor.is_audio_file("test.mp4") is False
    assert audio_processor.is_audio_file("test.mov") is False
    assert audio_processor.is_audio_file("test.txt") is False

@patch("src.audio.processor.AudioFileClip")
def test_get_audio_path_wav(mock_audio_file_clip, audio_processor):
    """Test get_audio_path with WAV file."""
    # Test with WAV file
    audio_path, needs_cleanup = audio_processor.get_audio_path("test.wav")
    assert audio_path == "test.wav"
    assert needs_cleanup is False
    mock_audio_file_clip.assert_not_called()

@patch("src.audio.processor.AudioFileClip")
def test_get_audio_path_mp3(mock_audio_file_clip, audio_processor):
    """Test get_audio_path with MP3 file."""
    # Mock the AudioFileClip
    mock_audio = MagicMock()
    mock_audio_file_clip.return_value = mock_audio
    
    # Create a temporary file to simulate the output
    with tempfile.NamedTemporaryFile(suffix=".wav") as temp_file:
        # Mock the write_audiofile method
        def side_effect(path, logger=None):
            with open(path, 'w') as f:
                f.write("test")
        
        mock_audio.write_audiofile.side_effect = side_effect
        
        # Patch os.path.exists to return True for our test file
        with patch("os.path.exists", return_value=True):
            # Patch os.remove to avoid actually removing files
            with patch("os.remove"):
                # Test with MP3 file
                with patch("src.audio.processor.timeout") as mock_timeout:
                    mock_timeout.return_value.__enter__.return_value = None
                    mock_timeout.return_value.__exit__.return_value = None
                    
                    # Mock the cache manager's get_cached_audio method to return None
                    if audio_processor.cache_manager:
                        with patch.object(audio_processor.cache_manager, 'get_cached_audio', return_value=None):
                            with patch.object(audio_processor.cache_manager, 'cache_audio', return_value=None):
                                audio_path, needs_cleanup = audio_processor.get_audio_path("test.mp3")
                    else:
                        audio_path, needs_cleanup = audio_processor.get_audio_path("test.mp3")
                    
                    assert audio_path == "test.wav"
                    assert needs_cleanup is True
                    mock_audio_file_clip.assert_called_once_with("test.mp3")
                    mock_audio.write_audiofile.assert_called_once()
                    mock_audio.close.assert_called_once()

@patch("src.audio.processor.VideoFileClip")
def test_extract_audio(mock_video_file_clip, audio_processor):
    """Test extract_audio method."""
    # Mock the VideoFileClip
    mock_video = MagicMock()
    mock_video_file_clip.return_value = mock_video
    mock_video.duration = 10.0
    
    # Mock the audio property
    mock_audio = MagicMock()
    mock_video.audio = mock_audio
    
    # Create a temporary file to simulate the output
    with tempfile.NamedTemporaryFile(suffix=".wav") as temp_file:
        # Mock the write_audiofile method
        def side_effect(path, logger=None):
            with open(path, 'w') as f:
                f.write("test")
        
        mock_audio.write_audiofile.side_effect = side_effect
        
        # Patch os.path.exists to return True for our test file
        with patch("os.path.exists", return_value=True):
            # Patch os.remove to avoid actually removing files
            with patch("os.remove"):
                # Test extract_audio
                with patch("src.audio.processor.timeout") as mock_timeout:
                    mock_timeout.return_value.__enter__.return_value = None
                    mock_timeout.return_value.__exit__.return_value = None
                    
                    audio_path = audio_processor.extract_audio("test.mp4")
                    
                    assert audio_path == "test.wav"
                    mock_video_file_clip.assert_called_once_with("test.mp4")
                    mock_audio.write_audiofile.assert_called_once()
                    mock_video.close.assert_called_once()

@patch("src.audio.processor.VideoFileClip")
def test_extract_audio_timeout(mock_video_file_clip, audio_processor):
    """Test extract_audio method with timeout."""
    # Mock the VideoFileClip
    mock_video = MagicMock()
    mock_video_file_clip.return_value = mock_video
    
    # Patch timeout to raise TimeoutException
    with patch("src.audio.processor.timeout") as mock_timeout:
        mock_timeout.return_value.__enter__.side_effect = TimeoutException("Test timeout")
        
        # Patch os.path.exists to return True for our test file
        with patch("os.path.exists", return_value=True):
            # Patch os.remove to avoid actually removing files
            with patch("os.remove"):
                # Test extract_audio with timeout
                with pytest.raises(TimeoutException):
                    audio_processor.extract_audio("test.mp4")
                    
                    mock_video_file_clip.assert_called_once_with("test.mp4")

@patch("librosa.load")
def test_load_audio(mock_librosa_load, audio_processor):
    """Test load_audio method."""
    # Skip this test if librosa is not installed
    pytest.importorskip("librosa")
    
    # Mock librosa.load
    mock_librosa_load.return_value = (np.zeros(16000), 16000)
    
    # Test load_audio
    audio = audio_processor.load_audio("test.wav")
    
    assert isinstance(audio, np.ndarray)
    mock_librosa_load.assert_called_once_with("test.wav", sr=16000, mono=True)

def test_process_audio_stream(audio_processor):
    """Test process_audio_stream method."""
    # Create a test audio array
    audio_data = np.zeros(100000)
    
    # Process in chunks with a specific chunk size in bytes
    # The actual number of samples per chunk will depend on the itemsize
    bytes_per_sample = audio_data.itemsize
    chunk_samples = AudioProcessor.CHUNK_SIZE // bytes_per_sample
    
    # Process in chunks
    chunks = list(audio_processor.process_audio_stream(audio_data))
    
    # Calculate expected number of chunks
    expected_chunks = (len(audio_data) + chunk_samples - 1) // chunk_samples
    
    # Check that we got the expected number of chunks
    assert len(chunks) == expected_chunks 