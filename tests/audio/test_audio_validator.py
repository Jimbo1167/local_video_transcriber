import pytest
from pathlib import Path
import wave
import numpy as np
from pydub import AudioSegment
from src.audio import AudioValidator, AudioFormatError

@pytest.fixture
def temp_wav_file(tmp_path):
    """Create a valid WAV file for testing."""
    file_path = tmp_path / "test.wav"
    sample_rate = 16000
    duration = 1.0  # seconds
    samples = np.zeros(int(duration * sample_rate))
    
    with wave.open(str(file_path), 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.astype(np.int16).tobytes())
    
    return file_path

@pytest.fixture
def temp_mp3_file(tmp_path, temp_wav_file):
    """Create a valid MP3 file for testing."""
    file_path = tmp_path / "test.mp3"
    audio = AudioSegment.from_wav(str(temp_wav_file))
    audio.export(str(file_path), format='mp3')
    return file_path

@pytest.fixture
def invalid_wav_file(tmp_path):
    """Create an invalid WAV file for testing."""
    file_path = tmp_path / "invalid.wav"
    file_path.write_text("Not a WAV file")
    return file_path

class TestAudioValidator:
    def test_supported_formats(self):
        """Test that supported formats are correctly defined."""
        assert '.wav' in AudioValidator.SUPPORTED_FORMATS
        assert '.mp3' in AudioValidator.SUPPORTED_FORMATS
        assert '.m4a' in AudioValidator.SUPPORTED_FORMATS
        assert '.aac' in AudioValidator.SUPPORTED_FORMATS
        assert '.ogg' not in AudioValidator.SUPPORTED_FORMATS
    
    @pytest.mark.parametrize("file_path,expected", [
        ("test.wav", True),
        ("test.mp3", True),
        ("test.m4a", True),
        ("test.aac", True),
        ("test.WAV", True),
        ("test.MP3", True),
        ("test.ogg", False),
        ("test.txt", False),
        ("test", False),
    ])
    def test_is_valid_format(self, file_path, expected):
        """Test format validation for various file extensions."""
        assert AudioValidator.is_valid_format(file_path) == expected
    
    def test_validate_wav_file_valid(self, temp_wav_file):
        """Test validation of a valid WAV file."""
        properties = AudioValidator.validate_wav_file(str(temp_wav_file))
        assert properties is not None
        assert properties['channels'] == 1
        assert properties['sample_width'] == 2
        assert properties['frame_rate'] == 16000
        assert properties['duration'] == pytest.approx(1.0)
    
    def test_validate_wav_file_invalid(self, invalid_wav_file):
        """Test validation of an invalid WAV file."""
        with pytest.raises(AudioFormatError) as exc_info:
            AudioValidator.validate_wav_file(str(invalid_wav_file))
        assert "Invalid WAV file" in str(exc_info.value)
    
    def test_validate_wav_file_nonexistent(self, tmp_path):
        """Test validation of a nonexistent WAV file."""
        with pytest.raises(AudioFormatError) as exc_info:
            AudioValidator.validate_wav_file(str(tmp_path / "nonexistent.wav"))
        assert "Invalid WAV file" in str(exc_info.value)
    
    def test_validate_audio_file_wav(self, temp_wav_file):
        """Test validation of a WAV file using validate_audio_file."""
        properties = AudioValidator.validate_audio_file(str(temp_wav_file))
        assert properties is not None
        assert properties['channels'] == 1
        assert properties['frame_rate'] == 16000
        assert properties['duration'] == pytest.approx(1.0)
    
    def test_validate_audio_file_mp3(self, temp_mp3_file):
        """Test validation of an MP3 file."""
        properties = AudioValidator.validate_audio_file(str(temp_mp3_file))
        assert properties is not None
        assert properties['channels'] == 1
        assert properties['frame_rate'] == 16000
        assert properties['duration'] > 0
    
    def test_validate_audio_file_unsupported(self, tmp_path):
        """Test validation of an unsupported file format."""
        unsupported_file = tmp_path / "test.ogg"
        unsupported_file.touch()
        with pytest.raises(AudioFormatError) as exc_info:
            AudioValidator.validate_audio_file(str(unsupported_file))
        assert "Unsupported audio format" in str(exc_info.value)
    
    def test_validate_audio_file_invalid(self, invalid_wav_file):
        """Test validation of an invalid audio file."""
        with pytest.raises(AudioFormatError) as exc_info:
            AudioValidator.validate_audio_file(str(invalid_wav_file))
        assert "Invalid audio file" in str(exc_info.value) 