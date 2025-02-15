import pytest
import wave
import numpy as np
from pathlib import Path
from pydub import AudioSegment
from src.audio import AudioProcessor, AudioProcessingError

@pytest.fixture
def audio_processor():
    """Create an AudioProcessor instance."""
    return AudioProcessor(target_sample_rate=16000)

@pytest.fixture
def stereo_wav_file(tmp_path):
    """Create a stereo WAV file for testing."""
    file_path = tmp_path / "stereo.wav"
    sample_rate = 44100
    duration = 1.0  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration))
    # Create stereo audio (two channels)
    left_channel = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    right_channel = np.sin(2 * np.pi * 880 * t)  # 880 Hz sine wave
    stereo = np.vstack((left_channel, right_channel)).T
    
    with wave.open(str(file_path), 'wb') as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(stereo.astype(np.int16).tobytes())
    
    return file_path

@pytest.fixture
def mono_wav_file(tmp_path):
    """Create a mono WAV file for testing."""
    file_path = tmp_path / "mono.wav"
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
def mp3_file(tmp_path, mono_wav_file):
    """Create an MP3 file for testing."""
    file_path = tmp_path / "test.mp3"
    audio = AudioSegment.from_wav(str(mono_wav_file))
    audio.export(str(file_path), format='mp3')
    return file_path

class TestAudioProcessor:
    def test_init_default_sample_rate(self):
        """Test default sample rate initialization."""
        processor = AudioProcessor()
        assert processor.target_sample_rate == 16000
    
    def test_init_custom_sample_rate(self):
        """Test custom sample rate initialization."""
        processor = AudioProcessor(target_sample_rate=44100)
        assert processor.target_sample_rate == 44100
    
    def test_convert_to_wav_already_wav(self, audio_processor, mono_wav_file):
        """Test converting a WAV file that's already in WAV format."""
        output_path = audio_processor.convert_to_wav(str(mono_wav_file))
        assert output_path == str(mono_wav_file)
        
        # Verify the file is still valid
        with wave.open(output_path, 'rb') as wav_file:
            assert wav_file.getnchannels() == 1
            assert wav_file.getframerate() == 16000
    
    def test_convert_to_wav_from_mp3(self, audio_processor, mp3_file, tmp_path):
        """Test converting an MP3 file to WAV format."""
        output_path = audio_processor.convert_to_wav(str(mp3_file))
        assert output_path.endswith('.wav')
        
        # Verify the converted file
        with wave.open(output_path, 'rb') as wav_file:
            assert wav_file.getnchannels() == 1
            assert wav_file.getframerate() == 16000
    
    def test_convert_to_wav_stereo_to_mono(self, audio_processor, stereo_wav_file):
        """Test converting a stereo WAV file to mono."""
        output_path = audio_processor.convert_to_wav(str(stereo_wav_file))
        
        # Verify the converted file is mono
        with wave.open(output_path, 'rb') as wav_file:
            assert wav_file.getnchannels() == 1
            assert wav_file.getframerate() == 16000
    
    def test_convert_to_wav_invalid_format(self, audio_processor, tmp_path):
        """Test converting an invalid format."""
        invalid_file = tmp_path / "test.txt"
        invalid_file.write_text("Not an audio file")
        
        with pytest.raises(AudioProcessingError) as exc_info:
            audio_processor.convert_to_wav(str(invalid_file))
        assert "Unsupported audio format" in str(exc_info.value)
    
    def test_normalize_audio(self, audio_processor, stereo_wav_file):
        """Test audio normalization."""
        output_path = audio_processor.normalize_audio(str(stereo_wav_file))
        
        # Verify the normalized file
        with wave.open(output_path, 'rb') as wav_file:
            assert wav_file.getnchannels() == 1  # Should be converted to mono
            assert wav_file.getframerate() == 16000
    
    def test_get_audio_duration(self, audio_processor, mono_wav_file):
        """Test getting audio duration."""
        duration = audio_processor.get_audio_duration(str(mono_wav_file))
        assert duration == pytest.approx(1.0)
    
    def test_trim_audio(self, audio_processor, mono_wav_file):
        """Test trimming audio."""
        # Trim to middle 0.5 seconds
        output_path = audio_processor.trim_audio(
            str(mono_wav_file),
            start_sec=0.25,
            end_sec=0.75
        )
        
        # Verify the trimmed duration
        duration = audio_processor.get_audio_duration(output_path)
        assert duration == pytest.approx(0.5)
    
    def test_trim_audio_invalid_times(self, audio_processor, mono_wav_file):
        """Test trimming with invalid time values."""
        with pytest.raises(AudioProcessingError):
            audio_processor.trim_audio(str(mono_wav_file), start_sec=2.0, end_sec=3.0)
    
    def test_cleanup_temp_files(self, audio_processor, mp3_file, tmp_path):
        """Test that temporary files are cleaned up."""
        output_path = tmp_path / "output.wav"
        
        # Convert MP3 to WAV
        audio_processor.convert_to_wav(str(mp3_file), str(output_path))
        
        # Check that only the output file exists
        wav_files = [f for f in tmp_path.glob("*.wav") if f.name != "mono.wav"]  # Exclude the fixture file
        assert len(wav_files) == 1
        assert wav_files[0].name == "output.wav" 