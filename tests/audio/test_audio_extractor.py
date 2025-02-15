import pytest
import os
from pathlib import Path
from unittest.mock import patch
import time
from src.audio import AudioExtractor, AudioExtractionError, ExtractionTimeoutError

@pytest.fixture
def test_files(request):
    """Get paths to test files."""
    fixtures_dir = Path(__file__).parent.parent / "fixtures"
    return {
        'video': str(fixtures_dir / "test_video.mp4"),
        'video_no_audio': str(fixtures_dir / "test_video_no_audio.mp4"),
        'audio': str(fixtures_dir / "test_audio.wav")
    }

@pytest.fixture
def audio_extractor():
    """Create an AudioExtractor instance with short timeout for testing."""
    return AudioExtractor(timeout_seconds=5)  # 5 seconds should be enough for our small test files

class TestAudioExtractor:
    def test_extract_audio_success(self, audio_extractor, test_files, tmp_path):
        """Test successful audio extraction from video."""
        output_path = str(tmp_path / "output.wav")
        
        # Extract audio
        result_path = audio_extractor.extract_audio(
            test_files['video'],
            output_path,
            normalize=False
        )
        
        # Verify the output
        assert os.path.exists(result_path)
        assert result_path == output_path
        assert os.path.getsize(result_path) > 0
    
    def test_extract_audio_with_normalization(self, audio_extractor, test_files, tmp_path):
        """Test audio extraction with normalization enabled."""
        output_path = str(tmp_path / "output_normalized.wav")
        
        # Extract and normalize audio
        result_path = audio_extractor.extract_audio(
            test_files['video'],
            output_path,
            normalize=True
        )
        
        assert os.path.exists(result_path)
        assert result_path == output_path
        assert os.path.getsize(result_path) > 0
    
    def test_extract_audio_no_audio_stream(self, audio_extractor, test_files, tmp_path):
        """Test extraction from video with no audio stream."""
        output_path = str(tmp_path / "output.wav")
        
        with pytest.raises(AudioExtractionError) as exc_info:
            audio_extractor.extract_audio(
                test_files['video_no_audio'],
                output_path
            )
        assert "No audio stream found" in str(exc_info.value)
        assert not os.path.exists(output_path)
    
    def test_extract_audio_nonexistent_file(self, audio_extractor, tmp_path):
        """Test extraction from non-existent file."""
        output_path = str(tmp_path / "output.wav")
        
        with pytest.raises(AudioExtractionError) as exc_info:
            audio_extractor.extract_audio(
                "nonexistent.mp4",
                output_path
            )
        assert not os.path.exists(output_path)
    
    def test_extract_audio_segment(self, audio_extractor, test_files, tmp_path):
        """Test extracting an audio segment."""
        output_path = str(tmp_path / "segment.wav")
        
        # Extract a 1-second segment
        result_path = audio_extractor.extract_audio_segment(
            test_files['video'],
            start_sec=0.0,
            end_sec=1.0,
            output_path=output_path
        )
        
        assert os.path.exists(result_path)
        assert result_path == output_path
        assert os.path.getsize(result_path) > 0
        
        # Verify no temporary files are left
        temp_files = list(tmp_path.glob("*.temp.wav"))
        assert len(temp_files) == 0
    
    def test_extract_audio_segment_invalid_times(self, audio_extractor, test_files, tmp_path):
        """Test extracting an audio segment with invalid times."""
        output_path = str(tmp_path / "segment.wav")
        
        # Test invalid start time
        with pytest.raises(AudioExtractionError):
            audio_extractor.extract_audio_segment(
                test_files['video'],
                start_sec=-1.0,
                end_sec=1.0,
                output_path=output_path
            )
        
        # Test end time beyond duration
        with pytest.raises(AudioExtractionError):
            audio_extractor.extract_audio_segment(
                test_files['video'],
                start_sec=0.0,
                end_sec=10.0,  # Our test video is only 2 seconds
                output_path=output_path
            )
        
        assert not os.path.exists(output_path)
    
    def test_timeout_handling(self, tmp_path):
        """Test handling of timeout during extraction."""
        # Create an extractor with a 1-second timeout
        quick_extractor = AudioExtractor(timeout_seconds=1)
        output_path = str(tmp_path / "output.wav")
        
        def slow_write_audiofile(*args, **kwargs):
            """Simulate a slow audio write operation."""
            time.sleep(2)  # Sleep for 2 seconds, longer than the timeout
        
        # Patch the write_audiofile method to be slow
        with patch('moviepy.audio.AudioClip.AudioClip.write_audiofile', side_effect=slow_write_audiofile):
            with pytest.raises(ExtractionTimeoutError):
                quick_extractor.extract_audio(
                    "tests/fixtures/test_video.mp4",
                    output_path
                )
            
            assert not os.path.exists(output_path) 