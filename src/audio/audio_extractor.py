from pathlib import Path
import os
from typing import Optional
import signal
from contextlib import contextmanager
from moviepy.editor import VideoFileClip
from .audio_processor import AudioProcessor, AudioProcessingError

class ExtractionTimeoutError(Exception):
    """Raised when audio extraction times out"""
    pass

class AudioExtractionError(Exception):
    """Raised when there are issues extracting audio from video"""
    pass

@contextmanager
def timeout(seconds: float, error_message: str = "Operation timed out"):
    """Context manager for timing out operations using SIGALRM.
    
    Args:
        seconds: Number of seconds to wait before timeout (can be float)
        error_message: Message to include in the timeout error
    """
    def signal_handler(signum, frame):
        raise ExtractionTimeoutError(error_message)
    
    # Convert float seconds to integer seconds, minimum 1 second
    seconds_int = max(1, int(seconds))
    
    # Set the signal handler and alarm
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds_int)
    
    try:
        yield
    finally:
        # Disable the alarm
        signal.alarm(0)

class AudioExtractor:
    """Handles extracting audio from video files"""
    
    def __init__(self, processor: Optional[AudioProcessor] = None, timeout_seconds: int = 300):
        """
        Initialize the audio extractor.
        
        Args:
            processor: Optional AudioProcessor instance for post-processing
            timeout_seconds: Timeout duration in seconds (default: 300)
        """
        self.processor = processor or AudioProcessor()
        self.timeout_seconds = timeout_seconds
    
    def extract_audio(self, video_path: str, output_path: Optional[str] = None,
                     normalize: bool = True) -> str:
        """
        Extract audio from a video file.
        
        Args:
            video_path: Path to the video file
            output_path: Optional path for the output audio file
            normalize: Whether to normalize the extracted audio
            
        Returns:
            str: Path to the extracted audio file
            
        Raises:
            AudioExtractionError: If extraction fails
            ExtractionTimeoutError: If extraction times out
        """
        try:
            # Generate output path if not provided
            if not output_path:
                output_path = str(Path(video_path).with_suffix('.wav'))
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Extract audio with timeout
            with timeout(self.timeout_seconds, "Audio extraction timed out"):
                video = VideoFileClip(video_path)
                
                if video.audio is None:
                    raise AudioExtractionError("No audio stream found in video file")
                
                # Extract audio
                video.audio.write_audiofile(
                    output_path,
                    fps=self.processor.target_sample_rate,
                    nbytes=2,  # 16-bit audio
                    buffersize=2000,
                    logger=None  # Disable progress bar
                )
                
                # Clean up video file handler
                video.close()
            
            # Normalize audio if requested
            if normalize:
                output_path = self.processor.normalize_audio(output_path)
            
            return output_path
            
        except ExtractionTimeoutError:
            # Clean up partial output file
            if os.path.exists(output_path):
                os.remove(output_path)
            raise
        except Exception as e:
            # Clean up partial output file
            if os.path.exists(output_path):
                os.remove(output_path)
            raise AudioExtractionError(f"Failed to extract audio: {str(e)}")
    
    def extract_audio_segment(self, video_path: str, start_sec: float, end_sec: float,
                            output_path: Optional[str] = None, normalize: bool = True) -> str:
        """
        Extract a segment of audio from a video file.
        
        Args:
            video_path: Path to the video file
            start_sec: Start time in seconds
            end_sec: End time in seconds
            output_path: Optional path for the output audio file
            normalize: Whether to normalize the extracted audio
            
        Returns:
            str: Path to the extracted audio segment
            
        Raises:
            AudioExtractionError: If extraction fails
            ExtractionTimeoutError: If extraction times out
        """
        try:
            # First extract the full audio
            temp_path = str(Path(video_path).with_suffix('.temp.wav'))
            full_audio_path = self.extract_audio(video_path, temp_path, normalize=False)
            
            # Then trim to the desired segment
            output_path = self.processor.trim_audio(
                full_audio_path,
                start_sec,
                end_sec,
                output_path
            )
            
            # Clean up temporary file
            os.remove(temp_path)
            
            # Normalize if requested
            if normalize:
                output_path = self.processor.normalize_audio(output_path)
            
            return output_path
            
        except Exception as e:
            # Clean up any temporary files
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise AudioExtractionError(f"Failed to extract audio segment: {str(e)}") 