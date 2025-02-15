from pathlib import Path
from typing import Set, Optional
import wave
import contextlib
from pydub import AudioSegment

class AudioFormatError(Exception):
    """Raised when there are issues with audio file format"""
    pass

class AudioValidator:
    """Validates audio files and their formats"""
    
    # Supported audio formats
    SUPPORTED_FORMATS: Set[str] = {'.wav', '.mp3', '.m4a', '.aac'}
    
    @classmethod
    def is_valid_format(cls, file_path: str) -> bool:
        """
        Check if the file has a supported audio format extension.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            bool: True if the format is supported, False otherwise
        """
        return Path(file_path).suffix.lower() in cls.SUPPORTED_FORMATS
    
    @classmethod
    def validate_wav_file(cls, file_path: str) -> Optional[dict]:
        """
        Validate a WAV file and return its properties if valid.
        
        Args:
            file_path: Path to the WAV file
            
        Returns:
            dict: Audio properties if valid, None if invalid
            
        Raises:
            AudioFormatError: If the file is not a valid WAV file
        """
        try:
            with contextlib.closing(wave.open(file_path, 'rb')) as wav_file:
                return {
                    'channels': wav_file.getnchannels(),
                    'sample_width': wav_file.getsampwidth(),
                    'frame_rate': wav_file.getframerate(),
                    'n_frames': wav_file.getnframes(),
                    'duration': wav_file.getnframes() / wav_file.getframerate()
                }
        except (wave.Error, EOFError, FileNotFoundError) as e:
            raise AudioFormatError(f"Invalid WAV file: {str(e)}")
    
    @classmethod
    def validate_audio_file(cls, file_path: str) -> Optional[dict]:
        """
        Validate any supported audio file and return its properties if valid.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            dict: Audio properties if valid, None if invalid
            
        Raises:
            AudioFormatError: If the file is invalid or unsupported
        """
        if not cls.is_valid_format(file_path):
            raise AudioFormatError(f"Unsupported audio format: {Path(file_path).suffix}")
        
        try:
            # For WAV files, use wave module for more detailed validation
            if file_path.lower().endswith('.wav'):
                return cls.validate_wav_file(file_path)
            
            # For other formats, use pydub
            audio = AudioSegment.from_file(file_path)
            return {
                'channels': audio.channels,
                'sample_width': audio.sample_width,
                'frame_rate': audio.frame_rate,
                'duration': len(audio) / 1000.0  # Convert milliseconds to seconds
            }
        except Exception as e:
            raise AudioFormatError(f"Invalid audio file: {str(e)}") 