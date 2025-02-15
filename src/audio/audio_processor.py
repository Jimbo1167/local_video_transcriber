from pathlib import Path
import os
from typing import Optional
import numpy as np
from pydub import AudioSegment
from .audio_validator import AudioValidator, AudioFormatError

class AudioProcessingError(Exception):
    """Raised when there are issues processing audio files"""
    pass

class AudioProcessor:
    """Handles audio file processing and conversion"""
    
    def __init__(self, target_sample_rate: int = 16000):
        """
        Initialize the audio processor.
        
        Args:
            target_sample_rate: Target sample rate for processed audio (default: 16000 Hz)
        """
        self.target_sample_rate = target_sample_rate
        self.validator = AudioValidator()
    
    def convert_to_wav(self, audio_path: str, output_path: Optional[str] = None) -> str:
        """
        Convert any supported audio format to WAV.
        
        Args:
            audio_path: Path to the input audio file
            output_path: Optional path for the output WAV file
            
        Returns:
            str: Path to the converted WAV file
            
        Raises:
            AudioProcessingError: If conversion fails
        """
        try:
            # Validate input file
            if not self.validator.is_valid_format(audio_path):
                raise AudioProcessingError(f"Unsupported audio format: {Path(audio_path).suffix}")
            
            # If file is already WAV and no output path specified, return as is
            if audio_path.lower().endswith('.wav') and not output_path:
                # Still need to check if it needs conversion
                audio = AudioSegment.from_wav(audio_path)
                if (audio.channels == 1 and 
                    audio.frame_rate == self.target_sample_rate):
                    return audio_path
            
            # Generate output path if not provided
            if not output_path:
                output_path = str(Path(audio_path).with_suffix('.wav'))
            
            # Convert file using pydub
            audio = AudioSegment.from_file(audio_path)
            
            # Ensure mono audio
            if audio.channels > 1:
                audio = audio.set_channels(1)
            
            # Ensure target sample rate
            if audio.frame_rate != self.target_sample_rate:
                audio = audio.set_frame_rate(self.target_sample_rate)
            
            # Export to WAV
            audio.export(output_path, format='wav')
            
            return output_path
            
        except Exception as e:
            raise AudioProcessingError(f"Failed to convert audio: {str(e)}")
    
    def normalize_audio(self, audio_path: str, output_path: Optional[str] = None) -> str:
        """
        Normalize audio volume and convert to mono if necessary.
        
        Args:
            audio_path: Path to the input audio file
            output_path: Optional path for the output file
            
        Returns:
            str: Path to the normalized audio file
            
        Raises:
            AudioProcessingError: If normalization fails
        """
        try:
            # Convert to WAV first if needed
            wav_path = self.convert_to_wav(audio_path)
            
            # Load audio
            audio = AudioSegment.from_wav(wav_path)
            
            # Normalize volume
            normalized = audio.normalize()
            
            # Determine output path
            if not output_path:
                output_path = wav_path
            
            # Export normalized audio
            normalized.export(output_path, format='wav')
            
            # Clean up temporary WAV if it was created
            if wav_path != audio_path and wav_path != output_path:
                os.remove(wav_path)
            
            return output_path
            
        except Exception as e:
            raise AudioProcessingError(f"Failed to normalize audio: {str(e)}")
    
    def get_audio_duration(self, audio_path: str) -> float:
        """
        Get the duration of an audio file in seconds.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            float: Duration in seconds
            
        Raises:
            AudioProcessingError: If duration cannot be determined
        """
        try:
            properties = self.validator.validate_audio_file(audio_path)
            return properties['duration']
        except Exception as e:
            raise AudioProcessingError(f"Failed to get audio duration: {str(e)}")
    
    def trim_audio(self, audio_path: str, start_sec: float, end_sec: float, 
                  output_path: Optional[str] = None) -> str:
        """
        Trim audio file to specified start and end times.
        
        Args:
            audio_path: Path to the audio file
            start_sec: Start time in seconds
            end_sec: End time in seconds
            output_path: Optional path for the output file
            
        Returns:
            str: Path to the trimmed audio file
            
        Raises:
            AudioProcessingError: If trimming fails
        """
        try:
            # Convert to WAV first if needed
            wav_path = self.convert_to_wav(audio_path)
            
            # Load audio
            audio = AudioSegment.from_wav(wav_path)
            
            # Validate time values
            duration = len(audio) / 1000.0  # Convert to seconds
            if start_sec < 0 or end_sec > duration or start_sec >= end_sec:
                raise AudioProcessingError(
                    f"Invalid time values: start={start_sec}, end={end_sec}, duration={duration}"
                )
            
            # Convert seconds to milliseconds
            start_ms = int(start_sec * 1000)
            end_ms = int(end_sec * 1000)
            
            # Trim audio
            trimmed = audio[start_ms:end_ms]
            
            # Determine output path
            if not output_path:
                output_path = str(Path(wav_path).with_stem(f"{Path(wav_path).stem}_trimmed"))
            
            # Export trimmed audio
            trimmed.export(output_path, format='wav')
            
            # Clean up temporary WAV if it was created
            if wav_path != audio_path and wav_path != output_path:
                os.remove(wav_path)
            
            return output_path
            
        except Exception as e:
            raise AudioProcessingError(f"Failed to trim audio: {str(e)}") 