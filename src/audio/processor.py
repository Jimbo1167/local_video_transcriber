import os
import time
import logging
from typing import Tuple, Optional, Generator, Iterator
import numpy as np
from moviepy.editor import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from contextlib import contextmanager
import threading

from ..config import Config
from ..cache.manager import CacheManager

logger = logging.getLogger(__name__)

class TimeoutException(Exception):
    """Raised when an operation times out."""
    pass

@contextmanager
def timeout(seconds, message="Operation timed out"):
    """Thread-based timeout context manager."""
    timer = None
    exception = TimeoutException(message)
    
    def timeout_handler():
        nonlocal exception
        raise exception
    
    try:
        timer = threading.Timer(seconds, timeout_handler)
        timer.start()
        yield
    finally:
        if timer:
            timer.cancel()

class AudioProcessor:
    """Handles audio extraction and processing for transcription."""
    
    CHUNK_SIZE = 10 * 1024 * 1024  # 10MB chunks
    
    def __init__(self, config: Config):
        """Initialize the audio processor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.timeout_seconds = config.audio_timeout
        
        # Initialize cache manager if caching is enabled
        self.cache_manager = CacheManager(config) if config.cache_enabled else None
    
    def is_audio_file(self, file_path: str) -> bool:
        """Check if the file is an audio file based on extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if the file is an audio file, False otherwise
        """
        return file_path.lower().endswith(('.wav', '.mp3', '.m4a', '.aac', '.flac', '.ogg'))
    
    def get_audio_path(self, input_path: str) -> tuple[str, bool]:
        """
        Get the path to an audio file, converting if necessary.
        
        Args:
            input_path: Path to the input file (audio or video)
            
        Returns:
            tuple: (audio_path, needs_cleanup) where:
                - audio_path: Path to the WAV audio file
                - needs_cleanup: Boolean indicating if the file needs to be cleaned up after use
            
        Raises:
            FileNotFoundError: If the input file doesn't exist
            ValueError: If the input file is not a supported format
        """
        # Check if the file exists
        if not os.path.exists(input_path):
            logger.error(f"Input file not found: {input_path}")
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Check if we have a cached version
        if self.cache_manager:
            cached_audio = self.cache_manager.get_cached_audio(input_path)
            if cached_audio:
                return cached_audio, False
            
        # If it's already a WAV file, return it
        if input_path.lower().endswith('.wav'):
            logger.info(f"Input is already a WAV file: {input_path}")
            return input_path, False
        
        # If it's an audio file, convert it to WAV
        if self.is_audio_file(input_path):
            # Convert non-WAV audio to WAV
            logger.info(f"Converting audio file to WAV format: {input_path}")
            wav_path = input_path.rsplit(".", 1)[0] + ".wav"
            try:
                with timeout(self.timeout_seconds, "Audio conversion timed out"):
                    audio = AudioFileClip(input_path)
                    audio.write_audiofile(wav_path, logger=None)
                    audio.close()
                    
                    # Cache the converted audio if caching is enabled
                    if self.cache_manager:
                        self.cache_manager.cache_audio(input_path, wav_path)
                        
                    return wav_path, True
            except Exception as e:
                if os.path.exists(wav_path):
                    os.remove(wav_path)
                logger.error(f"Error converting audio: {e}")
                raise Exception(f"Error converting audio: {e}")
        else:
            # It's a video file, extract the audio
            logger.info(f"Extracting audio from video file: {input_path}")
            try:
                wav_path = self.extract_audio(input_path)
                
                # Cache the extracted audio if caching is enabled
                if self.cache_manager:
                    self.cache_manager.cache_audio(input_path, wav_path)
                    
                return wav_path, True
            except Exception as e:
                logger.error(f"Error extracting audio: {e}")
                raise Exception(f"Error extracting audio: {e}")
    
    def extract_audio(self, video_path: str) -> str:
        """Extract audio from a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Path to the extracted audio file
            
        Raises:
            TimeoutException: If audio extraction times out
            Exception: If audio extraction fails
        """
        # Check if the file exists
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        # Create output path
        output_dir = os.path.dirname(video_path)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        wav_path = os.path.join(output_dir, f"{base_name}.wav")
        
        logger.info(f"Extracting audio from video: {video_path}")
        
        try:
            with timeout(self.timeout_seconds, "Audio extraction timed out"):
                # Load the video
                video = VideoFileClip(video_path)
                
                # Check if the video has audio
                if not video.audio:
                    video.close()
                    raise Exception(f"No audio track found in video: {video_path}")
                
                logger.info(f"Video duration: {video.duration} seconds")
                logger.info("Extracting audio... (this may take a few minutes)")
                
                start_time = time.time()
                
                # Extract audio
                video.audio.write_audiofile(wav_path, logger=None)
                
                elapsed = time.time() - start_time
                logger.info(f"Audio extraction complete in {elapsed:.1f} seconds")
                
                video.close()
                return wav_path
        except TimeoutException as e:
            logger.error(f"Timeout during audio extraction: {e}")
            if os.path.exists(wav_path):
                os.remove(wav_path)
            raise
        except Exception as e:
            logger.error(f"Error during audio extraction: {e}")
            if os.path.exists(wav_path):
                os.remove(wav_path)
            raise Exception(f"Error extracting audio: {e}")
    
    def load_audio(self, audio_path: str, target_sr: int = 16000) -> np.ndarray:
        """Load audio file into memory with efficient processing.
        
        Args:
            audio_path: Path to the audio file
            target_sr: Target sample rate
            
        Returns:
            Audio data as numpy array
        """
        logger.info(f"Loading audio file: {audio_path}")
        try:
            import librosa
            audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
            logger.info(f"Loaded audio: {len(audio)/target_sr:.1f} seconds at {target_sr}Hz")
            return audio
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            raise
    
    def process_audio_stream(self, audio_data: np.ndarray, 
                           chunk_size: int = CHUNK_SIZE) -> Generator[np.ndarray, None, None]:
        """Process audio data in chunks to avoid memory issues.
        
        Args:
            audio_data: Audio data as numpy array
            chunk_size: Size of each chunk in bytes
            
        Yields:
            Chunks of audio data
        """
        # Calculate chunk size in samples
        bytes_per_sample = audio_data.itemsize
        chunk_samples = chunk_size // bytes_per_sample
        
        # Process in chunks
        for i in range(0, len(audio_data), chunk_samples):
            yield audio_data[i:i+chunk_samples]
    
    def stream_audio_from_file(self, audio_path: str, chunk_duration: float = 5.0, 
                              target_sr: int = 16000) -> Iterator[np.ndarray]:
        """Stream audio from a file in chunks.
        
        Args:
            audio_path: Path to the audio file
            chunk_duration: Duration of each chunk in seconds
            target_sr: Target sample rate
            
        Yields:
            Chunks of audio data as numpy arrays
        """
        logger.info(f"Streaming audio from file: {audio_path}")
        try:
            import librosa
            import soundfile as sf
            
            # Get audio info without loading the entire file
            info = sf.info(audio_path)
            total_duration = info.duration
            sample_rate = info.samplerate
            
            # Calculate chunk size in frames
            chunk_size = int(chunk_duration * sample_rate)
            
            logger.info(f"Audio file: {total_duration:.1f} seconds at {sample_rate}Hz")
            logger.info(f"Streaming in chunks of {chunk_duration:.1f} seconds")
            
            # Stream the audio in chunks
            with sf.SoundFile(audio_path) as sf_file:
                # Resample if needed
                resampling_ratio = target_sr / sample_rate if sample_rate != target_sr else 1.0
                
                while sf_file.tell() < sf_file.frames:
                    # Read a chunk
                    chunk = sf_file.read(chunk_size)
                    
                    # Convert to float32 if needed
                    if chunk.dtype != np.float32:
                        chunk = chunk.astype(np.float32)
                    
                    # If stereo, convert to mono
                    if len(chunk.shape) > 1 and chunk.shape[1] > 1:
                        chunk = np.mean(chunk, axis=1)
                    
                    # Resample if needed
                    if resampling_ratio != 1.0:
                        chunk = librosa.resample(
                            chunk, 
                            orig_sr=sample_rate, 
                            target_sr=target_sr
                        )
                    
                    yield chunk
                    
            logger.info(f"Finished streaming audio from {audio_path}")
            
        except Exception as e:
            logger.error(f"Error streaming audio: {e}")
            raise 