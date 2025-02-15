import sys
print(sys.path)
from moviepy.editor import VideoFileClip
import numpy as np
from typing import Generator, Optional
import librosa

class AudioProcessor:
    CHUNK_SIZE = 10 * 1024 * 1024  # 10MB chunks
    
    @staticmethod
    def extract_audio(video_path: str, target_sr: int = 16000) -> np.ndarray:
        """Extract audio from video file and convert to mono with memory-efficient processing."""
        video = VideoFileClip(video_path)
        audio = video.audio
        
        # Get audio duration and calculate number of samples
        duration = audio.duration
        n_samples = int(duration * target_sr)
        
        # Pre-allocate the output array
        output = np.zeros(n_samples, dtype=np.float32)
        
        # Process in chunks
        chunk_samples = int(target_sr * 10)  # 10 seconds per chunk
        n_chunks = int(np.ceil(n_samples / chunk_samples))
        
        for i in range(n_chunks):
            start_sample = i * chunk_samples
            end_sample = min((i + 1) * chunk_samples, n_samples)
            
            # Convert time to seconds for moviepy
            start_time = start_sample / target_sr
            end_time = end_sample / target_sr
            
            # Extract chunk
            chunk = audio.subclip(start_time, end_time).to_soundarray(fps=target_sr)
            
            # Convert to mono if stereo
            if chunk.ndim > 1:
                chunk = chunk.mean(axis=1)
            
            # Store in output array
            output[start_sample:end_sample] = chunk
            
            # Clear memory
            del chunk
        
        # Clean up
        video.close()
        
        return output
    
    @staticmethod
    def process_audio_stream(audio_data: np.ndarray, 
                           chunk_size: int = CHUNK_SIZE) -> Generator[np.ndarray, None, None]:
        """Process audio data in chunks to reduce memory usage."""
        total_samples = len(audio_data)
        
        for start in range(0, total_samples, chunk_size):
            end = min(start + chunk_size, total_samples)
            yield audio_data[start:end]

# src/transcriber.py
from faster_whisper import WhisperModel
from .audio_processor import AudioProcessor
from typing import List, Tuple

class Transcriber:
    def __init__(self, model_size: str = "medium", device: str = "cpu"):
        self.model = WhisperModel(model_size, device=device)
        self.audio_processor = AudioProcessor()

    def transcribe(self, video_path: str) -> List[Tuple[float, float, str]]:
        """
        Transcribe video file and return list of segments.
        Returns: List of tuples (start_time, end_time, text)
        """
        audio = self.audio_processor.extract_audio(video_path)
        segments, _ = self.model.transcribe(
            audio,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        return [(segment.start, segment.end, segment.text) for segment in segments]
