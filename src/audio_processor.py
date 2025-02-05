import sys
print(sys.path)
from moviepy import VideoFileClip
import numpy as np

class AudioProcessor:
    @staticmethod
    def extract_audio(video_path: str) -> np.ndarray:
        """Extract audio from video file and convert to mono."""
        video = VideoFileClip(video_path)
        audio = video.audio
        return audio.to_soundarray(fps=16000)[:,0]

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
