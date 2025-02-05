import os
import time
from typing import List, Tuple
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
import torch
from moviepy import VideoFileClip
from tqdm import tqdm
import signal
from contextlib import contextmanager
import numpy as np

class TimeoutException(Exception):
    """Raised when an operation times out."""
    pass

@contextmanager
def timeout(seconds, message="Operation timed out"):
    """Context manager for timeouts."""
    def signal_handler(signum, frame):
        raise TimeoutException(message)
    
    # Register a function to raise a TimeoutException on the signal
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Disable the alarm
        signal.alarm(0)

class Transcriber:
    def __init__(self):
        load_dotenv()
        
        # Load environment variables
        self.hf_token = os.getenv("HF_TOKEN")
        self.whisper_model_size = os.getenv("WHISPER_MODEL", "base")
        self.diarization_model = os.getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization@2.1")
        self.language = os.getenv("LANGUAGE", "en")
        self.output_format = os.getenv("OUTPUT_FORMAT", "txt").strip().split("#")[0].strip()
        self.include_diarization = os.getenv("INCLUDE_DIARIZATION", "true").lower() == "true"
        
        # Timeouts in seconds
        self.audio_timeout = int(os.getenv("AUDIO_TIMEOUT", "300"))  # 5 minutes
        self.transcribe_timeout = int(os.getenv("TRANSCRIBE_TIMEOUT", "3600"))  # 1 hour
        self.diarize_timeout = int(os.getenv("DIARIZE_TIMEOUT", "3600"))  # 1 hour
        
        # Determine the best available device
        if torch.backends.mps.is_available():
            self.device = "mps"
            print("Using MPS (Metal Performance Shaders) for acceleration")
        elif torch.cuda.is_available():
            self.device = "cuda"
            print("Using CUDA for acceleration")
        else:
            self.device = "cpu"
            print("Using CPU for processing")
        
        print(f"Loading Whisper model ({self.whisper_model_size})...")
        self.whisper = WhisperModel(
            self.whisper_model_size,
            device="cpu",  # Whisper works better on CPU for Apple Silicon
            compute_type="int8"  # Use int8 quantization for better performance
        )
        
        if self.include_diarization:
            print("Loading diarization model...")
            self.diarizer = Pipeline.from_pretrained(
                self.diarization_model,
                use_auth_token=self.hf_token
            )
            if self.device == "mps":
                self.diarizer = self.diarizer.to(torch.device("mps"))
            elif self.device == "cuda":
                self.diarizer = self.diarizer.to(torch.device("cuda"))
    
    def extract_audio(self, video_path: str) -> str:
        """Extract audio from video file with timeout."""
        print("\nExtracting audio from video...")
        start_time = time.time()
        audio_path = video_path.rsplit(".", 1)[0] + ".wav"
        
        try:
            with timeout(self.audio_timeout, "Audio extraction timed out"):
                video = VideoFileClip(video_path)
                duration = video.duration
                print(f"Video duration: {duration:.1f} seconds")
                print("Extracting audio... (this may take a few minutes)")
                
                video.audio.write_audiofile(audio_path, 
                                          logger=None)
                print("\nAudio extraction complete.")
                return audio_path
        except TimeoutException as e:
            print(f"\nError: {e}")
            if os.path.exists(audio_path):
                os.remove(audio_path)
            raise
    
    def transcribe(self, video_path: str) -> List[Tuple[float, float, str, str]]:
        """Transcribe video file with timeouts and progress monitoring."""
        # Extract audio
        try:
            audio_path = self.extract_audio(video_path)
            
            print("\nStarting transcription...")
            with timeout(self.transcribe_timeout, "Transcription timed out"):
                segments, _ = self.whisper.transcribe(
                    audio_path,
                    language=self.language,
                    beam_size=5,
                    vad_filter=True,
                    vad_parameters={"min_silence_duration_ms": 500}
                )
                segments = list(segments)
                print(f"\nTranscription complete. Found {len(segments)} segments.")
            
            if not self.include_diarization:
                return [(s.start, s.end, s.text, "") for s in segments]
            
            # Perform diarization with timeout
            print("\nStarting speaker diarization...")
            with timeout(self.diarize_timeout, "Speaker diarization timed out"):
                diarization = self.diarizer(audio_path)
                print("Speaker diarization complete.")
            
            # Combine transcription with speaker information
            print("\nCombining transcription with speaker information...")
            result = []
            
            # Process in smaller batches to manage memory better
            batch_size = 50
            total_segments = len(segments)
            
            for i in range(0, total_segments, batch_size):
                batch = segments[i:i + batch_size]
                for segment in tqdm(batch, 
                                  desc=f"Processing segments {i+1}-{min(i+batch_size, total_segments)}",
                                  total=len(batch)):
                    speakers = {}
                    for turn, _, speaker in diarization.itertracks(yield_label=True):
                        if turn.start < segment.end and turn.end > segment.start:
                            overlap = min(turn.end, segment.end) - max(turn.start, segment.start)
                            speakers[speaker] = speakers.get(speaker, 0) + overlap
                    
                    speaker = max(speakers.items(), key=lambda x: x[1])[0] if speakers else "UNKNOWN"
                    result.append((segment.start, segment.end, segment.text, speaker))
                
                # Clear some memory after each batch
                if self.device == "mps":
                    torch.mps.empty_cache()
                elif self.device == "cuda":
                    torch.cuda.empty_cache()
            
            # Clean up
            print("\nCleaning up temporary files...")
            os.remove(audio_path)
            
            return result
        except Exception as e:
            print(f"\nError during transcription: {e}")
            if 'audio_path' in locals() and os.path.exists(audio_path):
                os.remove(audio_path)
            raise

    def save_transcript(self, segments: List[Tuple[float, float, str, str]], output_path: str):
        """Save transcription to file"""
        print(f"Writing {len(segments)} segments to {output_path}")
        print(f"Output format: {self.output_format}")
        
        if self.output_format not in ["txt", "srt", "vtt"]:
            raise ValueError(f"Invalid output format: {self.output_format}. Must be one of: txt, srt, vtt")
        
        # Debug: Print first segment
        if segments:
            print(f"Debug - First segment to write: {segments[0]}")
        
        # Create content string first
        content = []
        if self.output_format == "txt":
            for start, end, text, speaker in segments:
                line = f"[{speaker}] {text}\n" if self.include_diarization else f"{text}\n"
                content.append(line)
        elif self.output_format == "srt":
            for i, (start, end, text, speaker) in enumerate(segments, 1):
                content.extend([
                    f"{i}\n",
                    f"{self._format_timestamp(start)} --> {self._format_timestamp(end)}\n",
                    f"[{speaker}] {text}\n" if self.include_diarization else f"{text}\n",
                    "\n"
                ])
        elif self.output_format == "vtt":
            content.append("WEBVTT\n\n")
            for i, (start, end, text, speaker) in enumerate(segments, 1):
                content.extend([
                    f"{i}\n",
                    f"{self._format_timestamp(start, vtt=True)} --> {self._format_timestamp(end, vtt=True)}\n",
                    f"[{speaker}] {text}\n" if self.include_diarization else f"{text}\n",
                    "\n"
                ])
        
        # Debug: Print first few lines of content
        print(f"Debug - First few lines of content to write:")
        print("".join(content[:5]))
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write all content at once
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.writelines(content)
        except Exception as e:
            print(f"Error writing file: {e}")
            raise
        
        # Verify file was written
        if os.path.exists(output_path):
            size = os.path.getsize(output_path)
            print(f"File written successfully. Size: {size} bytes")
            
            # Debug: Read back first few lines
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    first_lines = "".join(f.readlines()[:5])
                print("Debug - First few lines read back from file:")
                print(first_lines)
            except Exception as e:
                print(f"Error reading back file: {e}")
        else:
            print("Warning: File was not created!")
    
    def _format_timestamp(self, seconds: float, vtt: bool = False) -> str:
        """Convert seconds to timestamp format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        if vtt:
            return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", ",") 