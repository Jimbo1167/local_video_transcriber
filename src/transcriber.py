import os
import time
from typing import List, Tuple, Optional
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
import torch
from moviepy.editor import VideoFileClip
from tqdm import tqdm
import threading
from contextlib import contextmanager
import numpy as np
import concurrent.futures
import warnings

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

class Config:
    """Configuration class to handle all settings."""
    def __init__(self, env_file: Optional[str] = None):
        if env_file:
            load_dotenv(env_file, override=True)
        
        self.hf_token = os.getenv("HF_TOKEN")
        self.whisper_model_size = os.getenv("WHISPER_MODEL", "base")
        self.diarization_model = os.getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization@2.1")
        self.language = os.getenv("LANGUAGE", "en")
        
        # Parse output format
        self._output_format = None
        output_format = os.getenv("OUTPUT_FORMAT")
        if output_format:
            output_format = output_format.strip()
            if "#" in output_format:
                output_format = output_format.split("#")[0].strip()
        self._output_format = output_format if output_format else "txt"
        
        # Parse diarization setting
        diarization = os.getenv("INCLUDE_DIARIZATION", "true")
        self.include_diarization = diarization.strip().lower() in ["true", "1", "yes", "on"]
        
        # Timeouts
        self.audio_timeout = int(os.getenv("AUDIO_TIMEOUT", "300"))
        self.transcribe_timeout = int(os.getenv("TRANSCRIBE_TIMEOUT", "3600"))
        self.diarize_timeout = int(os.getenv("DIARIZE_TIMEOUT", "3600"))

    @property
    def output_format(self) -> str:
        return self._output_format

    @output_format.setter
    def output_format(self, value: str):
        if value:
            value = value.strip()
            if "#" in value:
                value = value.split("#")[0].strip()
        self._output_format = value if value else "txt"

class Transcriber:
    def __init__(self, config: Optional[Config] = None):
        # Use provided config or create new one
        self.config = config or Config()
        
        # Cache directory for models
        self.cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "video_transcriber")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Copy config values to instance variables for backward compatibility
        self.hf_token = self.config.hf_token
        self.whisper_model_size = self.config.whisper_model_size
        self.diarization_model = self.config.diarization_model
        self.language = self.config.language
        self.output_format = self.config.output_format
        self.include_diarization = self.config.include_diarization
        self.audio_timeout = self.config.audio_timeout
        self.transcribe_timeout = self.config.transcribe_timeout
        self.diarize_timeout = self.config.diarize_timeout
        
        # Determine the best available device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using MPS (Metal Performance Shaders) for acceleration")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using CUDA for acceleration")
        else:
            self.device = torch.device("cpu")
            print("Using CPU for processing")
        
        # Configure warnings
        warnings.filterwarnings("ignore", category=FutureWarning, module="speechbrain.utils.autocast")
        warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.core.notebook")
        
        print(f"Loading Whisper model ({self.whisper_model_size})...")
        self.whisper = WhisperModel(
            self.whisper_model_size,
            device="cpu",  # Whisper works better on CPU for Apple Silicon
            compute_type="int8",  # Use int8 quantization for better performance
            download_root=os.path.join(self.cache_dir, "whisper")
        )
        
        if self.include_diarization:
            print("Loading diarization model...")
            self.diarizer = Pipeline.from_pretrained(
                self.diarization_model,
                use_auth_token=self.hf_token,
                cache_dir=os.path.join(self.cache_dir, "diarization")
            )
            # Use the device property directly
            self.diarizer = self.diarizer.to(self.device)
    
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
    
    def _is_audio_file(self, file_path: str) -> bool:
        """Check if the file is an audio file based on extension."""
        return file_path.lower().endswith(('.wav', '.mp3', '.m4a', '.aac'))
    
    def _get_audio_path(self, input_path: str) -> Tuple[str, bool]:
        """Get the audio path and whether it needs cleanup.
        
        Returns:
            Tuple[str, bool]: (audio_path, needs_cleanup)
            - audio_path: Path to the WAV file
            - needs_cleanup: True if we created a temporary file that needs cleanup
        """
        if self._is_audio_file(input_path):
            if input_path.lower().endswith('.wav'):
                return input_path, False
            else:
                # Convert non-WAV audio to WAV
                print("\nConverting audio to WAV format...")
                wav_path = input_path.rsplit(".", 1)[0] + ".wav"
                try:
                    with timeout(self.audio_timeout, "Audio conversion timed out"):
                        from moviepy.audio.io.AudioFileClip import AudioFileClip
                        audio = AudioFileClip(input_path)
                        audio.write_audiofile(wav_path, logger=None)
                        audio.close()
                        return wav_path, True
                except Exception as e:
                    if os.path.exists(wav_path):
                        os.remove(wav_path)
                    raise Exception(f"Error converting audio: {e}")
        else:
            # Extract audio from video
            print("\nExtracting audio from video...")
            wav_path = input_path.rsplit(".", 1)[0] + ".wav"
            try:
                with timeout(self.audio_timeout, "Audio extraction timed out"):
                    video = VideoFileClip(input_path)
                    video.audio.write_audiofile(wav_path, logger=None)
                    video.close()
                    return wav_path, True
            except Exception as e:
                if os.path.exists(wav_path):
                    os.remove(wav_path)
                raise Exception(f"Error extracting audio: {e}")
    
    def _combine_segments_with_speakers(self, segments, diarization):
        """Efficiently combine transcription segments with speaker information."""
        # Pre-process diarization data into a more efficient format
        speaker_ranges = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_ranges.append((turn.start, turn.end, speaker))
        
        # Sort ranges by start time for binary search
        speaker_ranges.sort(key=lambda x: x[0])
        
        def find_speaker(start_time, end_time):
            """Binary search to find overlapping speaker segments."""
            speakers = {}
            left = 0
            right = len(speaker_ranges) - 1
            
            # Find first potential overlapping segment
            while left <= right:
                mid = (left + right) // 2
                if speaker_ranges[mid][1] <= start_time:
                    left = mid + 1
                else:
                    right = mid - 1
            
            # Check all potentially overlapping segments
            i = left
            while i < len(speaker_ranges) and speaker_ranges[i][0] < end_time:
                range_start, range_end, speaker = speaker_ranges[i]
                if range_start < end_time and range_end > start_time:
                    overlap = min(range_end, end_time) - max(range_start, start_time)
                    speakers[speaker] = speakers.get(speaker, 0) + overlap
                i += 1
            
            return max(speakers.items(), key=lambda x: x[1])[0] if speakers else "UNKNOWN"
        
        # Process segments in batches
        result = []
        batch_size = 50
        total_segments = len(segments)
        
        for i in range(0, total_segments, batch_size):
            batch = segments[i:i + batch_size]
            batch_results = []
            
            for segment in tqdm(batch,
                              desc=f"Processing segments {i+1}-{min(i+batch_size, total_segments)}",
                              total=len(batch)):
                speaker = find_speaker(segment.start, segment.end)
                batch_results.append((segment.start, segment.end, segment.text, speaker))
            
            result.extend(batch_results)
            
            # Clear some memory after each batch
            if self.device.type == "mps":
                torch.mps.empty_cache()
            elif self.device.type == "cuda":
                torch.cuda.empty_cache()
        
        return result

    def transcribe(self, input_path: str) -> List[Tuple[float, float, str, str]]:
        """Transcribe video or audio file with timeouts and progress monitoring."""
        try:
            # Get audio path and whether we need to clean it up
            audio_path, needs_cleanup = self._get_audio_path(input_path)
            
            print("\nStarting transcription and diarization...")
            
            def run_transcription():
                with timeout(self.transcribe_timeout, "Transcription timed out"):
                    # Optimize parameters based on model size
                    if self.whisper_model_size in ["medium", "large-v1", "large-v2", "large-v3"]:
                        # Larger models: More aggressive VAD and higher beam size
                        segments, _ = self.whisper.transcribe(
                            audio_path,
                            language=self.language,
                            beam_size=6,  # Increased from 5
                            vad_filter=True,
                            vad_parameters={
                                "min_silence_duration_ms": 700,  # Increased from 500
                                "speech_pad_ms": 400,  # Added padding to avoid cutting words
                                "threshold": 0.45  # Slightly more aggressive voice detection
                            },
                            condition_on_previous_text=True,  # Help with context
                            initial_prompt="This is a conversation between multiple speakers."  # Help with context
                        )
                    else:
                        # Smaller models: Original parameters
                        segments, _ = self.whisper.transcribe(
                            audio_path,
                            language=self.language,
                            beam_size=5,
                            vad_filter=True,
                            vad_parameters={"min_silence_duration_ms": 500}
                        )
                    return list(segments)
            
            def run_diarization():
                if not self.include_diarization:
                    return None
                with timeout(self.diarize_timeout, "Speaker diarization timed out"):
                    return self.diarizer(audio_path)
            
            # Run transcription and diarization concurrently
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future_transcription = executor.submit(run_transcription)
                future_diarization = executor.submit(run_diarization) if self.include_diarization else None
                
                # Wait for transcription to complete
                segments = future_transcription.result()
                print(f"\nTranscription complete. Found {len(segments)} segments.")
                
                # If no diarization needed, return early
                if not self.include_diarization:
                    result = [(s.start, s.end, s.text, "") for s in segments]
                    if needs_cleanup:
                        os.remove(audio_path)
                    return result
                
                # Wait for diarization to complete
                diarization = future_diarization.result()
                print("Speaker diarization complete.")
            
            # Combine transcription with speaker information using optimized method
            print("\nCombining transcription with speaker information...")
            result = self._combine_segments_with_speakers(segments, diarization)
            
            # Clean up temporary files
            if needs_cleanup:
                print("\nCleaning up temporary files...")
                os.remove(audio_path)
            
            return result
        except Exception as e:
            print(f"\nError during transcription: {e}")
            if 'audio_path' in locals() and needs_cleanup and os.path.exists(audio_path):
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