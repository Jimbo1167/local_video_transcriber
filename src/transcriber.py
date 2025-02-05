import os
from typing import List, Tuple
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
import torch
from moviepy import VideoFileClip
from tqdm import tqdm

class Transcriber:
    def __init__(self):
        load_dotenv()
        
        # Load environment variables
        self.hf_token = os.getenv("HF_TOKEN")
        self.whisper_model_size = os.getenv("WHISPER_MODEL", "base")
        self.diarization_model = os.getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization@2.1")
        self.language = os.getenv("LANGUAGE", "en")
        self.output_format = os.getenv("OUTPUT_FORMAT", "txt")
        self.include_diarization = os.getenv("INCLUDE_DIARIZATION", "true").lower() == "true"
        
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
        # For Whisper, we'll use CPU compute_type for better compatibility with Apple Silicon
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
            # Move diarization model to MPS if available
            if self.device == "mps":
                self.diarizer = self.diarizer.to(torch.device("mps"))
            elif self.device == "cuda":
                self.diarizer = self.diarizer.to(torch.device("cuda"))
    
    def extract_audio(self, video_path: str) -> str:
        """Extract audio from video file"""
        print("Extracting audio from video...")
        audio_path = video_path.rsplit(".", 1)[0] + ".wav"
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path, verbose=False, logger=None)
        print("Audio extraction complete.")
        return audio_path
    
    def transcribe(self, video_path: str) -> List[Tuple[float, float, str, str]]:
        """Transcribe video file with speaker diarization"""
        # Extract audio
        audio_path = self.extract_audio(video_path)
        
        print("Starting transcription...")
        # Use beam_size=5 for better performance/quality trade-off
        segments, _ = self.whisper.transcribe(
            audio_path,
            language=self.language,
            beam_size=5,
            vad_filter=True,  # Enable voice activity detection
            vad_parameters={"min_silence_duration_ms": 500}  # Adjust silence detection
        )
        segments = list(segments)  # Convert generator to list
        print(f"Transcription complete. Found {len(segments)} segments.")
        
        if not self.include_diarization:
            return [(s.start, s.end, s.text, "") for s in segments]
        
        # Perform diarization
        print("Starting speaker diarization...")
        diarization = self.diarizer(audio_path)
        print("Speaker diarization complete.")
        
        # Combine transcription with speaker information
        print("Combining transcription with speaker information...")
        result = []
        
        # Process in smaller batches to manage memory better
        batch_size = 50  # Process 50 segments at a time
        for i in range(0, len(segments), batch_size):
            batch = segments[i:i + batch_size]
            for segment in tqdm(batch, desc=f"Processing segments {i+1}-{min(i+batch_size, len(segments))}"):
                # Find the speaker who talks the most during this segment
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
        print("Cleaning up temporary files...")
        os.remove(audio_path)
        
        return result

    def save_transcript(self, segments: List[Tuple[float, float, str, str]], output_path: str):
        """Save transcription to file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            if self.output_format == "txt":
                for start, end, text, speaker in segments:
                    if self.include_diarization:
                        f.write(f"[{speaker}] {text}\n")
                    else:
                        f.write(f"{text}\n")
            
            elif self.output_format == "srt":
                for i, (start, end, text, speaker) in enumerate(segments, 1):
                    f.write(f"{i}\n")
                    f.write(f"{self._format_timestamp(start)} --> {self._format_timestamp(end)}\n")
                    if self.include_diarization:
                        f.write(f"[{speaker}] {text}\n")
                    else:
                        f.write(f"{text}\n")
                    f.write("\n")
            
            elif self.output_format == "vtt":
                f.write("WEBVTT\n\n")
                for i, (start, end, text, speaker) in enumerate(segments, 1):
                    f.write(f"{i}\n")
                    f.write(f"{self._format_timestamp(start, vtt=True)} --> {self._format_timestamp(end, vtt=True)}\n")
                    if self.include_diarization:
                        f.write(f"[{speaker}] {text}\n")
                    else:
                        f.write(f"{text}\n")
                    f.write("\n")
    
    def _format_timestamp(self, seconds: float, vtt: bool = False) -> str:
        """Convert seconds to timestamp format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        if vtt:
            return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", ",") 