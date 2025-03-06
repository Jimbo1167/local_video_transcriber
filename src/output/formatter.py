import os
import json
import logging
from typing import List, Tuple, Dict, Any, Optional
import math

from ..config import Config

logger = logging.getLogger(__name__)

class OutputFormatter:
    """Handles formatting and saving transcripts in different formats."""
    
    def __init__(self, config: Config):
        """Initialize the output formatter.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.format = config.output_format
    
    def save_transcript(self, segments: List[Tuple[float, float, str, str]], output_path: str):
        """Save transcription to file in the specified format.
        
        Args:
            segments: List of (start_time, end_time, text, speaker) tuples
            output_path: Path to save the transcript
            
        Raises:
            ValueError: If the output format is not supported
            IOError: If there's an error writing the file
        """
        logger.info(f"Saving transcript in {self.format} format to {output_path}")
        
        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            if self.format == "txt":
                self._save_txt(segments, output_path)
            elif self.format == "srt":
                self._save_srt(segments, output_path)
            elif self.format == "vtt":
                self._save_vtt(segments, output_path)
            elif self.format == "json":
                self._save_json(segments, output_path)
            else:
                raise ValueError(f"Unsupported output format: {self.format}")
                
            logger.info(f"Transcript saved successfully to {output_path}")
        except Exception as e:
            logger.error(f"Error saving transcript: {e}")
            raise
    
    def _format_timestamp(self, seconds: float, vtt: bool = False) -> str:
        """Format seconds as timestamp.
        
        Args:
            seconds: Time in seconds
            vtt: Whether to use VTT format (with milliseconds)
            
        Returns:
            Formatted timestamp string
        """
        hours = math.floor(seconds / 3600)
        minutes = math.floor((seconds % 3600) / 60)
        seconds = seconds % 60
        
        if vtt:
            return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace(".", ".")
        else:
            return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace(".", ",")
    
    def _save_txt(self, segments: List[Tuple[float, float, str, str]], output_path: str):
        """Save transcript in plain text format.
        
        Args:
            segments: List of (start_time, end_time, text, speaker) tuples
            output_path: Path to save the transcript
        """
        with open(output_path, "w", encoding="utf-8") as f:
            for start, end, text, speaker in segments:
                timestamp = f"[{self._format_timestamp(start, False)} --> {self._format_timestamp(end, False)}]"
                if speaker:
                    f.write(f"{timestamp} {speaker}: {text}\n")
                else:
                    f.write(f"{timestamp} {text}\n")
    
    def _save_srt(self, segments: List[Tuple[float, float, str, str]], output_path: str):
        """Save transcript in SRT format.
        
        Args:
            segments: List of (start_time, end_time, text, speaker) tuples
            output_path: Path to save the transcript
        """
        with open(output_path, "w", encoding="utf-8") as f:
            for i, (start, end, text, speaker) in enumerate(segments, 1):
                f.write(f"{i}\n")
                f.write(f"{self._format_timestamp(start)} --> {self._format_timestamp(end)}\n")
                if speaker:
                    f.write(f"{speaker}: {text}\n\n")
                else:
                    f.write(f"{text}\n\n")
    
    def _save_vtt(self, segments: List[Tuple[float, float, str, str]], output_path: str):
        """Save transcript in WebVTT format.
        
        Args:
            segments: List of (start_time, end_time, text, speaker) tuples
            output_path: Path to save the transcript
        """
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
            for i, (start, end, text, speaker) in enumerate(segments, 1):
                f.write(f"{i}\n")
                f.write(f"{self._format_timestamp(start, True)} --> {self._format_timestamp(end, True)}\n")
                if speaker:
                    f.write(f"{speaker}: {text}\n\n")
                else:
                    f.write(f"{text}\n\n")
    
    def _save_json(self, segments: List[Tuple[float, float, str, str]], output_path: str):
        """Save transcript in JSON format.
        
        Args:
            segments: List of (start_time, end_time, text, speaker) tuples
            output_path: Path to save the transcript
        """
        json_data = []
        for start, end, text, speaker in segments:
            json_data.append({
                "start": start,
                "end": end,
                "text": text,
                "speaker": speaker
            })
            
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    def format_transcript_for_display(self, segments: List[Tuple[float, float, str, str]]) -> str:
        """Format transcript for display in the console.
        
        Args:
            segments: List of (start_time, end_time, text, speaker) tuples
            
        Returns:
            Formatted transcript string
        """
        lines = []
        for start, end, text, speaker in segments:
            timestamp = f"[{self._format_timestamp(start, False)} --> {self._format_timestamp(end, False)}]"
            if speaker:
                lines.append(f"{timestamp} {speaker}: {text}")
            else:
                lines.append(f"{timestamp} {text}")
                
        return "\n".join(lines) 