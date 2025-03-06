#!/usr/bin/env python3
"""
Script to transcribe audio or video files using streaming to reduce memory usage.
"""

import os
import sys
import time
import argparse
import logging
from typing import Dict, Any

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import Config
from src.transcriber import Transcriber

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def display_segment(segment: Dict[str, Any], include_words: bool = False):
    """Display a transcription segment."""
    start = segment["start"]
    end = segment["end"]
    text = segment["text"]
    
    # Format timestamp as [MM:SS.mmm]
    start_str = f"{int(start // 60):02d}:{int(start % 60):02d}.{int((start % 1) * 1000):03d}"
    end_str = f"{int(end // 60):02d}:{int(end % 60):02d}.{int((end % 1) * 1000):03d}"
    
    # Display the segment
    print(f"[{start_str} --> {end_str}] {text}")
    
    # Display words if requested
    if include_words and "words" in segment and segment["words"]:
        print("  Words:")
        for word in segment["words"]:
            word_start = word["start"]
            word_end = word["end"]
            word_text = word["word"]
            
            # Format timestamp as [MM:SS.mmm]
            word_start_str = f"{int(word_start // 60):02d}:{int(word_start % 60):02d}.{int((word_start % 1) * 1000):03d}"
            word_end_str = f"{int(word_end // 60):02d}:{int(word_end % 60):02d}.{int((word_end % 1) * 1000):03d}"
            
            print(f"    [{word_start_str} --> {word_end_str}] {word_text}")
    
    print()  # Empty line for readability

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Transcribe audio or video files using streaming")
    parser.add_argument("input_path", help="Path to the audio or video file")
    parser.add_argument("--diarize", action="store_true", help="Include speaker diarization")
    parser.add_argument("--words", action="store_true", help="Include word-level timestamps")
    parser.add_argument("--model", default="medium", help="Whisper model size (tiny, base, small, medium, large)")
    parser.add_argument("--language", default="en", help="Language code (e.g., en, fr, de)")
    parser.add_argument("--output", help="Output file path (optional)")
    
    args = parser.parse_args()
    
    # Validate input path
    if not os.path.exists(args.input_path):
        logger.error(f"Input file not found: {args.input_path}")
        return 1
    
    # Create configuration
    config = Config()
    config.whisper_model_size = args.model
    config.language = args.language
    config.include_diarization = args.diarize
    
    # Create transcriber
    transcriber = Transcriber(config)
    
    # Start timing
    start_time = time.time()
    
    try:
        print(f"Starting streaming transcription of {args.input_path}")
        print(f"Model: {args.model}, Language: {args.language}, Diarization: {'Yes' if args.diarize else 'No'}")
        print("-" * 80)
        
        # Open output file if specified
        output_file = None
        if args.output:
            output_file = open(args.output, "w", encoding="utf-8")
        
        # Choose the appropriate transcription method
        if args.diarize:
            segments_generator = transcriber.transcribe_stream_with_diarization(args.input_path)
        else:
            segments_generator = transcriber.transcribe_stream(args.input_path)
        
        # Process segments as they become available
        for segment in segments_generator:
            # Display the segment
            display_segment(segment, args.words)
            
            # Write to output file if specified
            if output_file:
                start_str = f"{int(segment['start'] // 60):02d}:{int(segment['start'] % 60):02d}.{int((segment['start'] % 1) * 1000):03d}"
                end_str = f"{int(segment['end'] // 60):02d}:{int(segment['end'] % 60):02d}.{int((segment['end'] % 1) * 1000):03d}"
                
                speaker = f" ({segment['speaker']})" if "speaker" in segment and segment["speaker"] else ""
                output_file.write(f"[{start_str} --> {end_str}]{speaker} {segment['text']}\n")
        
        # Close output file if opened
        if output_file:
            output_file.close()
            print(f"Transcript saved to {args.output}")
        
        # Display timing information
        elapsed = time.time() - start_time
        print(f"Transcription completed in {elapsed:.1f} seconds")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nTranscription interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 