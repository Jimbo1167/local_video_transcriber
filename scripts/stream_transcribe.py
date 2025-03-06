#!/usr/bin/env python3
"""
Script to transcribe audio or video files using streaming to reduce memory usage.

This script provides a memory-efficient approach to transcription by processing
audio in chunks, making it suitable for large files or systems with limited resources.
"""

import os
import sys
import time
import argparse
import logging
from typing import Dict, Any
from pathlib import Path

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
    
    # Add speaker if available
    speaker = f" ({segment['speaker']})" if "speaker" in segment and segment["speaker"] else ""
    
    # Display the segment
    print(f"[{start_str} --> {end_str}]{speaker} {text}")
    
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

def get_default_output_path(input_path, format="txt"):
    """Generate default output path based on input filename.
    
    Args:
        input_path: Path to the input file
        format: Output format extension
        
    Returns:
        Path to the output file
    """
    # Get the input filename without extension
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    # Create output path in transcripts directory with appropriate extension
    return os.path.join("transcripts", f"{base_name}_stream.{format}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Transcribe audio or video files using streaming to reduce memory usage"
    )
    parser.add_argument("input_path", help="Path to the audio or video file")
    parser.add_argument("--output", "-o", help="Output file path (default: transcripts/<input_filename>_stream.txt)")
    parser.add_argument("--diarize", "-d", action="store_true", help="Include speaker diarization")
    parser.add_argument("--words", "-w", action="store_true", help="Include word-level timestamps")
    parser.add_argument("--model", "-m", help="Whisper model size (tiny, base, small, medium, large)")
    parser.add_argument("--language", "-l", help="Language code (e.g., en, fr, de)")
    parser.add_argument("--format", "-f", choices=["txt", "srt", "vtt", "json"], help="Output format (default: txt)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input path
    if not os.path.exists(args.input_path):
        logger.error(f"Input file not found: {args.input_path}")
        return 1
    
    logger.info("\n=== Starting Streaming Transcription Process ===")
    start_time = time.time()
    
    # Create configuration from .env file
    logger.info("Initializing configuration...")
    config = Config(".env")  # Explicitly load from .env file
    
    # Override config with command line arguments
    if args.model:
        config.whisper_model_size = args.model
    if args.language:
        config.language = args.language
    if args.diarize:
        config.include_diarization = True
    if args.format:
        config._output_format = args.format
    
    # Validate the configuration
    if not config.validate():
        logger.error("Invalid configuration. Please check your settings.")
        sys.exit(1)
    
    # Set default output path if not specified
    output_format = args.format if args.format else "txt"
    if args.output is None:
        args.output = get_default_output_path(args.input_path, output_format)
    
    # Create transcriber
    transcriber = Transcriber(config)
    
    try:
        logger.info(f"Starting streaming transcription of {args.input_path}")
        logger.info(f"Model: {config.whisper_model_size}, Language: {config.language}, Diarization: {'Yes' if config.include_diarization else 'No'}")
        logger.info("-" * 80)
        
        # Ensure the output directory exists if output is specified
        if args.output:
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            output_file = open(args.output, "w", encoding="utf-8")
        else:
            output_file = None
        
        # Choose the appropriate transcription method
        if config.include_diarization:
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
            logger.info(f"Transcript saved to {args.output}")
        
        # Display timing information
        elapsed = time.time() - start_time
        logger.info(f"Transcription completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        logger.info("=====================================")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("\nTranscription interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 