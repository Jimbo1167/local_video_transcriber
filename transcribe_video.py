import sys
from src.transcriber import Transcriber, Config
import time
import argparse
import os

def get_default_output_path(input_path, transcriber):
    """Generate default output path based on input filename."""
    # Get the input filename without extension
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    # Create output path in transcripts directory with appropriate extension
    return os.path.join("transcripts", f"{base_name}.{transcriber.output_format}")

def main():
    parser = argparse.ArgumentParser(
        description='Transcribe a video or audio file with speaker diarization. '
                   'Supports video files (mov, mp4, etc.) and audio files (wav, mp3, m4a, aac).'
    )
    parser.add_argument('input_path', 
                       help='Path to the video or audio file to transcribe')
    parser.add_argument('--output', '-o', 
                       help='Output path for the transcript (default: transcripts/<input_filename>.<format>)')
    
    args = parser.parse_args()
    
    print("\n=== Starting Transcription Process ===")
    start_time = time.time()
    
    print("\nInitializing transcriber...")
    config = Config(".env")  # Explicitly load from .env file
    transcriber = Transcriber(config)
    
    # Set default output path if not specified
    if args.output is None:
        args.output = get_default_output_path(args.input_path, transcriber)
    
    print(f"\nProcessing {args.input_path}...")
    segments = transcriber.transcribe(args.input_path)
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    print(f"\nSaving transcript to {args.output}...")
    transcriber.save_transcript(segments, args.output)
    
    elapsed_time = time.time() - start_time
    print(f"\nDone! Transcript saved.")
    print(f"Total processing time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    print("=====================================")

if __name__ == "__main__":
    main() 