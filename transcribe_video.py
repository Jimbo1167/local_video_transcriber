import sys
from src.transcriber import Transcriber
import time
import argparse

def main():
    parser = argparse.ArgumentParser(
        description='Transcribe a video or audio file with speaker diarization. '
                   'Supports video files (mov, mp4, etc.) and audio files (wav, mp3, m4a, aac).'
    )
    parser.add_argument('input_path', 
                       help='Path to the video or audio file to transcribe')
    parser.add_argument('--output', '-o', 
                       default="transcripts/output.txt",
                       help='Output path for the transcript (default: transcripts/output.txt)')
    
    args = parser.parse_args()
    
    print("\n=== Starting Transcription Process ===")
    start_time = time.time()
    
    print("\nInitializing transcriber...")
    transcriber = Transcriber()
    
    print(f"\nProcessing {args.input_path}...")
    segments = transcriber.transcribe(args.input_path)
    
    print(f"\nSaving transcript to {args.output}...")
    transcriber.save_transcript(segments, args.output)
    
    elapsed_time = time.time() - start_time
    print(f"\nDone! Transcript saved.")
    print(f"Total processing time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    print("=====================================")

if __name__ == "__main__":
    main() 