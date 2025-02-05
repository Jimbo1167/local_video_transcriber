import sys
from src.transcriber import Transcriber
import time
import argparse

def main():
    parser = argparse.ArgumentParser(description='Transcribe a video file with speaker diarization.')
    parser.add_argument('video_path', help='Path to the video file to transcribe')
    parser.add_argument('--output', '-o', 
                       default="transcripts/output.txt",
                       help='Output path for the transcript (default: transcripts/output.txt)')
    
    args = parser.parse_args()
    
    print("\n=== Starting Video Transcription Process ===")
    start_time = time.time()
    
    print("\nInitializing transcriber...")
    transcriber = Transcriber()
    
    print(f"\nTranscribing {args.video_path}...")
    segments = transcriber.transcribe(args.video_path)
    
    print(f"\nSaving transcript to {args.output}...")
    transcriber.save_transcript(segments, args.output)
    
    elapsed_time = time.time() - start_time
    print(f"\nDone! Transcript saved.")
    print(f"Total processing time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    print("=====================================")

if __name__ == "__main__":
    main() 