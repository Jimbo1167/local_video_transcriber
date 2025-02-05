import sys
print(sys.path)

from src.transcriber import Transcriber
import time

def main():
    # Replace these paths with your video file and desired output location
    video_path = "path/to/your/video.mp4"  # Example: videos/meeting.mp4
    output_path = "transcripts/output.txt"  # Example: transcripts/meeting_transcript.txt
    
    print("\n=== Starting Video Transcription Process ===")
    start_time = time.time()
    
    print("\nInitializing transcriber...")
    transcriber = Transcriber()
    
    print(f"\nTranscribing {video_path}...")
    segments = transcriber.transcribe(video_path)
    
    print(f"\nSaving transcript to {output_path}...")
    transcriber.save_transcript(segments, output_path)
    
    elapsed_time = time.time() - start_time
    print(f"\nDone! Transcript saved.")
    print(f"Total processing time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    print("=====================================")

if __name__ == "__main__":
    main() 