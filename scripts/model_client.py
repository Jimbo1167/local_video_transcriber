#!/usr/bin/env python3
"""
Client script for interacting with the model server.

This script provides a command-line interface for sending transcription requests
to the model server and viewing server status.
"""

import os
import sys
import time
import json
import logging
import argparse
import requests
from typing import Dict, Any, Optional, List
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def get_server_status(server_url: str) -> Dict[str, Any]:
    """Get the status of the model server.
    
    Args:
        server_url: URL of the model server
        
    Returns:
        Server status information
    """
    url = f"{server_url}/status"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error getting server status: {e}")
        sys.exit(1)

def transcribe_file(server_url: str, audio_path: str) -> Dict[str, Any]:
    """Send a transcription request to the model server.
    
    Args:
        server_url: URL of the model server
        audio_path: Path to the audio file
        
    Returns:
        Transcription results
    """
    url = f"{server_url}/transcribe"
    
    # Ensure the audio path is absolute
    audio_path = os.path.abspath(audio_path)
    
    # Check if the file exists
    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found: {audio_path}")
        sys.exit(1)
    
    # Prepare request data
    data = {
        "audio_path": audio_path
    }
    
    try:
        logger.info(f"Sending transcription request for {audio_path}")
        start_time = time.time()
        
        response = requests.post(url, json=data)
        response.raise_for_status()
        
        elapsed = time.time() - start_time
        logger.info(f"Request completed in {elapsed:.1f} seconds")
        
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error sending transcription request: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_data = e.response.json()
                logger.error(f"Server error: {error_data.get('error', 'Unknown error')}")
            except:
                logger.error(f"Server response: {e.response.text}")
        sys.exit(1)

def display_transcription(result: Dict[str, Any], output_path: Optional[str] = None):
    """Display transcription results and optionally save to a file.
    
    Args:
        result: Transcription results
        output_path: Path to save the results to (optional)
    """
    segments = result.get("segments", [])
    processing_time = result.get("processing_time", 0.0)
    
    print("\n=== Transcription Results ===")
    print(f"Processing time: {processing_time:.1f} seconds")
    print(f"Segments: {len(segments)}")
    print("-----------------------------")
    
    for i, segment in enumerate(segments):
        start = segment["start"]
        end = segment["end"]
        text = segment["text"]
        
        # Format timestamp as [MM:SS.mmm]
        start_str = f"{int(start // 60):02d}:{int(start % 60):02d}.{int((start % 1) * 1000):03d}"
        end_str = f"{int(end // 60):02d}:{int(end % 60):02d}.{int((end % 1) * 1000):03d}"
        
        print(f"[{start_str} --> {end_str}] {text}")
    
    print("-----------------------------")
    
    # Save to file if requested
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("=== Transcription Results ===\n")
            f.write(f"Processing time: {processing_time:.1f} seconds\n")
            f.write(f"Segments: {len(segments)}\n")
            f.write("-----------------------------\n")
            
            for segment in segments:
                start = segment["start"]
                end = segment["end"]
                text = segment["text"]
                
                # Format timestamp as [MM:SS.mmm]
                start_str = f"{int(start // 60):02d}:{int(start % 60):02d}.{int((start % 1) * 1000):03d}"
                end_str = f"{int(end // 60):02d}:{int(end % 60):02d}.{int((end % 1) * 1000):03d}"
                
                f.write(f"[{start_str} --> {end_str}] {text}\n")
            
            f.write("-----------------------------\n")
        
        logger.info(f"Results saved to {output_path}")

def display_server_status(status: Dict[str, Any]):
    """Display server status information.
    
    Args:
        status: Server status information
    """
    uptime = status.get("uptime", 0.0)
    model = status.get("model", {})
    stats = status.get("stats", {})
    
    # Format uptime
    days = int(uptime // (24 * 3600))
    hours = int((uptime % (24 * 3600)) // 3600)
    minutes = int((uptime % 3600) // 60)
    seconds = int(uptime % 60)
    
    if days > 0:
        uptime_str = f"{days}d {hours}h {minutes}m {seconds}s"
    elif hours > 0:
        uptime_str = f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        uptime_str = f"{minutes}m {seconds}s"
    else:
        uptime_str = f"{seconds}s"
    
    print("\n=== Server Status ===")
    print(f"Status: {status.get('status', 'unknown')}")
    print(f"Uptime: {uptime_str}")
    print("\nModel Information:")
    print(f"  Model Size: {model.get('model_size', 'unknown')}")
    print(f"  Language: {model.get('language', 'unknown')}")
    print(f"  Device: {model.get('device', 'unknown')}")
    print("\nStatistics:")
    print(f"  Total Requests: {stats.get('requests', 0)}")
    print(f"  Successful: {stats.get('successful', 0)}")
    print(f"  Failed: {stats.get('failed', 0)}")
    print(f"  Average Processing Time: {stats.get('avg_processing_time', 0.0):.1f} seconds")
    print("=====================")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Client for interacting with the model server"
    )
    parser.add_argument("--server", default="http://localhost:8000",
                       help="URL of the model server (default: http://localhost:8000)")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Get server status")
    
    # Transcribe command
    transcribe_parser = subparsers.add_parser("transcribe", help="Transcribe an audio file")
    transcribe_parser.add_argument("audio_path", help="Path to the audio file")
    transcribe_parser.add_argument("--output", "-o", help="Path to save the results to")
    
    args = parser.parse_args()
    
    # Default to status if no command is specified
    if not args.command:
        args.command = "status"
    
    # Execute the command
    if args.command == "status":
        status = get_server_status(args.server)
        display_server_status(status)
    elif args.command == "transcribe":
        result = transcribe_file(args.server, args.audio_path)
        display_transcription(result, args.output)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 