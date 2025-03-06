#!/usr/bin/env python3
"""
Model Server for persistent Whisper model instances.

This script implements a simple server that keeps Whisper models loaded in memory
and provides an API for transcription requests, avoiding the overhead of loading
the model for each transcription.
"""

import os
import sys
import time
import json
import logging
import argparse
import threading
import multiprocessing
from typing import Dict, Any, Optional, List
from http.server import HTTPServer, BaseHTTPRequestHandler
import socketserver
import urllib.parse

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import Config
from src.transcription.engine import TranscriptionEngine
from src.audio.processor import AudioProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Global variables
config = None
transcription_engine = None
audio_processor = None
model_lock = threading.Lock()
stats = {
    "requests": 0,
    "successful": 0,
    "failed": 0,
    "total_processing_time": 0.0,
    "start_time": time.time()
}

class ModelRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the model server."""
    
    def _send_json_response(self, data: Dict[str, Any], status: int = 200):
        """Send a JSON response."""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))
    
    def _send_error(self, message: str, status: int = 400):
        """Send an error response."""
        self._send_json_response({"error": message}, status)
    
    def do_GET(self):
        """Handle GET requests."""
        parsed_path = urllib.parse.urlparse(self.path)
        
        # Handle status endpoint
        if parsed_path.path == '/status':
            # Calculate uptime
            uptime = time.time() - stats["start_time"]
            
            # Get model info
            model_info = {
                "model_size": config.whisper_model_size,
                "language": config.language,
                "device": config.device
            }
            
            # Calculate average processing time
            avg_time = 0.0
            if stats["successful"] > 0:
                avg_time = stats["total_processing_time"] / stats["successful"]
            
            # Send response
            self._send_json_response({
                "status": "running",
                "uptime": uptime,
                "model": model_info,
                "stats": {
                    "requests": stats["requests"],
                    "successful": stats["successful"],
                    "failed": stats["failed"],
                    "avg_processing_time": avg_time
                }
            })
            return
        
        # Handle unknown endpoints
        self._send_error("Unknown endpoint", 404)
    
    def do_POST(self):
        """Handle POST requests."""
        parsed_path = urllib.parse.urlparse(self.path)
        
        # Handle transcribe endpoint
        if parsed_path.path == '/transcribe':
            # Get content length
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self._send_error("Empty request body")
                return
            
            # Parse request body
            try:
                body = self.rfile.read(content_length)
                request_data = json.loads(body.decode('utf-8'))
            except json.JSONDecodeError:
                self._send_error("Invalid JSON")
                return
            
            # Check for required fields
            if 'audio_path' not in request_data:
                self._send_error("Missing required field: audio_path")
                return
            
            # Get audio path
            audio_path = request_data.get('audio_path')
            if not os.path.exists(audio_path):
                self._send_error(f"Audio file not found: {audio_path}", 404)
                return
            
            # Update stats
            stats["requests"] += 1
            
            # Process the request
            try:
                start_time = time.time()
                
                # Acquire lock to ensure exclusive access to the model
                with model_lock:
                    # Transcribe the audio
                    segments = transcription_engine.transcribe(audio_path)
                
                # Convert segments to a serializable format
                result = []
                for segment in segments:
                    result.append({
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": segment["text"]
                    })
                
                # Update stats
                elapsed = time.time() - start_time
                stats["successful"] += 1
                stats["total_processing_time"] += elapsed
                
                # Send response
                self._send_json_response({
                    "success": True,
                    "segments": result,
                    "processing_time": elapsed
                })
                
            except Exception as e:
                logger.error(f"Error processing request: {str(e)}")
                stats["failed"] += 1
                self._send_error(f"Error processing request: {str(e)}", 500)
            
            return
        
        # Handle unknown endpoints
        self._send_error("Unknown endpoint", 404)


class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""
    pass


def initialize_models(config_path: Optional[str] = None):
    """Initialize the models and processors."""
    global config, transcription_engine, audio_processor
    
    logger.info("Initializing models...")
    
    # Load configuration
    config = Config(config_path or ".env")
    
    # Initialize audio processor
    audio_processor = AudioProcessor(config)
    
    # Initialize transcription engine
    transcription_engine = TranscriptionEngine(config)
    
    # Force model loading
    logger.info(f"Loading Whisper model: {config.whisper_model_size}")
    transcription_engine._load_model()
    
    logger.info("Models initialized successfully")


def run_server(host: str, port: int):
    """Run the model server."""
    server_address = (host, port)
    httpd = ThreadedHTTPServer(server_address, ModelRequestHandler)
    
    logger.info(f"Starting model server on {host}:{port}")
    logger.info(f"Model: {config.whisper_model_size}, Device: {config.device}")
    logger.info("Press Ctrl+C to stop the server")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        httpd.server_close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run a model server for persistent Whisper model instances"
    )
    parser.add_argument("--host", default="localhost",
                       help="Host to bind the server to (default: localhost)")
    parser.add_argument("--port", "-p", type=int, default=8000,
                       help="Port to bind the server to (default: 8000)")
    parser.add_argument("--config", "-c",
                       help="Path to configuration file (default: .env)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize models
    initialize_models(args.config)
    
    # Run server
    run_server(args.host, args.port)
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 