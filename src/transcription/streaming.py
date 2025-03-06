import os
import time
import logging
import numpy as np
from typing import List, Dict, Any, Generator, Optional, Iterator
from queue import Queue
from threading import Thread

from faster_whisper import WhisperModel

from ..config import Config
from ..audio.processor import AudioProcessor

logger = logging.getLogger(__name__)

class StreamingTranscriber:
    """Handles streaming transcription of audio data using the Whisper model."""
    
    def __init__(self, whisper_model: WhisperModel, config: Config):
        """Initialize the streaming transcriber.
        
        Args:
            whisper_model: Initialized WhisperModel instance
            config: Configuration object
        """
        self.whisper = whisper_model
        self.config = config
        self.language = config.language
        self.sample_rate = 16000  # Whisper expects 16kHz audio
        
        # Buffer for collecting audio chunks
        self.buffer_size = 30 * self.sample_rate  # 30 seconds buffer
        self.min_chunk_size = 5 * self.sample_rate  # 5 seconds minimum for processing
        
        logger.info(f"Streaming transcriber initialized with buffer size: {self.buffer_size/self.sample_rate:.1f}s")
    
    def process_stream(self, audio_stream: Iterator[np.ndarray]) -> Generator[Dict[str, Any], None, None]:
        """Process an audio stream and yield transcription segments as they become available.
        
        Args:
            audio_stream: Iterator yielding chunks of audio data as numpy arrays
            
        Yields:
            Transcription segments as they become available
        """
        buffer = np.array([], dtype=np.float32)
        
        for chunk in audio_stream:
            # Add chunk to buffer
            if len(buffer) == 0:
                buffer = chunk
            else:
                buffer = np.concatenate([buffer, chunk])
            
            # Process buffer if it's large enough
            if len(buffer) >= self.min_chunk_size:
                logger.debug(f"Processing buffer of size {len(buffer)/self.sample_rate:.1f}s")
                
                # Process the audio buffer
                segments, _ = self.whisper.transcribe(
                    buffer,
                    language=self.language,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500)
                )
                
                # Convert segments to dictionaries and yield them
                for segment in segments:
                    yield {
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text.strip(),
                        "words": [{"start": word.start, "end": word.end, "word": word.word} 
                                 for word in segment.words] if segment.words else []
                    }
                
                # Keep a small overlap for context (last 0.5 seconds)
                overlap_samples = int(0.5 * self.sample_rate)
                if len(buffer) > overlap_samples:
                    buffer = buffer[-overlap_samples:]
                else:
                    buffer = np.array([], dtype=np.float32)
        
        # Process any remaining audio in the buffer
        if len(buffer) > 0:
            logger.debug(f"Processing final buffer of size {len(buffer)/self.sample_rate:.1f}s")
            segments, _ = self.whisper.transcribe(
                buffer,
                language=self.language,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            for segment in segments:
                yield {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "words": [{"start": word.start, "end": word.end, "word": word.word} 
                             for word in segment.words] if segment.words else []
                }

class AsyncStreamingTranscriber:
    """Handles asynchronous streaming transcription using a background thread."""
    
    def __init__(self, whisper_model: WhisperModel, config: Config):
        """Initialize the async streaming transcriber.
        
        Args:
            whisper_model: Initialized WhisperModel instance
            config: Configuration object
        """
        self.streaming_transcriber = StreamingTranscriber(whisper_model, config)
        self.result_queue = Queue()
        self.is_running = False
        self.thread = None
    
    def _process_stream_thread(self, audio_stream: Iterator[np.ndarray]):
        """Background thread function to process the audio stream.
        
        Args:
            audio_stream: Iterator yielding chunks of audio data
        """
        try:
            for segment in self.streaming_transcriber.process_stream(audio_stream):
                self.result_queue.put(segment)
            # Signal end of transcription
            self.result_queue.put(None)
        except Exception as e:
            logger.error(f"Error in streaming transcription thread: {e}")
            self.result_queue.put({"error": str(e)})
            self.result_queue.put(None)
        finally:
            self.is_running = False
    
    def start_processing(self, audio_stream: Iterator[np.ndarray]):
        """Start processing the audio stream in a background thread.
        
        Args:
            audio_stream: Iterator yielding chunks of audio data
        """
        if self.is_running:
            raise RuntimeError("Streaming transcription is already running")
        
        self.is_running = True
        self.thread = Thread(target=self._process_stream_thread, args=(audio_stream,))
        self.thread.daemon = True
        self.thread.start()
    
    def get_results(self) -> Generator[Dict[str, Any], None, None]:
        """Get transcription results as they become available.
        
        Yields:
            Transcription segments as they become available
        """
        while True:
            segment = self.result_queue.get()
            if segment is None:  # End of transcription
                break
            yield segment
    
    def stop(self):
        """Stop the streaming transcription."""
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0) 