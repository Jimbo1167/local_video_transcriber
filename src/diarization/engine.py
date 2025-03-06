import os
import time
import logging
from typing import List, Dict, Any, Optional
import concurrent.futures

from pyannote.audio import Pipeline
import torch
import warnings

from ..config import Config
from ..audio.processor import timeout, TimeoutException
from ..cache.manager import CacheManager

logger = logging.getLogger(__name__)

class DiarizationEngine:
    """Handles speaker diarization for audio files."""
    
    def __init__(self, config: Config, test_mode: bool = False):
        """Initialize the diarization engine.
        
        Args:
            config: Configuration object
            test_mode: If True, use mock models for testing
        """
        self.config = config
        self.timeout_seconds = config.diarize_timeout
        self.include_diarization = config.include_diarization
        self.hf_token = config.hf_token
        self.diarization_model = config.diarization_model
        self.test_mode = test_mode
        
        # Cache directory for models
        self.cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "video_transcriber")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize cache manager if caching is enabled
        self.cache_manager = CacheManager(config) if config.cache_enabled else None
        
        # Configure warnings
        warnings.filterwarnings("ignore", category=FutureWarning, module="speechbrain.utils.autocast")
        warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.core.notebook")
        
        # Determine the best available device
        if config.force_cpu:
            self.device = torch.device("cpu")
            logger.info("Forcing CPU usage as specified in configuration")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("Using MPS (Metal Performance Shaders) for acceleration")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("Using CUDA for acceleration")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU for processing (no GPU acceleration available)")
        
        # Load the diarization model if needed
        self.diarizer = None
        if self.include_diarization:
            self._load_model()
    
    def _load_model(self):
        """Load the diarization model."""
        if not self.include_diarization:
            logger.info("Diarization is disabled, skipping model loading")
            return
            
        if self.test_mode:
            logger.info("Test mode enabled, using mock diarizer")
            # Create a mock diarizer for testing
            class MockDiarizer:
                def __init__(self, device):
                    self.device = device
                
                def to(self, device):
                    return self
                
                def __call__(self, audio_path):
                    class MockDiarization:
                        def itertracks(self, yield_label=False):
                            class Segment:
                                def __init__(self, start, end):
                                    self.start = start
                                    self.end = end
                            
                            return [
                                (Segment(0.0, 2.0), None, "SPEAKER_01"),
                                (Segment(2.0, 4.0), None, "SPEAKER_02")
                            ]
                    return MockDiarization()
            
            self.diarizer = MockDiarizer(self.device)
            logger.info("Mock diarization model loaded successfully")
            return
            
        if not self.hf_token:
            logger.warning("HF_TOKEN is not set. Speaker diarization may not work properly.")
            
        try:
            logger.info(f"Loading diarization model: {self.diarization_model}")
            self.diarizer = Pipeline.from_pretrained(
                self.diarization_model,
                use_auth_token=self.hf_token,
                cache_dir=os.path.join(self.cache_dir, "diarization")
            )
            # Use the device property directly
            self.diarizer = self.diarizer.to(self.device)
            logger.info("Diarization model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading diarization model: {e}")
            raise
    
    def diarize(self, audio_path: str) -> Optional[List[Dict[str, Any]]]:
        """Perform speaker diarization with timeout.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            List of diarization segments or None if diarization is disabled
            
        Raises:
            TimeoutException: If diarization times out
            Exception: If diarization fails
        """
        if not self.include_diarization:
            logger.info("Diarization is disabled, skipping")
            return None
        
        # Check if we have cached results
        if self.cache_manager:
            cached_diarization = self.cache_manager.get_cached_diarization(audio_path)
            if cached_diarization:
                return cached_diarization
            
        if not self.diarizer:
            logger.warning("Diarizer not initialized, attempting to load model")
            self._load_model()
            if not self.diarizer:
                logger.error("Failed to load diarization model")
                return None
        
        logger.info(f"Starting speaker diarization for {audio_path}")
        start_time = time.time()
        
        try:
            with timeout(self.timeout_seconds, "Diarization timed out"):
                # Run diarization
                diarization = self.diarizer(audio_path)
                
                # Process the diarization results
                segments = []
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    segments.append({
                        "start": turn.start,
                        "end": turn.end,
                        "speaker": speaker
                    })
                
                # Sort segments by start time
                segments.sort(key=lambda x: x["start"])
                
                elapsed = time.time() - start_time
                logger.info(f"Diarization completed in {elapsed:.1f} seconds, found {len(segments)} segments")
                
                # Cache the results if caching is enabled
                if self.cache_manager:
                    self.cache_manager.cache_diarization(audio_path, segments)
                
                return segments
                
        except TimeoutException:
            logger.error(f"Diarization timed out after {self.timeout_seconds} seconds")
            raise
        except Exception as e:
            logger.error(f"Error during diarization: {str(e)}")
            raise Exception(f"Error during diarization: {str(e)}")
    
    def diarize_with_progress(self, audio_path: str) -> Optional[List[Dict[str, Any]]]:
        """Perform speaker diarization with progress reporting.
        
        This method runs diarization in a separate thread and reports progress.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            List of diarization segments or None if diarization is disabled
        """
        if not self.include_diarization:
            logger.info("Diarization is disabled, skipping")
            return None
            
        logger.info("Starting speaker diarization with progress reporting")
        
        # Create a future to store the result
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.diarize, audio_path)
            
            # Wait for the result with progress reporting
            while not future.done():
                logger.info("Speaker diarization in progress...")
                time.sleep(5)
            
            # Get the result or raise the exception
            return future.result() 