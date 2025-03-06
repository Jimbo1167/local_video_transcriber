import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

class Config:
    """Configuration class to handle all settings for the video transcriber.
    
    This class loads configuration from environment variables and provides
    validation and type conversion for the settings.
    """
    def __init__(self, env_file: Optional[str] = None):
        """Initialize configuration from environment variables.
        
        Args:
            env_file: Optional path to a .env file to load
        """
        if env_file:
            logger.info(f"Loading configuration from {env_file}")
            load_dotenv(env_file, override=True)
        else:
            logger.info("Using existing environment variables for configuration")
        
        # API tokens and model settings
        self.hf_token = os.getenv("HF_TOKEN")
        self.whisper_model_size = os.getenv("WHISPER_MODEL", "base")
        self.diarization_model = os.getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization@2.1")
        self.language = os.getenv("LANGUAGE", "en")
        
        # Parse output format
        self._output_format = None
        output_format = os.getenv("OUTPUT_FORMAT")
        if output_format:
            output_format = output_format.strip()
            if "#" in output_format:
                output_format = output_format.split("#")[0].strip()
        self._output_format = output_format if output_format else "txt"
        
        # Parse diarization setting
        diarization = os.getenv("INCLUDE_DIARIZATION", "true")
        self.include_diarization = diarization.strip().lower() in ["true", "1", "yes", "on"]
        
        # Timeouts
        self.audio_timeout = int(os.getenv("AUDIO_TIMEOUT", "300"))
        self.transcribe_timeout = int(os.getenv("TRANSCRIBE_TIMEOUT", "3600"))
        self.diarize_timeout = int(os.getenv("DIARIZE_TIMEOUT", "3600"))
        
        # Device settings
        self.force_cpu = os.getenv("FORCE_CPU", "false").strip().lower() in ["true", "1", "yes", "on"]
        
        # Cache settings
        cache_enabled = os.getenv("CACHE_ENABLED", "true")
        self.cache_enabled = cache_enabled.strip().lower() in ["true", "1", "yes", "on"]
        
        # Cache expiration in seconds (default: 7 days)
        self.cache_expiration = int(os.getenv("CACHE_EXPIRATION", str(7 * 24 * 60 * 60)))
        
        # Maximum cache size in bytes (default: 10GB)
        self.max_cache_size = int(os.getenv("MAX_CACHE_SIZE", str(10 * 1024 * 1024 * 1024)))
        
        logger.debug(f"Configuration loaded: {self.to_dict()}")

    @property
    def output_format(self) -> str:
        """Get the output format for transcripts."""
        return self._output_format

    @output_format.setter
    def output_format(self, value: str):
        """Set the output format for transcripts.
        
        Args:
            value: The output format (txt, srt, vtt, json)
        """
        if value:
            value = value.strip()
            if "#" in value:
                value = value.split("#")[0].strip()
        self._output_format = value if value else "txt"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary.
        
        Returns:
            Dict containing all configuration values
        """
        return {
            "whisper_model_size": self.whisper_model_size,
            "diarization_model": self.diarization_model,
            "language": self.language,
            "output_format": self.output_format,
            "include_diarization": self.include_diarization,
            "audio_timeout": self.audio_timeout,
            "transcribe_timeout": self.transcribe_timeout,
            "diarize_timeout": self.diarize_timeout,
            "force_cpu": self.force_cpu,
            "cache_enabled": self.cache_enabled,
            "cache_expiration": self.cache_expiration,
            "max_cache_size": self.max_cache_size
        }
    
    def validate(self) -> bool:
        """Validate the configuration.
        
        Returns:
            True if the configuration is valid, False otherwise
        """
        # Check if HF token is provided when diarization is enabled
        if self.include_diarization and not self.hf_token:
            logger.warning("Speaker diarization is enabled but HF_TOKEN is not set")
            return False
        
        # Check if output format is valid
        valid_formats = ["txt", "srt", "vtt", "json"]
        if self.output_format not in valid_formats:
            logger.warning(f"Invalid output format: {self.output_format}. Must be one of {valid_formats}")
            return False
        
        return True 