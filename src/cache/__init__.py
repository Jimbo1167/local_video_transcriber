"""
Cache module for the video transcriber.

This module provides caching functionality for audio files, transcription results,
and diarization results to improve performance and reduce redundant processing.
"""

import logging

logger = logging.getLogger(__name__) 