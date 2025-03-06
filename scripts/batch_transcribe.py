#!/usr/bin/env python3
"""
Script to batch process multiple audio or video files for transcription.

This script allows processing multiple files in parallel, with configurable
worker count and support for both regular and streaming transcription modes.
"""

import os
import sys
import time
import glob
import argparse
import logging
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from tqdm import tqdm

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import Config
from src.transcriber import Transcriber
from src.utils.resource_monitor import AdaptiveWorkerPool, get_optimal_worker_count

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def process_file(
    input_path: str, 
    config: Config, 
    output_dir: str, 
    use_streaming: bool = False,
    output_format: Optional[str] = None
) -> Tuple[str, bool, float]:
    """
    Process a single file for transcription.
    
    Args:
        input_path: Path to the input file
        config: Configuration object
        output_dir: Directory to save output files
        use_streaming: Whether to use streaming transcription
        output_format: Output format override
        
    Returns:
        Tuple of (input_path, success, processing_time)
    """
    start_time = time.time()
    file_name = os.path.basename(input_path)
    
    try:
        # Override output format if specified
        original_format = config._output_format
        if output_format:
            config._output_format = output_format
        
        # Create transcriber
        transcriber = Transcriber(config)
        
        # Generate output path
        base_name = os.path.splitext(file_name)[0]
        output_path = os.path.join(output_dir, f"{base_name}.{config._output_format}")
        
        # Process the file
        if use_streaming:
            # For streaming mode, we need to collect segments and save them
            segments = []
            
            if config.include_diarization:
                generator = transcriber.transcribe_stream_with_diarization(input_path)
            else:
                generator = transcriber.transcribe_stream(input_path)
                
            for segment in generator:
                # Convert segment dict to tuple format expected by save_transcript
                if "speaker" in segment and segment["speaker"]:
                    segments.append((segment["start"], segment["end"], segment["text"], segment["speaker"]))
                else:
                    segments.append((segment["start"], segment["end"], segment["text"], "SPEAKER"))
        else:
            # Regular transcription
            segments = transcriber.transcribe(input_path)
        
        # Save the transcript
        transcriber.save_transcript(segments, output_path)
        
        # Restore original format
        if output_format:
            config._output_format = original_format
            
        elapsed = time.time() - start_time
        return (input_path, True, elapsed)
        
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Error processing {file_name}: {str(e)}")
        return (input_path, False, elapsed)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Batch process multiple audio or video files for transcription"
    )
    parser.add_argument("input_pattern", help="Glob pattern for input files (e.g., 'videos/*.mp4')")
    parser.add_argument("--output-dir", "-o", default="transcripts", 
                       help="Output directory for transcripts (default: transcripts)")
    parser.add_argument("--workers", "-w", type=int, default=0,
                       help="Number of worker processes (0 for auto-detection, default: 0)")
    parser.add_argument("--min-workers", type=int, default=1,
                       help="Minimum number of worker processes (default: 1)")
    parser.add_argument("--max-workers", type=int, default=None,
                       help="Maximum number of worker processes (default: CPU count)")
    parser.add_argument("--adaptive", "-a", action="store_true",
                       help="Use adaptive worker pool that adjusts based on system load")
    parser.add_argument("--diarize", "-d", action="store_true", 
                       help="Include speaker diarization")
    parser.add_argument("--model", "-m", 
                       help="Whisper model size (tiny, base, small, medium, large)")
    parser.add_argument("--language", "-l", 
                       help="Language code (e.g., en, fr, de)")
    parser.add_argument("--format", "-f", choices=["txt", "srt", "vtt", "json"], 
                       help="Output format (default: from config)")
    parser.add_argument("--streaming", "-s", action="store_true",
                       help="Use streaming transcription (reduces memory usage)")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        # Reduce logging noise during batch processing
        logging.getLogger().setLevel(logging.WARNING)
        logger.setLevel(logging.INFO)
    
    # Find input files
    input_files = glob.glob(args.input_pattern, recursive=True)
    if not input_files:
        logger.error(f"No files found matching pattern: {args.input_pattern}")
        return 1
    
    logger.info(f"Found {len(input_files)} files to process")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create configuration
    config = Config(".env")
    
    # Override config with command line arguments
    if args.model:
        config.whisper_model_size = args.model
    if args.language:
        config.language = args.language
    if args.diarize:
        config.include_diarization = True
    
    # Validate the configuration
    if not config.validate():
        logger.error("Invalid configuration. Please check your settings.")
        return 1
    
    # Determine worker count
    if args.workers <= 0:
        # Auto-detect optimal worker count
        worker_count = get_optimal_worker_count(
            min_workers=args.min_workers,
            max_workers=args.max_workers,
            reserve_memory_gb=4.0,  # Reserve more memory for transcription
            reserve_cpu_percent=30.0  # Reserve more CPU for system
        )
    else:
        worker_count = args.workers
    
    # Display configuration
    logger.info(f"Processing with model: {config.whisper_model_size}, language: {config.language}")
    logger.info(f"Diarization: {'Enabled' if config.include_diarization else 'Disabled'}")
    logger.info(f"Mode: {'Streaming' if args.streaming else 'Regular'}")
    
    if args.adaptive:
        logger.info(f"Using adaptive worker pool (min={args.min_workers}, max={args.max_workers or 'CPU count'})")
    else:
        logger.info(f"Using fixed worker count: {worker_count}")
    
    # Process files in parallel
    results = []
    successful = 0
    failed = 0
    total_time = 0.0
    
    # Start timing
    batch_start_time = time.time()
    
    if args.adaptive:
        # Use adaptive worker pool
        with AdaptiveWorkerPool(
            min_workers=args.min_workers,
            max_workers=args.max_workers,
            cpu_threshold=80.0,
            memory_threshold=80.0,
            gpu_threshold=80.0,
            adjustment_interval=10.0
        ) as pool:
            # Submit all tasks
            futures = []
            for input_path in input_files:
                futures.append(pool.submit(
                    process_file, 
                    input_path, 
                    config, 
                    args.output_dir, 
                    args.streaming,
                    args.format
                ))
            
            # Process results as they complete with progress bar
            with tqdm(total=len(input_files), desc="Processing files") as progress:
                for future in concurrent.futures.as_completed(futures):
                    input_path, success, elapsed = future.result()
                    file_name = os.path.basename(input_path)
                    
                    if success:
                        successful += 1
                        logger.info(f"Successfully processed {file_name} in {elapsed:.1f} seconds")
                    else:
                        failed += 1
                        logger.error(f"Failed to process {file_name} after {elapsed:.1f} seconds")
                    
                    total_time += elapsed
                    results.append((input_path, success, elapsed))
                    progress.update(1)
    else:
        # Use fixed worker pool
        with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(
                    process_file, 
                    input_path, 
                    config, 
                    args.output_dir, 
                    args.streaming,
                    args.format
                ): input_path for input_path in input_files
            }
            
            # Process results as they complete with progress bar
            with tqdm(total=len(input_files), desc="Processing files") as progress:
                for future in concurrent.futures.as_completed(future_to_file):
                    input_path, success, elapsed = future.result()
                    file_name = os.path.basename(input_path)
                    
                    if success:
                        successful += 1
                        logger.info(f"Successfully processed {file_name} in {elapsed:.1f} seconds")
                    else:
                        failed += 1
                        logger.error(f"Failed to process {file_name} after {elapsed:.1f} seconds")
                    
                    total_time += elapsed
                    results.append((input_path, success, elapsed))
                    progress.update(1)
    
    # Calculate total batch processing time
    batch_elapsed = time.time() - batch_start_time
    
    # Display summary
    logger.info("\n=== Batch Processing Summary ===")
    logger.info(f"Total files processed: {len(input_files)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total processing time (sum): {total_time:.1f} seconds")
    logger.info(f"Actual elapsed time: {batch_elapsed:.1f} seconds")
    logger.info(f"Parallelization efficiency: {(total_time / batch_elapsed) if batch_elapsed > 0 else 0:.1f}x")
    if successful > 0:
        logger.info(f"Average time per file: {total_time/successful:.1f} seconds")
    logger.info("===============================")
    
    # Write report to file
    report_path = os.path.join(args.output_dir, "batch_report.txt")
    with open(report_path, "w") as f:
        f.write("=== Batch Processing Report ===\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {config.whisper_model_size}\n")
        f.write(f"Language: {config.language}\n")
        f.write(f"Diarization: {'Enabled' if config.include_diarization else 'Disabled'}\n")
        f.write(f"Mode: {'Streaming' if args.streaming else 'Regular'}\n")
        f.write(f"Worker mode: {'Adaptive' if args.adaptive else 'Fixed'}\n")
        f.write(f"Worker count: {worker_count if not args.adaptive else 'Adaptive'}\n\n")
        
        f.write(f"Total files: {len(input_files)}\n")
        f.write(f"Successful: {successful}\n")
        f.write(f"Failed: {failed}\n")
        f.write(f"Total processing time (sum): {total_time:.1f} seconds\n")
        f.write(f"Actual elapsed time: {batch_elapsed:.1f} seconds\n")
        f.write(f"Parallelization efficiency: {(total_time / batch_elapsed) if batch_elapsed > 0 else 0:.1f}x\n")
        if successful > 0:
            f.write(f"Average time per file: {total_time/successful:.1f} seconds\n\n")
        
        f.write("=== File Details ===\n")
        for input_path, success, elapsed in sorted(results, key=lambda x: x[0]):
            status = "SUCCESS" if success else "FAILED"
            f.write(f"{status}: {input_path} ({elapsed:.1f}s)\n")
    
    logger.info(f"Detailed report saved to {report_path}")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
