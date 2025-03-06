"""
Utilities for monitoring system resources and managing worker pools.
"""

import os
import time
import logging
import threading
import multiprocessing
from typing import Optional, Dict, Any, Callable, List
import concurrent.futures
import psutil

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

class ResourceMonitor:
    """Monitor system resources like CPU, memory, and GPU usage."""
    
    def __init__(self, interval: float = 1.0):
        """Initialize the resource monitor.
        
        Args:
            interval: Monitoring interval in seconds
        """
        self.interval = interval
        self._stop_event = threading.Event()
        self._monitor_thread = None
        self.cpu_percent = 0.0
        self.memory_percent = 0.0
        self.gpu_memory_percent = 0.0
        self.gpu_utilization = 0.0
        self.history: List[Dict[str, float]] = []
        self.max_history_size = 60  # Keep last 60 readings (1 minute at 1s interval)
    
    def start(self):
        """Start monitoring resources in a background thread."""
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            logger.warning("Resource monitor is already running")
            return
            
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
        logger.debug("Resource monitor started")
    
    def stop(self):
        """Stop the monitoring thread."""
        if self._monitor_thread is None or not self._monitor_thread.is_alive():
            return
            
        self._stop_event.set()
        self._monitor_thread.join(timeout=2.0)
        logger.debug("Resource monitor stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while not self._stop_event.is_set():
            self._update_metrics()
            self._stop_event.wait(self.interval)
    
    def _update_metrics(self):
        """Update resource metrics."""
        # CPU usage
        self.cpu_percent = psutil.cpu_percent(interval=None)
        
        # Memory usage
        memory = psutil.virtual_memory()
        self.memory_percent = memory.percent
        
        # GPU metrics if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                # Get current device
                device = torch.cuda.current_device()
                
                # Get GPU memory usage
                memory_allocated = torch.cuda.memory_allocated(device)
                memory_reserved = torch.cuda.memory_reserved(device)
                memory_total = torch.cuda.get_device_properties(device).total_memory
                
                self.gpu_memory_percent = 100.0 * memory_allocated / memory_total
                
                # Try to get GPU utilization using nvidia-smi
                try:
                    import subprocess
                    result = subprocess.run(
                        ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                        stdout=subprocess.PIPE,
                        text=True
                    )
                    self.gpu_utilization = float(result.stdout.strip())
                except (subprocess.SubprocessError, ValueError):
                    self.gpu_utilization = 0.0
            except Exception as e:
                logger.debug(f"Error getting GPU metrics: {e}")
                self.gpu_memory_percent = 0.0
                self.gpu_utilization = 0.0
        else:
            self.gpu_memory_percent = 0.0
            self.gpu_utilization = 0.0
        
        # Add to history
        metrics = {
            'timestamp': time.time(),
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'gpu_memory_percent': self.gpu_memory_percent,
            'gpu_utilization': self.gpu_utilization
        }
        
        self.history.append(metrics)
        
        # Trim history if needed
        if len(self.history) > self.max_history_size:
            self.history = self.history[-self.max_history_size:]
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current resource metrics.
        
        Returns:
            Dictionary of resource metrics
        """
        # Update metrics if not running in background
        if self._monitor_thread is None or not self._monitor_thread.is_alive():
            self._update_metrics()
            
        return {
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'gpu_memory_percent': self.gpu_memory_percent,
            'gpu_utilization': self.gpu_utilization
        }
    
    def get_average_metrics(self, seconds: int = 5) -> Dict[str, float]:
        """Get average metrics over the specified time period.
        
        Args:
            seconds: Number of seconds to average over
            
        Returns:
            Dictionary of average resource metrics
        """
        if not self.history:
            return self.get_metrics()
            
        # Calculate how many samples to use
        samples = min(seconds, len(self.history))
        recent_history = self.history[-samples:]
        
        # Calculate averages
        avg_cpu = sum(m['cpu_percent'] for m in recent_history) / samples
        avg_memory = sum(m['memory_percent'] for m in recent_history) / samples
        avg_gpu_memory = sum(m['gpu_memory_percent'] for m in recent_history) / samples
        avg_gpu_util = sum(m['gpu_utilization'] for m in recent_history) / samples
        
        return {
            'cpu_percent': avg_cpu,
            'memory_percent': avg_memory,
            'gpu_memory_percent': avg_gpu_memory,
            'gpu_utilization': avg_gpu_util
        }
    
    def __enter__(self):
        """Start monitoring when used as a context manager."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop monitoring when exiting context."""
        self.stop()


class AdaptiveWorkerPool:
    """Worker pool that adapts to system resource availability."""
    
    def __init__(
        self, 
        min_workers: int = 1, 
        max_workers: Optional[int] = None,
        cpu_threshold: float = 85.0,
        memory_threshold: float = 85.0,
        gpu_threshold: float = 85.0,
        adjustment_interval: float = 5.0
    ):
        """Initialize the adaptive worker pool.
        
        Args:
            min_workers: Minimum number of workers
            max_workers: Maximum number of workers (default: CPU count)
            cpu_threshold: CPU usage threshold percentage to reduce workers
            memory_threshold: Memory usage threshold percentage to reduce workers
            gpu_threshold: GPU usage threshold percentage to reduce workers
            adjustment_interval: Seconds between worker count adjustments
        """
        self.min_workers = max(1, min_workers)
        
        if max_workers is None:
            self.max_workers = multiprocessing.cpu_count()
        else:
            self.max_workers = max(self.min_workers, max_workers)
            
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.gpu_threshold = gpu_threshold
        self.adjustment_interval = adjustment_interval
        
        self.current_workers = self.max_workers
        self.resource_monitor = ResourceMonitor(interval=1.0)
        self._executor = None
        self._stop_event = threading.Event()
        self._adjustment_thread = None
        
        logger.debug(f"Adaptive worker pool initialized with {self.min_workers}-{self.max_workers} workers")
    
    def start(self):
        """Start the worker pool and resource monitoring."""
        if self._executor is not None:
            logger.warning("Worker pool is already running")
            return
            
        # Start with max workers
        self.current_workers = self.max_workers
        self._executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.current_workers)
        
        # Start resource monitoring
        self.resource_monitor.start()
        
        # Start adjustment thread
        self._stop_event.clear()
        self._adjustment_thread = threading.Thread(target=self._adjustment_loop)
        self._adjustment_thread.daemon = True
        self._adjustment_thread.start()
        
        logger.info(f"Started adaptive worker pool with {self.current_workers} workers")
    
    def stop(self):
        """Stop the worker pool and resource monitoring."""
        if self._executor is None:
            return
            
        # Stop adjustment thread
        self._stop_event.set()
        if self._adjustment_thread and self._adjustment_thread.is_alive():
            self._adjustment_thread.join(timeout=2.0)
        
        # Stop resource monitoring
        self.resource_monitor.stop()
        
        # Shutdown executor
        self._executor.shutdown(wait=True)
        self._executor = None
        
        logger.info("Stopped adaptive worker pool")
    
    def _adjustment_loop(self):
        """Periodically adjust worker count based on resource usage."""
        while not self._stop_event.is_set():
            self._adjust_workers()
            self._stop_event.wait(self.adjustment_interval)
    
    def _adjust_workers(self):
        """Adjust the number of workers based on resource usage."""
        if self._executor is None:
            return
            
        # Get average metrics
        metrics = self.resource_monitor.get_average_metrics(seconds=int(self.adjustment_interval))
        
        # Determine if we need to adjust
        cpu_overloaded = metrics['cpu_percent'] > self.cpu_threshold
        memory_overloaded = metrics['memory_percent'] > self.memory_threshold
        gpu_overloaded = (metrics['gpu_memory_percent'] > self.gpu_threshold or 
                         metrics['gpu_utilization'] > self.gpu_threshold)
        
        # Calculate new worker count
        new_workers = self.current_workers
        
        if cpu_overloaded or memory_overloaded or gpu_overloaded:
            # Reduce workers if overloaded
            new_workers = max(self.min_workers, self.current_workers - 1)
            reason = []
            if cpu_overloaded:
                reason.append(f"CPU at {metrics['cpu_percent']:.1f}%")
            if memory_overloaded:
                reason.append(f"Memory at {metrics['memory_percent']:.1f}%")
            if gpu_overloaded:
                reason.append(f"GPU at {metrics['gpu_utilization']:.1f}%/{metrics['gpu_memory_percent']:.1f}%")
                
            if new_workers < self.current_workers:
                logger.info(f"Reducing workers to {new_workers} due to: {', '.join(reason)}")
        elif (metrics['cpu_percent'] < self.cpu_threshold * 0.7 and 
              metrics['memory_percent'] < self.memory_threshold * 0.7 and
              metrics['gpu_memory_percent'] < self.gpu_threshold * 0.7):
            # Increase workers if resources are available
            new_workers = min(self.max_workers, self.current_workers + 1)
            if new_workers > self.current_workers:
                logger.info(f"Increasing workers to {new_workers} due to available resources")
        
        # Apply the change if needed
        if new_workers != self.current_workers:
            self.current_workers = new_workers
            
            # We can't resize an existing executor, so we need to create a new one
            old_executor = self._executor
            self._executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.current_workers)
            
            # Shutdown the old executor without waiting
            # This allows existing tasks to complete but doesn't accept new ones
            old_executor.shutdown(wait=False)
    
    def submit(self, fn, *args, **kwargs):
        """Submit a task to the worker pool.
        
        Args:
            fn: Function to execute
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Future object representing the execution of the callable
        """
        if self._executor is None:
            raise RuntimeError("Worker pool is not running")
            
        return self._executor.submit(fn, *args, **kwargs)
    
    def map(self, fn, *iterables, timeout=None, chunksize=1):
        """Map a function to an iterable of arguments.
        
        Args:
            fn: Function to execute
            *iterables: Iterables of arguments to pass to the function
            timeout: Maximum number of seconds to wait for results
            chunksize: Size of chunks to submit to the pool
            
        Returns:
            Iterator of results
        """
        if self._executor is None:
            raise RuntimeError("Worker pool is not running")
            
        return self._executor.map(fn, *iterables, timeout=timeout, chunksize=chunksize)
    
    def __enter__(self):
        """Start the worker pool when used as a context manager."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the worker pool when exiting context."""
        self.stop()


def get_optimal_worker_count(
    min_workers: int = 1, 
    max_workers: Optional[int] = None,
    reserve_memory_gb: float = 2.0,
    reserve_cpu_percent: float = 20.0
) -> int:
    """Calculate the optimal number of worker processes based on system resources.
    
    Args:
        min_workers: Minimum number of workers
        max_workers: Maximum number of workers (default: CPU count)
        reserve_memory_gb: Amount of memory to reserve in GB
        reserve_cpu_percent: Percentage of CPU to reserve
        
    Returns:
        Optimal number of worker processes
    """
    # Get CPU count
    cpu_count = multiprocessing.cpu_count()
    
    # Set default max_workers if not provided
    if max_workers is None:
        max_workers = cpu_count
    
    # Calculate CPU-based worker count
    cpu_based = max(1, int(cpu_count * (100 - reserve_cpu_percent) / 100))
    
    # Calculate memory-based worker count
    memory = psutil.virtual_memory()
    total_memory_gb = memory.total / (1024 ** 3)
    available_memory_gb = total_memory_gb - reserve_memory_gb
    
    # Estimate memory per worker (assume 2GB per worker by default)
    memory_per_worker_gb = 2.0
    memory_based = max(1, int(available_memory_gb / memory_per_worker_gb))
    
    # Take the minimum of CPU and memory based counts
    optimal = min(cpu_based, memory_based)
    
    # Clamp to min/max range
    optimal = max(min_workers, min(optimal, max_workers))
    
    logger.debug(f"Optimal worker count: {optimal} (CPU: {cpu_based}, Memory: {memory_based})")
    return optimal 