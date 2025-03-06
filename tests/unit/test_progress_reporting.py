import os
import sys
import time
import pytest
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import the src package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.progress import ProgressReporter, MultiProgressReporter, create_callback_progress


class TestProgressReporter:
    """Test the ProgressReporter class."""
    
    def test_init(self):
        """Test initialization of ProgressReporter."""
        progress = ProgressReporter(total=100, desc="Test", unit="it")
        assert progress.total == 100
        assert progress.desc == "Test"
        assert progress.unit == "it"
        assert progress.completed == 0
    
    def test_update(self):
        """Test updating progress."""
        progress = ProgressReporter(total=100, desc="Test", unit="it")
        progress.update(10)
        assert progress.completed == 10
        progress.update(20)
        assert progress.completed == 30
    
    def test_context_manager(self):
        """Test using ProgressReporter as a context manager."""
        with ProgressReporter(total=100, desc="Test", unit="it") as progress:
            assert progress.total == 100
            progress.update(50)
            assert progress.completed == 50
    
    def test_set_description(self):
        """Test setting description."""
        progress = ProgressReporter(total=100, desc="Test", unit="it")
        progress.set_description("New description")
        assert progress.desc == "New description"
    
    def test_set_postfix(self):
        """Test setting postfix."""
        progress = ProgressReporter(total=100, desc="Test", unit="it")
        # Create a progress bar to test set_postfix
        progress.start()
        progress.set_postfix(status="Running", value=42)
        # Just verify it doesn't raise an exception
        assert progress.progress_bar is not None
        progress.close()
    
    def test_add_checkpoint(self):
        """Test adding checkpoints."""
        progress = ProgressReporter(total=100, desc="Test", unit="it")
        progress.add_checkpoint("start")
        progress.update(50)
        progress.add_checkpoint("middle")
        progress.update(50)
        progress.add_checkpoint("end")
        
        checkpoints = progress.checkpoints
        assert len(checkpoints) == 3
        assert checkpoints[0]['name'] == "start"
        assert checkpoints[1]['name'] == "middle"
        assert checkpoints[2]['name'] == "end"
        
        # Check that timestamps are increasing
        assert checkpoints[0]['time'] <= checkpoints[1]['time']
        assert checkpoints[1]['time'] <= checkpoints[2]['time']
    
    def test_time_estimation(self):
        """Test time estimation."""
        progress = ProgressReporter(total=100, desc="Test", unit="it")
        progress.start()
        progress.update(25)  # 25% complete
        
        # Since we just started, the time remaining should be a reasonable value
        remaining = progress.get_estimated_time_remaining()
        assert remaining is not None
        
        # Test formatted time
        formatted = progress.get_formatted_time_remaining()
        assert isinstance(formatted, str)
        
        progress.close()
    
    def test_elapsed_time(self):
        """Test elapsed time calculation."""
        progress = ProgressReporter(total=100, desc="Test", unit="it")
        progress.start()
        time.sleep(0.1)  # Sleep a bit to ensure elapsed time is non-zero
        
        elapsed = progress.get_elapsed_time()
        assert elapsed > 0
        
        # Test formatted time
        formatted = progress.get_formatted_elapsed_time()
        assert isinstance(formatted, str)
        
        progress.close()
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    def test_resource_usage(self, mock_virtual_memory, mock_cpu_percent):
        """Test resource usage monitoring."""
        # Mock psutil responses
        mock_cpu_percent.return_value = 10.0
        mock_memory = MagicMock()
        mock_memory.percent = 50.0
        mock_virtual_memory.return_value = mock_memory
        
        progress = ProgressReporter(total=100, desc="Test", unit="it", monitor_resources=True)
        progress.start()
        
        # Force an update of metrics
        progress.resource_monitor._update_metrics()
        
        usage = progress.get_resource_usage()
        
        assert "cpu_percent" in usage
        assert usage["cpu_percent"] == 10.0
        assert "memory_percent" in usage
        assert usage["memory_percent"] == 50.0
        
        progress.close()
    
    def test_get_summary(self):
        """Test getting summary information."""
        progress = ProgressReporter(total=100, desc="Test", unit="it")
        progress.start()
        progress.update(50)
        progress.add_checkpoint("middle")
        
        summary = progress.get_summary()
        assert "total" in summary
        assert summary["total"] == 100
        assert "completed" in summary
        assert summary["completed"] == 50
        assert "elapsed" in summary
        assert "checkpoints" in summary
        assert len(summary["checkpoints"]) == 1
        assert summary["checkpoints"][0]["name"] == "middle"
        
        progress.close()


class TestMultiProgressReporter:
    """Test the MultiProgressReporter class."""
    
    def test_init(self):
        """Test initialization of MultiProgressReporter."""
        multi = MultiProgressReporter()
        assert len(multi.reporters) == 0
    
    def test_add_reporter(self):
        """Test adding reporters."""
        multi = MultiProgressReporter()
        reporter1 = multi.add_reporter("task1", total=100, desc="Task 1")
        reporter2 = multi.add_reporter("task2", total=200, desc="Task 2")
        
        assert len(multi.reporters) == 2
        assert "task1" in multi.reporters
        assert "task2" in multi.reporters
        assert reporter1 is multi.reporters["task1"]
        assert reporter2 is multi.reporters["task2"]
        
        multi.close()
    
    def test_update(self):
        """Test updating progress for a specific reporter."""
        multi = MultiProgressReporter()
        multi.add_reporter("task1", total=100, desc="Task 1")
        multi.add_reporter("task2", total=200, desc="Task 2")
        
        multi.update("task1", 30)
        multi.update("task2", 50)
        
        assert multi.reporters["task1"].completed == 30
        assert multi.reporters["task2"].completed == 50
        
        multi.close()
    
    def test_get_reporter(self):
        """Test getting a specific reporter."""
        multi = MultiProgressReporter()
        reporter = multi.add_reporter("task1", total=100, desc="Task 1")
        
        retrieved = multi.get_reporter("task1")
        assert retrieved is reporter
        
        # Test getting a non-existent reporter
        assert multi.get_reporter("non_existent") is None
        
        multi.close()
    
    def test_context_manager(self):
        """Test using MultiProgressReporter as a context manager."""
        with MultiProgressReporter() as multi:
            multi.add_reporter("task1", total=100, desc="Task 1")
            multi.update("task1", 50)
            assert multi.reporters["task1"].completed == 50
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    def test_resource_usage(self, mock_virtual_memory, mock_cpu_percent):
        """Test resource usage monitoring for multiple reporters."""
        # Mock psutil responses
        mock_cpu_percent.return_value = 10.0
        mock_memory = MagicMock()
        mock_memory.percent = 50.0
        mock_virtual_memory.return_value = mock_memory
        
        multi = MultiProgressReporter()
        multi.start()
        
        # Force an update of metrics
        multi.resource_monitor._update_metrics()
        
        usage = multi.get_resource_usage()
        
        assert "cpu_percent" in usage
        assert usage["cpu_percent"] == 10.0
        assert "memory_percent" in usage
        assert usage["memory_percent"] == 50.0
        
        multi.close()
    
    def test_get_summary(self):
        """Test getting summary information for all reporters."""
        multi = MultiProgressReporter()
        multi.add_reporter("task1", total=100, desc="Task 1")
        multi.add_reporter("task2", total=200, desc="Task 2")
        
        multi.update("task1", 30)
        multi.update("task2", 50)
        
        summary = multi.get_summary()
        assert "task1" in summary
        assert "task2" in summary
        assert summary["task1"]["completed"] == 30
        assert summary["task2"]["completed"] == 50
        
        multi.close()


def test_create_callback_progress():
    """Test creating a callback function for progress updates."""
    # Create a mock callback function
    mock_callback = MagicMock()
    
    # Create a progress callback
    progress_callback = create_callback_progress(mock_callback, total=100, desc="Test")
    
    # Use the progress callback
    progress_callback(30)
    
    # Check that the mock callback was called with the right arguments
    mock_callback.assert_called_with(30, 100, None)
    
    # Test with status
    progress_callback(50, "Running")
    mock_callback.assert_called_with(50, 100, "Running") 