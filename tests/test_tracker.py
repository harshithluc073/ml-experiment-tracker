"""
Unit tests for ExperimentTracker.
"""

import pytest
from pathlib import Path
import tempfile
import shutil

from experiment_tracker import ExperimentTracker


class TestExperimentTracker:
    """Tests for ExperimentTracker class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_create_tracker(self):
        """Test creating an experiment tracker."""
        tracker = ExperimentTracker(
            project_name="test_project",
            auto_start=False
        )
        
        assert tracker.project_name == "test_project"
        assert tracker.backend is None
    
    def test_start_tracker(self):
        """Test starting a tracker."""
        tracker = ExperimentTracker(
            project_name="test_project",
            auto_start=False
        )
        
        tracker.start()
        
        assert tracker.backend is not None
        assert tracker.run is not None
        assert tracker._started is True
    
    def test_log_params(self):
        """Test logging parameters."""
        with ExperimentTracker(project_name="test") as tracker:
            tracker.log_params({"lr": 0.001, "batch_size": 32})
            
            assert tracker.run.params["lr"] == 0.001
            assert tracker.run.params["batch_size"] == 32
    
    def test_log_metrics(self):
        """Test logging metrics."""
        with ExperimentTracker(project_name="test") as tracker:
            tracker.log_metrics({"loss": 0.5}, step=1)
            
            assert "loss" in tracker.run.metrics
    
    def test_log_artifact(self, temp_dir):
        """Test logging an artifact."""
        # Create test artifact
        artifact_path = Path(temp_dir) / "test.txt"
        artifact_path.write_text("test content")
        
        with ExperimentTracker(project_name="test") as tracker:
            tracker.log_artifact(artifact_path)
            
            assert len(tracker.run.artifacts) > 0
    
    def test_set_tags(self):
        """Test setting tags."""
        with ExperimentTracker(project_name="test") as tracker:
            tracker.set_tags(["production", "baseline"])
            
            assert "production" in tracker.run.tags
            assert "baseline" in tracker.run.tags
    
    def test_context_manager(self):
        """Test using tracker as context manager."""
        with ExperimentTracker(project_name="test") as tracker:
            tracker.log_params({"lr": 0.001})
            run_id = tracker.get_run_id()
        
        # After exiting context, run should be finished
        assert run_id is not None
    
    def test_get_run_id(self):
        """Test getting run ID."""
        with ExperimentTracker(project_name="test") as tracker:
            run_id = tracker.get_run_id()
            
            assert run_id is not None
            assert run_id.startswith("run_")
    
    def test_get_backend_type(self):
        """Test getting backend type."""
        with ExperimentTracker(project_name="test") as tracker:
            backend_type = tracker.get_backend_type()
            
            assert backend_type in ["local", "mlflow", "wandb"]
    
    def test_backend_priority(self):
        """Test backend priority selection."""
        tracker = ExperimentTracker(
            project_name="test",
            backend_priority=["local"],
            auto_start=False
        )
        
        tracker.start()
        
        assert tracker.get_backend_type() == "local"
    
    def test_track_experiment_function(self):
        """Test track_experiment convenience function."""
        from experiment_tracker import track_experiment
        
        with track_experiment(project_name="test") as tracker:
            assert isinstance(tracker, ExperimentTracker)
            assert tracker.project_name == "test"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])