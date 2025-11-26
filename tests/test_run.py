"""
Unit tests for run management.
"""

import pytest
from pathlib import Path
import tempfile
import shutil

from experiment_tracker.core.run import Run, RunManager


class TestRun:
    """Tests for Run class."""
    
    def test_create_run(self):
        """Test creating a run."""
        run = Run(
            project_name="test_project",
            experiment_name="test_exp",
            run_name="test_run"
        )
        
        assert run.project_name == "test_project"
        assert run.experiment_name == "test_exp"
        assert run.run_name == "test_run"
        assert run.run_id.startswith("run_")
        assert run.status == "running"
    
    def test_log_param(self):
        """Test logging a parameter."""
        run = Run(project_name="test")
        
        run.log_param("learning_rate", 0.001)
        
        assert "learning_rate" in run.params
        assert run.params["learning_rate"] == 0.001
    
    def test_log_params(self):
        """Test logging multiple parameters."""
        run = Run(project_name="test")
        
        params = {"lr": 0.001, "batch_size": 32}
        run.log_params(params)
        
        assert run.params["lr"] == 0.001
        assert run.params["batch_size"] == 32
    
    def test_log_metric(self):
        """Test logging a metric."""
        run = Run(project_name="test")
        
        run.log_metric("loss", 0.5, step=1)
        
        assert "loss" in run.metrics
        assert len(run.metrics["loss"]) == 1
        assert run.metrics["loss"][0]["value"] == 0.5
        assert run.metrics["loss"][0]["step"] == 1
    
    def test_log_metrics(self):
        """Test logging multiple metrics."""
        run = Run(project_name="test")
        
        metrics = {"loss": 0.5, "accuracy": 0.92}
        run.log_metrics(metrics, step=1)
        
        assert "loss" in run.metrics
        assert "accuracy" in run.metrics
    
    def test_set_tag(self):
        """Test setting a tag."""
        run = Run(project_name="test")
        
        run.set_tag("production")
        
        assert "production" in run.tags
    
    def test_set_tags(self):
        """Test setting multiple tags."""
        run = Run(project_name="test")
        
        run.set_tags(["production", "baseline"])
        
        assert "production" in run.tags
        assert "baseline" in run.tags
    
    def test_finish_run(self):
        """Test finishing a run."""
        run = Run(project_name="test")
        
        run.finish(status="completed")
        
        assert run.status == "completed"
        assert run.finished_at is not None
        assert run.duration is not None
    
    def test_to_dict(self):
        """Test converting run to dictionary."""
        run = Run(project_name="test")
        run.log_param("lr", 0.001)
        run.log_metric("loss", 0.5)
        
        run_dict = run.to_dict()
        
        assert isinstance(run_dict, dict)
        assert run_dict["project_name"] == "test"
        assert "params" in run_dict
        assert "metrics" in run_dict


class TestRunManager:
    """Tests for RunManager class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_create_run_manager(self, temp_dir):
        """Test creating a run manager."""
        manager = RunManager(storage_dir=temp_dir)
        
        assert manager.storage_dir == Path(temp_dir)
        assert manager.storage_dir.exists()
    
    def test_create_run(self, temp_dir):
        """Test creating a run through manager."""
        manager = RunManager(storage_dir=temp_dir)
        
        run = manager.create_run(
            project_name="test_project",
            experiment_name="test_exp"
        )
        
        assert isinstance(run, Run)
        assert run.project_name == "test_project"
        assert run.run_dir is not None
        assert Path(run.run_dir).exists()
    
    def test_save_run(self, temp_dir):
        """Test saving a run."""
        manager = RunManager(storage_dir=temp_dir)
        run = manager.create_run(project_name="test")
        
        run.log_param("lr", 0.001)
        manager.save_run(run)
        
        # Verify run.json exists
        run_json = Path(run.run_dir) / "run.json"
        assert run_json.exists()
    
    def test_load_run(self, temp_dir):
        """Test loading a run."""
        manager = RunManager(storage_dir=temp_dir)
        
        # Create and save run
        run = manager.create_run(project_name="test")
        run.log_param("lr", 0.001)
        manager.save_run(run)
        
        # Load run
        loaded_run = manager.load_run(run.run_id)
        
        assert loaded_run is not None
        assert loaded_run.run_id == run.run_id
        assert loaded_run.params["lr"] == 0.001
    
    def test_list_runs(self, temp_dir):
        """Test listing runs."""
        manager = RunManager(storage_dir=temp_dir)
        
        # Create multiple runs
        run1 = manager.create_run(project_name="test", run_name="run1")
        run2 = manager.create_run(project_name="test", run_name="run2")
        
        manager.save_run(run1)
        manager.save_run(run2)
        
        # List runs
        runs = manager.list_runs(project_name="test")
        
        assert len(runs) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])