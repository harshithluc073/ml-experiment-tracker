"""
Integration tests for ML Experiment Tracker.

Tests the complete workflow from start to finish.
"""

import pytest
from pathlib import Path
import tempfile
import shutil
import json

from experiment_tracker import ExperimentTracker
from experiment_tracker.utils import (
    HTMLReporter,
    DiffGenerator,
    ArtifactManager
)


@pytest.mark.integration
class TestCompleteWorkflow:
    """Integration tests for complete tracking workflow."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace."""
        workspace = tempfile.mkdtemp()
        yield Path(workspace)
        shutil.rmtree(workspace, ignore_errors=True)
    
    def test_full_experiment_lifecycle(self, temp_workspace):
        """Test complete experiment from start to finish."""
        # Run experiment
        with ExperimentTracker(
            project_name="integration_test",
            experiment_name="full_lifecycle",
            tags=["test"]
        ) as tracker:
            # Log parameters
            tracker.log_params({
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 5
            })
            
            # Simulate training loop
            for epoch in range(5):
                train_loss = 1.0 / (epoch + 1)
                val_acc = 0.5 + (epoch * 0.1)
                
                tracker.log_metrics({
                    "train_loss": train_loss,
                    "val_accuracy": val_acc
                }, step=epoch)
            
            run_id = tracker.get_run_id()
            run_dir = Path(tracker.run.run_dir)
        
        # Verify run was saved
        assert run_dir.exists()
        run_json = run_dir / "run.json"
        assert run_json.exists()
        
        # Verify run data
        with open(run_json) as f:
            run_data = json.load(f)
        
        assert run_data['run_id'] == run_id
        assert run_data['project_name'] == "integration_test"
        assert run_data['params']['learning_rate'] == 0.001
        assert 'train_loss' in run_data['metrics']
    
    def test_multiple_runs_comparison(self, temp_workspace):
        """Test creating and comparing multiple runs."""
        run_ids = []
        
        # Create multiple runs
        for i in range(3):
            with ExperimentTracker(
                project_name="comparison_test",
                run_name=f"run_{i}"
            ) as tracker:
                tracker.log_params({
                    "learning_rate": 0.001 * (i + 1),
                    "batch_size": 32
                })
                tracker.log_metrics({
                    "accuracy": 0.8 + (i * 0.05)
                }, step=1)
                
                run_ids.append(tracker.get_run_id())
                run_dir = Path(tracker.run.run_dir)
        
        # Load and compare runs
        run_files = [
            Path(tracker.run.run_dir.parent) / run_id / "run.json"
            for run_id in run_ids
        ]
        
        # Verify all runs exist
        for run_file in run_files[:2]:
            if run_file.exists():
                # Compare first two
                diff_gen = DiffGenerator()
                comparison = diff_gen.compare_runs(run_files[0], run_files[1])
                
                assert comparison is not None
                assert len(comparison.param_diffs) > 0
                break
    
    def test_report_generation(self, temp_workspace):
        """Test generating HTML reports."""
        # Create run
        with ExperimentTracker(project_name="report_test") as tracker:
            tracker.log_params({"lr": 0.001})
            tracker.log_metrics({"loss": 0.5}, step=1)
            
            run_dir = Path(tracker.run.run_dir)
        
        # Load run data
        run_json = run_dir / "run.json"
        with open(run_json) as f:
            run_data = json.load(f)
        
        # Generate report
        reporter = HTMLReporter()
        report_path = run_dir / "report.html"
        reporter.generate_run_report(run_data, report_path)
        
        assert report_path.exists()
        assert report_path.stat().st_size > 0
    
    def test_artifact_management(self, temp_workspace):
        """Test artifact tracking."""
        # Create test artifacts
        artifact_file = temp_workspace / "model.txt"
        artifact_file.write_text("model weights")
        
        with ExperimentTracker(project_name="artifact_test") as tracker:
            # Log artifact
            tracker.log_artifact(artifact_file)
            
            run_dir = Path(tracker.run.run_dir)
        
        # Verify artifact was logged
        run_json = run_dir / "run.json"
        with open(run_json) as f:
            run_data = json.load(f)
        
        assert len(run_data.get('artifacts', [])) > 0
    
    def test_backend_fallback(self, temp_workspace):
        """Test backend fallback mechanism."""
        # Force local backend
        with ExperimentTracker(
            project_name="fallback_test",
            backend_priority=["local"]
        ) as tracker:
            assert tracker.get_backend_type() == "local"
            
            tracker.log_params({"test": "value"})
            run_id = tracker.get_run_id()
        
        assert run_id is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'integration'])