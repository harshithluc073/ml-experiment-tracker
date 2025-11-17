"""
Run metadata management for ML Experiment Tracker.

Handles creation, persistence, and querying of experiment run metadata
including parameters, metrics, artifacts, and run continuation detection.
"""

import os
import json
import uuid
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict


@dataclass
class RunMetadata:
    """Metadata for a single experiment run."""
    
    run_id: str
    run_name: str
    project_name: str
    experiment_name: Optional[str]
    status: str  # running, completed, failed, stopped
    start_time: str
    end_time: Optional[str] = None
    duration_seconds: Optional[float] = None
    
    # Git information
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None
    git_dirty: bool = False
    git_remote: Optional[str] = None
    
    # System information
    python_version: Optional[str] = None
    platform: Optional[str] = None
    hostname: Optional[str] = None
    user: Optional[str] = None
    
    # Tracking backend used
    backend: str = "local"
    backend_run_id: Optional[str] = None
    backend_url: Optional[str] = None
    
    # Tags and description
    tags: List[str] = field(default_factory=list)
    description: str = ""
    
    # Parent run (for continuation)
    parent_run_id: Optional[str] = None
    
    # Metrics summary
    metrics_summary: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunMetadata":
        """Create from dictionary."""
        return cls(**data)


class Run:
    """
    Manages a single experiment run with parameters, metrics, and artifacts.
    """
    
    def __init__(
        self,
        project_name: str,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
        run_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        description: str = "",
        parent_run_id: Optional[str] = None,
        storage_dir: str = "./experiment_logs"
    ):
        """
        Initialize a new run.
        
        Args:
            project_name: Name of the project
            experiment_name: Optional experiment grouping
            run_name: Optional custom run name
            run_id: Optional custom run ID (auto-generated if not provided)
            tags: Optional list of tags
            description: Optional run description
            parent_run_id: Optional parent run ID for continuation
            storage_dir: Directory for storing run metadata
        """
        self.run_id = run_id or self._generate_run_id()
        self.run_name = run_name or self._generate_run_name()
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.tags = tags or []
        self.description = description
        self.parent_run_id = parent_run_id
        
        self.storage_dir = Path(storage_dir)
        self.run_dir = self.storage_dir / self.project_name / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize run data
        self.start_time = datetime.now()
        self.end_time = None
        self.status = "running"
        
        self.params: Dict[str, Any] = {}
        self.metrics: Dict[str, List[Dict[str, Any]]] = {}
        self.artifacts: List[Dict[str, str]] = []
        self.system_info: Dict[str, Any] = {}
        self.git_info: Dict[str, Any] = {}
        
        # Backend information
        self.backend = "local"
        self.backend_run_id = None
        self.backend_url = None
        
        # Initialize metadata file
        self._init_metadata()
    
    def _generate_run_id(self) -> str:
        """Generate a unique run ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = str(uuid.uuid4())[:8]
        return f"run_{timestamp}_{random_suffix}"
    
    def _generate_run_name(self) -> str:
        """Generate a human-readable run name."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return f"run_{timestamp}"
    
    def _init_metadata(self) -> None:
        """Initialize or load metadata file."""
        self.metadata_file = self.run_dir / "metadata.json"
        
        if self.metadata_file.exists():
            # Load existing metadata
            with open(self.metadata_file, "r") as f:
                data = json.load(f)
                self.params = data.get("params", {})
                self.metrics = data.get("metrics", {})
                self.artifacts = data.get("artifacts", [])
                self.system_info = data.get("system_info", {})
                self.git_info = data.get("git_info", {})
        else:
            # Create new metadata
            self._save_metadata()
    
    def log_param(self, key: str, value: Any) -> None:
        """Log a single parameter."""
        self.params[key] = value
        self._save_metadata()
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log multiple parameters."""
        self.params.update(params)
        self._save_metadata()
    
    def log_metric(
        self,
        key: str,
        value: float,
        step: Optional[int] = None,
        timestamp: Optional[float] = None
    ) -> None:
        """Log a single metric value."""
        if key not in self.metrics:
            self.metrics[key] = []
        
        metric_entry = {
            "value": value,
            "step": step if step is not None else len(self.metrics[key]),
            "timestamp": timestamp or datetime.now().timestamp()
        }
        
        self.metrics[key].append(metric_entry)
        self._save_metadata()
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ) -> None:
        """Log multiple metrics at once."""
        timestamp = datetime.now().timestamp()
        
        for key, value in metrics.items():
            self.log_metric(key, value, step=step, timestamp=timestamp)
    
    def log_artifact(
        self,
        artifact_path: Union[str, Path],
        artifact_type: str = "file"
    ) -> None:
        """
        Log an artifact (file, model, etc.).
        
        Args:
            artifact_path: Path to the artifact
            artifact_type: Type of artifact (file, model, plot, etc.)
        """
        artifact_path = Path(artifact_path)
        
        artifact_entry = {
            "path": str(artifact_path),
            "name": artifact_path.name,
            "type": artifact_type,
            "size_bytes": artifact_path.stat().st_size if artifact_path.exists() else 0,
            "logged_at": datetime.now().isoformat()
        }
        
        self.artifacts.append(artifact_entry)
        self._save_metadata()
    
    def set_system_info(self, system_info: Dict[str, Any]) -> None:
        """Set system information."""
        self.system_info = system_info
        self._save_metadata()
    
    def set_git_info(self, git_info: Dict[str, Any]) -> None:
        """Set git information."""
        self.git_info = git_info
        self._save_metadata()
    
    def set_backend_info(
        self,
        backend: str,
        backend_run_id: Optional[str] = None,
        backend_url: Optional[str] = None
    ) -> None:
        """Set backend tracking information."""
        self.backend = backend
        self.backend_run_id = backend_run_id
        self.backend_url = backend_url
        self._save_metadata()
    
    def set_tag(self, tag: str) -> None:
        """Add a single tag."""
        if tag not in self.tags:
            self.tags.append(tag)
            self._save_metadata()
    
    def set_tags(self, tags: List[str]) -> None:
        """Set multiple tags."""
        for tag in tags:
            if tag not in self.tags:
                self.tags.append(tag)
        self._save_metadata()
    
    def finish(self, status: str = "completed") -> None:
        """
        Mark run as finished.
        
        Args:
            status: Final status (completed, failed, stopped)
        """
        self.end_time = datetime.now()
        self.status = status
        
        if self.start_time:
            duration = (self.end_time - self.start_time).total_seconds()
        else:
            duration = 0
        
        # Calculate metrics summary
        self.metrics_summary = self._calculate_metrics_summary()
        
        self._save_metadata()
    
    def _calculate_metrics_summary(self) -> Dict[str, Any]:
        """Calculate summary statistics for metrics."""
        summary = {}
        
        for metric_name, values in self.metrics.items():
            if not values:
                continue
            
            metric_values = [v["value"] for v in values]
            
            summary[metric_name] = {
                "final": metric_values[-1],
                "best": min(metric_values) if "loss" in metric_name.lower() or "error" in metric_name.lower() else max(metric_values),
                "mean": sum(metric_values) / len(metric_values),
                "count": len(metric_values)
            }
        
        return summary
    
    def _save_metadata(self) -> None:
        """Save metadata to file."""
        metadata = {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "project_name": self.project_name,
            "experiment_name": self.experiment_name,
            "status": self.status,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "tags": self.tags,
            "description": self.description,
            "parent_run_id": self.parent_run_id,
            "backend": self.backend,
            "backend_run_id": self.backend_run_id,
            "backend_url": self.backend_url,
            "params": self.params,
            "metrics": self.metrics,
            "metrics_summary": self.metrics_summary,
            "artifacts": self.artifacts,
            "system_info": self.system_info,
            "git_info": self.git_info
        }
        
        with open(self.metadata_file, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def get_metadata(self) -> RunMetadata:
        """Get run metadata as a structured object."""
        duration = None
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()
        
        return RunMetadata(
            run_id=self.run_id,
            run_name=self.run_name,
            project_name=self.project_name,
            experiment_name=self.experiment_name,
            status=self.status,
            start_time=self.start_time.isoformat() if self.start_time else None,
            end_time=self.end_time.isoformat() if self.end_time else None,
            duration_seconds=duration,
            git_commit=self.git_info.get("commit"),
            git_branch=self.git_info.get("branch"),
            git_dirty=self.git_info.get("dirty", False),
            git_remote=self.git_info.get("remote"),
            python_version=self.system_info.get("python_version"),
            platform=self.system_info.get("platform"),
            hostname=self.system_info.get("hostname"),
            user=self.system_info.get("user"),
            backend=self.backend,
            backend_run_id=self.backend_run_id,
            backend_url=self.backend_url,
            tags=self.tags,
            description=self.description,
            parent_run_id=self.parent_run_id,
            metrics_summary=self.metrics_summary
        )
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Run(id='{self.run_id}', "
            f"name='{self.run_name}', "
            f"project='{self.project_name}', "
            f"status='{self.status}')"
        )


class RunManager:
    """
    Manages multiple runs and provides querying capabilities.
    """
    
    def __init__(self, storage_dir: str = "./experiment_logs"):
        """
        Initialize run manager.
        
        Args:
            storage_dir: Directory for storing run metadata
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    def create_run(
        self,
        project_name: str,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        description: str = "",
        parent_run_id: Optional[str] = None
    ) -> Run:
        """Create a new run."""
        return Run(
            project_name=project_name,
            experiment_name=experiment_name,
            run_name=run_name,
            tags=tags,
            description=description,
            parent_run_id=parent_run_id,
            storage_dir=str(self.storage_dir)
        )
    
    def get_run(self, project_name: str, run_id: str) -> Optional[Run]:
        """
        Get a run by ID.
        
        Args:
            project_name: Name of the project
            run_id: Run ID to retrieve
            
        Returns:
            Run object if found, None otherwise
        """
        run_dir = self.storage_dir / project_name / run_id
        metadata_file = run_dir / "metadata.json"
        
        if not metadata_file.exists():
            return None
        
        # Load metadata and reconstruct run
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        
        run = Run(
            project_name=project_name,
            run_id=run_id,
            storage_dir=str(self.storage_dir)
        )
        
        # Restore state
        run.run_name = metadata.get("run_name", run.run_name)
        run.experiment_name = metadata.get("experiment_name")
        run.status = metadata.get("status", "unknown")
        run.tags = metadata.get("tags", [])
        run.description = metadata.get("description", "")
        run.parent_run_id = metadata.get("parent_run_id")
        run.params = metadata.get("params", {})
        run.metrics = metadata.get("metrics", {})
        run.artifacts = metadata.get("artifacts", [])
        run.system_info = metadata.get("system_info", {})
        run.git_info = metadata.get("git_info", {})
        
        start_time_str = metadata.get("start_time")
        if start_time_str:
            run.start_time = datetime.fromisoformat(start_time_str)
        
        end_time_str = metadata.get("end_time")
        if end_time_str:
            run.end_time = datetime.fromisoformat(end_time_str)
        
        return run
    
    def list_runs(
        self,
        project_name: str,
        experiment_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        status: Optional[str] = None
    ) -> List[RunMetadata]:
        """
        List runs matching criteria.
        
        Args:
            project_name: Name of the project
            experiment_name: Optional experiment filter
            tags: Optional tag filters (any match)
            status: Optional status filter
            
        Returns:
            List of RunMetadata objects
        """
        project_dir = self.storage_dir / project_name
        
        if not project_dir.exists():
            return []
        
        runs = []
        
        for run_dir in project_dir.iterdir():
            if not run_dir.is_dir():
                continue
            
            metadata_file = run_dir / "metadata.json"
            if not metadata_file.exists():
                continue
            
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            
            # Apply filters
            if experiment_name and metadata.get("experiment_name") != experiment_name:
                continue
            
            if status and metadata.get("status") != status:
                continue
            
            if tags:
                run_tags = metadata.get("tags", [])
                if not any(tag in run_tags for tag in tags):
                    continue
            
            # Create RunMetadata
            run_metadata = RunMetadata(
                run_id=metadata.get("run_id"),
                run_name=metadata.get("run_name"),
                project_name=metadata.get("project_name"),
                experiment_name=metadata.get("experiment_name"),
                status=metadata.get("status"),
                start_time=metadata.get("start_time"),
                end_time=metadata.get("end_time"),
                tags=metadata.get("tags", []),
                description=metadata.get("description", ""),
                parent_run_id=metadata.get("parent_run_id"),
                backend=metadata.get("backend", "local"),
                backend_run_id=metadata.get("backend_run_id"),
                backend_url=metadata.get("backend_url"),
                metrics_summary=metadata.get("metrics_summary", {})
            )
            
            runs.append(run_metadata)
        
        # Sort by start time (newest first)
        runs.sort(key=lambda r: r.start_time or "", reverse=True)
        
        return runs
    
    def find_parent_run(
        self,
        project_name: str,
        experiment_name: Optional[str] = None,
        params_hash: Optional[str] = None
    ) -> Optional[str]:
        """
        Find a potential parent run for continuation.
        
        Args:
            project_name: Name of the project
            experiment_name: Optional experiment filter
            params_hash: Optional parameter hash for matching
            
        Returns:
            Parent run ID if found, None otherwise
        """
        runs = self.list_runs(
            project_name=project_name,
            experiment_name=experiment_name,
            status="completed"
        )
        
        if not runs:
            return None
        
        # Return most recent completed run
        return runs[0].run_id
    
    def delete_run(self, project_name: str, run_id: str) -> bool:
        """
        Delete a run.
        
        Args:
            project_name: Name of the project
            run_id: Run ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        run_dir = self.storage_dir / project_name / run_id
        
        if not run_dir.exists():
            return False
        
        import shutil
        shutil.rmtree(run_dir)
        return True


if __name__ == "__main__":
    # Example usage
    manager = RunManager()
    
    # Create a run
    run = manager.create_run(
        project_name="test_project",
        experiment_name="exp_001",
        tags=["test", "demo"]
    )
    
    # Log parameters
    run.log_params({
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10
    })
    
    # Log metrics
    for epoch in range(10):
        run.log_metrics({
            "loss": 1.0 / (epoch + 1),
            "accuracy": 0.5 + (epoch * 0.05)
        }, step=epoch)
    
    # Finish run
    run.finish()
    
    print(f"Created run: {run}")
    print(f"Metrics summary: {run.metrics_summary}")
    
    # List runs
    runs = manager.list_runs("test_project")
    print(f"\nFound {len(runs)} runs")
    for r in runs:
        print(f"  - {r.run_id}: {r.status}")