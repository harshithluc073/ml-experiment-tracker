"""
Core ExperimentTracker implementation.

Provides a unified, high-level interface for experiment tracking that
automatically selects the best available backend and manages the complete
experiment lifecycle.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

from experiment_tracker.core.config import Config
from experiment_tracker.core.run import Run, RunManager
from experiment_tracker.backends.base import BaseBackend, create_backend
from experiment_tracker.utils.env_detector import detect_environment, EnvironmentDetector


class ExperimentTracker:
    """
    High-level interface for ML experiment tracking.
    
    Automatically selects the best available backend (W&B, MLflow, or Local)
    and provides a simple, unified API for logging experiments.
    
    Examples:
        Basic usage:
            tracker = ExperimentTracker(project_name="my_project")
            tracker.log_params({"lr": 0.001})
            tracker.log_metrics({"loss": 0.5}, step=1)
            tracker.finish()
        
        Context manager:
            with ExperimentTracker(project_name="my_project") as tracker:
                tracker.log_params({"lr": 0.001})
                tracker.log_metrics({"loss": 0.5})
    """
    
    def __init__(
        self,
        project_name: str,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        description: str = "",
        config: Optional[Union[Config, Dict[str, Any]]] = None,
        backend_priority: Optional[List[str]] = None,
        auto_start: bool = True
    ):
        """
        Initialize ExperimentTracker.
        
        Args:
            project_name: Name of the project
            experiment_name: Optional experiment grouping
            run_name: Optional custom run name
            tags: Optional list of tags
            description: Optional run description
            config: Configuration (Config object or dict)
            backend_priority: List of backends to try (e.g., ["wandb", "mlflow", "local"])
            auto_start: Automatically start tracking on initialization
        """
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tags = tags or []
        self.description = description
        
        # Load or create config
        if config is None:
            self.config = Config(project_name=project_name)
        elif isinstance(config, dict):
            config['project_name'] = project_name
            self.config = Config.from_dict(config)
        else:
            self.config = config
        
        # Set backend priority
        if backend_priority is None:
            backend_priority = self.config.backend.priority
        self.backend_priority = backend_priority
        
        # Initialize components
        self.backend: Optional[BaseBackend] = None
        self.run: Optional[Run] = None
        self.run_manager = RunManager(storage_dir=self.config.backend.local_dir)
        self.env_detector: Optional[EnvironmentDetector] = None
        
        # State
        self._started = False
        self._finished = False
        
        # Auto-start if requested
        if auto_start:
            self.start()
    
    def start(self) -> None:
        """Start tracking the experiment."""
        if self._started:
            return
        
        # Detect environment
        if self.config.logging.log_system_info:
            self.env_detector = detect_environment()
        
        # Create backend
        backend_config = self._prepare_backend_config()
        
        self.backend = create_backend(
            priority=self.backend_priority,
            project_name=self.project_name,
            experiment_name=self.experiment_name,
            run_name=self.run_name,
            tags=self.tags,
            description=self.description,
            config=backend_config
        )
        
        # Initialize backend
        if not self.backend.initialize():
            raise RuntimeError("Failed to initialize backend")
        
        # Create local run for metadata
        self.run = self.run_manager.create_run(
            project_name=self.project_name,
            experiment_name=self.experiment_name,
            run_name=self.run_name or self.backend.run_name,
            tags=self.tags,
            description=self.description
        )
        
        # Set backend info in run
        self.run.set_backend_info(
            backend=self.backend.get_backend_type().value,
            backend_run_id=self.backend.get_run_id(),
            backend_url=self.backend.get_run_url()
        )
        
        # Log system info
        if self.env_detector and self.config.logging.log_system_info:
            self._log_system_info()
        
        self._started = True
        
        print(f"✓ Experiment tracking started")
        print(f"  Backend: {type(self.backend).__name__}")
        print(f"  Run ID: {self.backend.get_run_id()}")
        if self.backend.get_run_url():
            print(f"  View: {self.backend.get_run_url()}")
    
    def _prepare_backend_config(self) -> Dict[str, Any]:
        """Prepare backend-specific configuration."""
        backend_config = {
            'storage_dir': self.config.backend.local_dir,
            'tracking_uri': self.config.backend.mlflow_tracking_uri,
            'auto_start_ui': self.config.backend.mlflow_auto_start,
            'entity': self.config.backend.wandb_entity,
            'mode': self.config.backend.wandb_mode,
        }
        return backend_config
    
    def _log_system_info(self) -> None:
        """Log system information to backend and run."""
        if not self.env_detector:
            return
        
        system_info = self.env_detector.system_info
        
        # Log to backend
        self.backend.log_system_info(system_info)
        
        # Log to local run
        self.run.set_system_info(system_info)
        
        # Log git info if available
        if self.config.logging.log_git_info:
            git_info = self._get_git_info()
            if git_info:
                self.run.set_git_info(git_info)
    
    def _get_git_info(self) -> Optional[Dict[str, Any]]:
        """Get git repository information."""
        try:
            import git
            
            # Try to find git repo
            repo = git.Repo(search_parent_directories=True)
            
            git_info = {
                'commit': repo.head.commit.hexsha,
                'branch': repo.active_branch.name,
                'dirty': repo.is_dirty(),
                'remote': repo.remotes.origin.url if repo.remotes else None
            }
            
            return git_info
        
        except Exception:
            return None
    
    def log_param(self, key: str, value: Any) -> None:
        """
        Log a single parameter.
        
        Args:
            key: Parameter name
            value: Parameter value
        """
        self._ensure_started()
        
        self.backend.log_param(key, value)
        self.run.log_param(key, value)
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log multiple parameters.
        
        Args:
            params: Dictionary of parameters
        """
        self._ensure_started()
        
        self.backend.log_params(params)
        self.run.log_params(params)
    
    def log_metric(
        self,
        key: str,
        value: float,
        step: Optional[int] = None,
        timestamp: Optional[float] = None
    ) -> None:
        """
        Log a single metric value.
        
        Args:
            key: Metric name
            value: Metric value
            step: Optional step/epoch number
            timestamp: Optional timestamp
        """
        self._ensure_started()
        
        self.backend.log_metric(key, value, step=step, timestamp=timestamp)
        self.run.log_metric(key, value, step=step, timestamp=timestamp)
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        timestamp: Optional[float] = None
    ) -> None:
        """
        Log multiple metrics at once.
        
        Args:
            metrics: Dictionary of metrics
            step: Optional step/epoch number
            timestamp: Optional timestamp
        """
        self._ensure_started()
        
        self.backend.log_metrics(metrics, step=step, timestamp=timestamp)
        self.run.log_metrics(metrics, step=step)
    
    def log_artifact(
        self,
        artifact_path: Union[str, Path],
        artifact_type: str = "file",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log an artifact (file, model, etc.).
        
        Args:
            artifact_path: Path to the artifact
            artifact_type: Type of artifact
            metadata: Optional metadata
        """
        self._ensure_started()
        
        self.backend.log_artifact(artifact_path, artifact_type, metadata)
        self.run.log_artifact(artifact_path, artifact_type)
    
    def log_model(
        self,
        model: Any,
        artifact_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log a trained model.
        
        Args:
            model: The model object
            artifact_path: Path where model should be saved
            metadata: Optional model metadata
        """
        self._ensure_started()
        
        self.backend.log_model(model, artifact_path, metadata)
        # Model is logged as artifact in run via backend
    
    def set_tag(self, tag: str) -> None:
        """
        Add a single tag.
        
        Args:
            tag: Tag to add
        """
        self._ensure_started()
        
        if tag not in self.tags:
            self.tags.append(tag)
            self.run.set_tag(tag)
    
    def set_tags(self, tags: Union[List[str], Dict[str, Any]]) -> None:
        """
        Set tags for the run.
        
        Args:
            tags: List of tags or dictionary of key-value tags
        """
        self._ensure_started()
        
        if isinstance(tags, list):
            for tag in tags:
                self.set_tag(tag)
        else:
            self.backend.set_tags(tags)
            self.run.set_tags(list(tags.values()))
    
    def finish(self, status: str = "completed") -> None:
        """
        Finish the experiment tracking.
        
        Args:
            status: Final status (completed, failed, stopped)
        """
        if self._finished:
            return
        
        if not self._started:
            return
        
        # Finish backend
        self.backend.finish(status=status)
        
        # Finish local run
        self.run.finish(status=status)
        
        self._finished = True
        
        print(f"✓ Experiment tracking finished")
        print(f"  Status: {status}")
        print(f"  Run ID: {self.run.run_id}")
    
    def _ensure_started(self) -> None:
        """Ensure tracking has been started."""
        if not self._started:
            self.start()
    
    def get_run_id(self) -> Optional[str]:
        """Get the run ID."""
        return self.backend.get_run_id() if self.backend else None
    
    def get_run_url(self) -> Optional[str]:
        """Get the URL to view the run."""
        return self.backend.get_run_url() if self.backend else None
    
    def get_backend_type(self) -> Optional[str]:
        """Get the backend type being used."""
        return self.backend.get_backend_type().value if self.backend else None
    
    def __enter__(self):
        """Context manager entry."""
        if not self._started:
            self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is not None:
            self.finish(status="failed")
        else:
            self.finish(status="completed")
        return False
    
    def __repr__(self) -> str:
        """String representation."""
        backend_name = type(self.backend).__name__ if self.backend else "None"
        return (
            f"ExperimentTracker("
            f"project='{self.project_name}', "
            f"backend={backend_name}, "
            f"started={self._started})"
        )


# Convenience function
def track_experiment(
    project_name: str,
    experiment_name: Optional[str] = None,
    **kwargs
) -> ExperimentTracker:
    """
    Convenience function to create an ExperimentTracker.
    
    Args:
        project_name: Name of the project
        experiment_name: Optional experiment name
        **kwargs: Additional arguments for ExperimentTracker
    
    Returns:
        ExperimentTracker instance
    
    Example:
        with track_experiment("my_project") as tracker:
            tracker.log_params({"lr": 0.001})
            tracker.log_metrics({"loss": 0.5})
    """
    return ExperimentTracker(
        project_name=project_name,
        experiment_name=experiment_name,
        **kwargs
    )