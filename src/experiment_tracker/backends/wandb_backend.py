"""
Weights & Biases (W&B) backend for ML Experiment Tracker.

Integrates with W&B for cloud-based experiment tracking,
collaboration, and public sharing.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

from experiment_tracker.backends.base import (
    BaseBackend,
    BackendType,
    BackendStatus
)


class WandbBackend(BaseBackend):
    """
    Weights & Biases (W&B) tracking backend.
    
    Provides cloud-based experiment tracking with:
    - Automatic cloud sync
    - Collaborative features
    - Public project sharing
    - Rich visualizations
    - Model versioning
    """
    
    def __init__(
        self,
        project_name: str,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
        run_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        description: str = "",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize W&B backend.
        
        Args:
            project_name: Name of the project
            experiment_name: Optional experiment name (used as group)
            run_name: Optional run name
            run_id: Optional run ID
            tags: Optional tags
            description: Optional description
            config: Optional configuration with:
                - entity: W&B entity/team name
                - mode: 'online', 'offline', or 'disabled'
                - anonymous: 'allow', 'must', or 'never'
        """
        super().__init__(
            project_name=project_name,
            experiment_name=experiment_name,
            run_name=run_name,
            run_id=run_id,
            tags=tags,
            description=description,
            config=config
        )
        
        # W&B configuration
        self.entity = self.config.get('entity', None)
        self.mode = self.config.get('mode', 'online')
        self.anonymous = self.config.get('anonymous', 'never')
        
        # W&B objects (initialized later)
        self.wandb = None
        self.run = None
    
    def _import_wandb(self):
        """Import and configure W&B."""
        try:
            import wandb
            self.wandb = wandb
            
            # Set mode
            os.environ['WANDB_MODE'] = self.mode
            
            return True
        except ImportError:
            raise ImportError(
                "Weights & Biases not installed. Install with: pip install wandb"
            )
    
    def initialize(self) -> bool:
        """
        Initialize W&B backend and start tracking.
        
        Returns:
            True if initialization successful
        """
        try:
            # Import W&B
            self._import_wandb()
            
            # Prepare init kwargs
            init_kwargs = {
                'project': self.project_name,
                'name': self.run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'tags': self.tags or [],
                'notes': self.description,
                'config': {},
            }
            
            # Add entity if specified
            if self.entity:
                init_kwargs['entity'] = self.entity
            
            # Add group (experiment name) if specified
            if self.experiment_name:
                init_kwargs['group'] = self.experiment_name
            
            # Add run ID if specified
            if self.run_id:
                init_kwargs['id'] = self.run_id
                init_kwargs['resume'] = 'allow'
            
            # Set anonymous mode
            if self.anonymous:
                init_kwargs['anonymous'] = self.anonymous
            
            # Initialize W&B run
            self.run = self.wandb.init(**init_kwargs)
            
            # Update run_id from W&B
            self.run_id = self.run.id
            
            self._initialized = True
            self._run_active = True
            
            # Print run URL
            if self.run.url:
                print(f"W&B run: {self.run.url}")
            
            return True
        
        except Exception as e:
            print(f"Error initializing W&B backend: {e}")
            return False
    
    def log_param(self, key: str, value: Any) -> None:
        """Log a single parameter."""
        self._validate_initialized()
        
        # W&B uses config for parameters
        self.wandb.config.update({key: value}, allow_val_change=True)
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log multiple parameters."""
        self._validate_initialized()
        
        # Update config with all params
        self.wandb.config.update(params, allow_val_change=True)
    
    def log_metric(
        self,
        key: str,
        value: float,
        step: Optional[int] = None,
        timestamp: Optional[float] = None
    ) -> None:
        """Log a single metric value."""
        self._validate_initialized()
        
        log_dict = {key: value}
        
        # W&B uses step for x-axis
        if step is not None:
            self.wandb.log(log_dict, step=step)
        else:
            self.wandb.log(log_dict)
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        timestamp: Optional[float] = None
    ) -> None:
        """Log multiple metrics."""
        self._validate_initialized()
        
        # Log all metrics at once
        if step is not None:
            self.wandb.log(metrics, step=step)
        else:
            self.wandb.log(metrics)
    
    def log_artifact(
        self,
        artifact_path: Union[str, Path],
        artifact_type: str = "file",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log an artifact."""
        self._validate_initialized()
        
        artifact_path = Path(artifact_path)
        
        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact not found: {artifact_path}")
        
        # Log file or directory
        if artifact_path.is_file():
            self.wandb.save(str(artifact_path))
        elif artifact_path.is_dir():
            # Log all files in directory
            for file in artifact_path.rglob('*'):
                if file.is_file():
                    self.wandb.save(str(file))
    
    def log_model(
        self,
        model: Any,
        artifact_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log a trained model using W&B Artifacts.
        
        W&B has rich model logging with versioning and lineage tracking.
        """
        self._validate_initialized()
        
        # Save model temporarily
        import pickle
        import tempfile
        
        # Determine file extension
        if not artifact_path.endswith(('.pkl', '.pth', '.h5', '.joblib')):
            artifact_path += '.pkl'
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(
            mode='wb',
            suffix=Path(artifact_path).suffix,
            delete=False
        ) as f:
            # Try framework-specific save methods
            if artifact_path.endswith('.pth'):
                try:
                    import torch
                    torch.save(model, f.name)
                except Exception:
                    pickle.dump(model, f)
            elif artifact_path.endswith('.h5'):
                try:
                    model.save(f.name)
                except Exception:
                    pickle.dump(model, f)
            else:
                pickle.dump(model, f)
            
            temp_path = f.name
        
        try:
            # Create W&B artifact
            artifact = self.wandb.Artifact(
                name=Path(artifact_path).stem,
                type='model',
                description=metadata.get('description', 'Model artifact') if metadata else 'Model artifact',
                metadata=metadata or {}
            )
            
            # Add file to artifact
            artifact.add_file(temp_path, name=Path(artifact_path).name)
            
            # Log artifact
            self.run.log_artifact(artifact)
        
        finally:
            # Cleanup temp file
            Path(temp_path).unlink()
    
    def set_tags(self, tags: Dict[str, Any]) -> None:
        """Set tags for the run."""
        self._validate_initialized()
        
        # W&B uses tags as a list and config for key-value pairs
        for key, value in tags.items():
            # Add to config
            self.wandb.config.update({key: value}, allow_val_change=True)
            
            # Also add as tag if it's a string
            if isinstance(value, str) and value not in self.run.tags:
                self.run.tags = self.run.tags + (value,)
    
    def log_system_info(self, system_info: Dict[str, Any]) -> None:
        """Log system information."""
        self._validate_initialized()
        
        # Log as config with 'system_' prefix
        system_config = {f"system_{k}": v for k, v in system_info.items()}
        self.wandb.config.update(system_config, allow_val_change=True)
    
    def finish(self, status: str = "completed") -> None:
        """Finish the run."""
        self._validate_initialized()
        
        # Map status to W&B exit code
        exit_code = 0
        if status == "failed":
            exit_code = 1
        elif status == "stopped":
            exit_code = -1
        
        # Finish W&B run
        self.wandb.finish(exit_code=exit_code)
        
        self._run_active = False
    
    def get_run_id(self) -> Optional[str]:
        """Get the W&B run ID."""
        return self.run_id
    
    def get_run_url(self) -> Optional[str]:
        """Get the URL to view the run in W&B."""
        if self.run and hasattr(self.run, 'url'):
            return self.run.url
        
        # Construct URL manually if available
        if self.run_id and self.entity and self.project_name:
            return f"https://wandb.ai/{self.entity}/{self.project_name}/runs/{self.run_id}"
        
        return None
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if W&B is installed."""
        try:
            import wandb
            return True
        except ImportError:
            return False
    
    @classmethod
    def get_status(cls) -> BackendStatus:
        """Get W&B backend status."""
        try:
            import wandb
            
            # Check if logged in
            if wandb.api.api_key:
                return BackendStatus.AVAILABLE
            else:
                return BackendStatus.NOT_CONFIGURED
        
        except ImportError:
            return BackendStatus.UNAVAILABLE
        except Exception:
            return BackendStatus.NOT_CONFIGURED
    
    @classmethod
    def get_backend_type(cls) -> BackendType:
        """Get backend type."""
        return BackendType.WANDB
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is not None:
            self.finish(status="failed")
        else:
            self.finish(status="completed")
        return False


# Register the backend
from experiment_tracker.backends.base import BackendFactory
BackendFactory.register_backend(BackendType.WANDB, WandbBackend)