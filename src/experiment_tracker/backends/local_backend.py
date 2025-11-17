"""
Local file-based backend for ML Experiment Tracker.

Stores experiments as JSON metadata and CSV metrics in the local filesystem.
Always available with no external dependencies.
"""

import os
import json
import csv
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

from experiment_tracker.backends.base import (
    BaseBackend,
    BackendType,
    BackendStatus
)


class LocalBackend(BaseBackend):
    """
    Local file-based tracking backend.
    
    Stores all experiment data in local directories:
    - Metadata: JSON files
    - Metrics: CSV files
    - Artifacts: Copied to local directory
    - Plots: Auto-generated with matplotlib/plotly
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
        Initialize local backend.
        
        Args:
            project_name: Name of the project
            experiment_name: Optional experiment name
            run_name: Optional run name
            run_id: Optional run ID
            tags: Optional tags
            description: Optional description
            config: Optional configuration with 'storage_dir' key
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
        
        # Get storage directory from config or use default
        self.storage_dir = Path(self.config.get('storage_dir', './experiment_logs'))
        
        # Initialize paths
        self._setup_directories()
        
        # Initialize data stores
        self.params: Dict[str, Any] = {}
        self.metrics: Dict[str, List[Dict[str, Any]]] = {}
        self.artifacts: List[Dict[str, str]] = []
        self.system_info: Dict[str, Any] = {}
        self.tags_dict: Dict[str, Any] = {}
        
        # Metadata file
        self.metadata_file = self.run_dir / "metadata.json"
        self.metrics_file = self.run_dir / "metrics.csv"
    
    def _setup_directories(self) -> None:
        """Create directory structure for the run."""
        # Base project directory
        self.project_dir = self.storage_dir / self.project_name
        
        # Experiment directory (if specified)
        if self.experiment_name:
            self.experiment_dir = self.project_dir / self.experiment_name
        else:
            self.experiment_dir = self.project_dir
        
        # Generate run ID if not provided
        if not self.run_id:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            import uuid
            self.run_id = f"run_{timestamp}_{str(uuid.uuid4())[:8]}"
        
        # Run directory
        self.run_dir = self.experiment_dir / self.run_id
        
        # Create directories
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Artifacts directory
        self.artifacts_dir = self.run_dir / "artifacts"
        self.artifacts_dir.mkdir(exist_ok=True)
        
        # Plots directory
        self.plots_dir = self.run_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
    
    def initialize(self) -> bool:
        """
        Initialize the local backend.
        
        Returns:
            True (always succeeds for local backend)
        """
        try:
            # Generate run name if not provided
            if not self.run_name:
                self.run_name = f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            
            # Initialize metadata
            self.start_time = datetime.now()
            self._save_metadata()
            
            self._initialized = True
            self._run_active = True
            
            return True
        
        except Exception as e:
            print(f"Error initializing local backend: {e}")
            return False
    
    def log_param(self, key: str, value: Any) -> None:
        """Log a single parameter."""
        self._validate_initialized()
        self.params[key] = value
        self._save_metadata()
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log multiple parameters."""
        self._validate_initialized()
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
        self._validate_initialized()
        
        if key not in self.metrics:
            self.metrics[key] = []
        
        timestamp = timestamp or datetime.now().timestamp()
        step = step if step is not None else len(self.metrics[key])
        
        self.metrics[key].append({
            'value': value,
            'step': step,
            'timestamp': timestamp
        })
        
        # Update CSV file
        self._append_to_csv(key, value, step, timestamp)
        self._save_metadata()
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        timestamp: Optional[float] = None
    ) -> None:
        """Log multiple metrics."""
        self._validate_initialized()
        
        timestamp = timestamp or datetime.now().timestamp()
        
        for key, value in metrics.items():
            self.log_metric(key, value, step=step, timestamp=timestamp)
    
    def log_artifact(
        self,
        artifact_path: Union[str, Path],
        artifact_type: str = "file",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log an artifact by copying it to artifacts directory."""
        self._validate_initialized()
        
        artifact_path = Path(artifact_path)
        
        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact not found: {artifact_path}")
        
        # Copy artifact to artifacts directory
        dest_path = self.artifacts_dir / artifact_path.name
        
        if artifact_path.is_file():
            shutil.copy2(artifact_path, dest_path)
        elif artifact_path.is_dir():
            shutil.copytree(artifact_path, dest_path, dirs_exist_ok=True)
        
        # Record artifact info
        artifact_info = {
            'name': artifact_path.name,
            'type': artifact_type,
            'path': str(dest_path.relative_to(self.run_dir)),
            'size_bytes': dest_path.stat().st_size if dest_path.is_file() else 0,
            'logged_at': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        self.artifacts.append(artifact_info)
        self._save_metadata()
    
    def log_model(
        self,
        model: Any,
        artifact_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log a trained model.
        
        Supports multiple formats: pickle, joblib, PyTorch, TensorFlow.
        """
        self._validate_initialized()
        
        model_path = self.artifacts_dir / artifact_path
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine format and save
        if artifact_path.endswith('.pkl'):
            import pickle
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        elif artifact_path.endswith('.joblib'):
            import joblib
            joblib.dump(model, model_path)
        
        elif artifact_path.endswith('.pt') or artifact_path.endswith('.pth'):
            try:
                import torch
                torch.save(model, model_path)
            except ImportError:
                raise ImportError("PyTorch not installed. Install with: pip install torch")
        
        elif artifact_path.endswith('.h5') or artifact_path.endswith('.keras'):
            try:
                model.save(model_path)
            except Exception as e:
                raise RuntimeError(f"Failed to save model: {e}")
        
        else:
            # Default to pickle
            import pickle
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        # Record model info
        model_info = {
            'name': artifact_path,
            'type': 'model',
            'path': str(model_path.relative_to(self.run_dir)),
            'size_bytes': model_path.stat().st_size,
            'logged_at': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        self.artifacts.append(model_info)
        self._save_metadata()
    
    def set_tags(self, tags: Dict[str, Any]) -> None:
        """Set tags for the run."""
        self._validate_initialized()
        self.tags_dict.update(tags)
        self._save_metadata()
    
    def log_system_info(self, system_info: Dict[str, Any]) -> None:
        """Log system information."""
        self._validate_initialized()
        self.system_info = system_info
        self._save_metadata()
    
    def finish(self, status: str = "completed") -> None:
        """Finish the run and save final state."""
        self._validate_initialized()
        
        self.end_time = datetime.now()
        self.status = status
        self._run_active = False
        
        # Generate plots if we have metrics
        if self.metrics:
            self._generate_plots()
        
        # Save final metadata
        self._save_metadata()
    
    def get_run_id(self) -> Optional[str]:
        """Get the run ID."""
        return self.run_id
    
    def get_run_url(self) -> Optional[str]:
        """Get the local path to the run directory."""
        return f"file://{self.run_dir.absolute()}"
    
    @classmethod
    def is_available(cls) -> bool:
        """Local backend is always available."""
        return True
    
    @classmethod
    def get_status(cls) -> BackendStatus:
        """Local backend is always available."""
        return BackendStatus.AVAILABLE
    
    @classmethod
    def get_backend_type(cls) -> BackendType:
        """Get backend type."""
        return BackendType.LOCAL
    
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
    
    def _save_metadata(self) -> None:
        """Save metadata to JSON file."""
        metadata = {
            'run_id': self.run_id,
            'run_name': self.run_name,
            'project_name': self.project_name,
            'experiment_name': self.experiment_name,
            'status': getattr(self, 'status', 'running'),
            'start_time': getattr(self, 'start_time', datetime.now()).isoformat(),
            'end_time': getattr(self, 'end_time', None).isoformat() if hasattr(self, 'end_time') and self.end_time else None,
            'tags': self.tags,
            'tags_dict': self.tags_dict,
            'description': self.description,
            'params': self.params,
            'metrics_summary': self._calculate_metrics_summary(),
            'artifacts': self.artifacts,
            'system_info': self.system_info,
            'backend': 'local',
            'run_directory': str(self.run_dir)
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def _append_to_csv(self, key: str, value: float, step: int, timestamp: float) -> None:
        """Append metric to CSV file."""
        file_exists = self.metrics_file.exists()
        
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # Write header if file is new
            if not file_exists:
                writer.writerow(['metric', 'value', 'step', 'timestamp'])
            
            writer.writerow([key, value, step, timestamp])
    
    def _calculate_metrics_summary(self) -> Dict[str, Any]:
        """Calculate summary statistics for metrics."""
        summary = {}
        
        for metric_name, values in self.metrics.items():
            if not values:
                continue
            
            metric_values = [v['value'] for v in values]
            
            # Determine if lower is better (for loss/error metrics)
            is_loss = any(term in metric_name.lower() for term in ['loss', 'error'])
            
            summary[metric_name] = {
                'final': metric_values[-1],
                'best': min(metric_values) if is_loss else max(metric_values),
                'mean': sum(metric_values) / len(metric_values),
                'min': min(metric_values),
                'max': max(metric_values),
                'count': len(metric_values)
            }
        
        return summary
    
    def _generate_plots(self) -> None:
        """Generate plots for metrics."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            
            # Create a plot for each metric
            for metric_name, values in self.metrics.items():
                if not values:
                    continue
                
                steps = [v['step'] for v in values]
                metric_values = [v['value'] for v in values]
                
                plt.figure(figsize=(10, 6))
                plt.plot(steps, metric_values, marker='o', linewidth=2, markersize=4)
                plt.xlabel('Step')
                plt.ylabel(metric_name)
                plt.title(f'{metric_name} over time')
                plt.grid(True, alpha=0.3)
                
                # Save plot
                plot_path = self.plots_dir / f"{metric_name.replace('/', '_')}.png"
                plt.savefig(plot_path, dpi=100, bbox_inches='tight')
                plt.close()
            
            # Create combined plot if multiple metrics
            if len(self.metrics) > 1:
                fig, axes = plt.subplots(
                    len(self.metrics),
                    1,
                    figsize=(12, 4 * len(self.metrics))
                )
                
                if len(self.metrics) == 1:
                    axes = [axes]
                
                for ax, (metric_name, values) in zip(axes, self.metrics.items()):
                    steps = [v['step'] for v in values]
                    metric_values = [v['value'] for v in values]
                    
                    ax.plot(steps, metric_values, marker='o', linewidth=2, markersize=4)
                    ax.set_xlabel('Step')
                    ax.set_ylabel(metric_name)
                    ax.set_title(metric_name)
                    ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                combined_path = self.plots_dir / "all_metrics.png"
                plt.savefig(combined_path, dpi=100, bbox_inches='tight')
                plt.close()
        
        except ImportError:
            # Matplotlib not available, skip plotting
            pass
        except Exception as e:
            print(f"Warning: Failed to generate plots: {e}")


# Register the backend
from experiment_tracker.backends.base import BackendFactory
BackendFactory.register_backend(BackendType.LOCAL, LocalBackend)