"""
MLflow backend for ML Experiment Tracker.

Integrates with MLflow tracking server for experiment logging,
with automatic server startup and web UI access.
"""

import os
import subprocess
import time
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

from experiment_tracker.backends.base import (
    BaseBackend,
    BackendType,
    BackendStatus
)


class MLflowBackend(BaseBackend):
    """
    MLflow tracking backend.
    
    Provides integration with MLflow for:
    - Experiment tracking
    - Parameter and metric logging
    - Artifact storage
    - Model registry
    - Web UI visualization
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
        Initialize MLflow backend.
        
        Args:
            project_name: Name of the project
            experiment_name: Optional experiment name (combined with project)
            run_name: Optional run name
            run_id: Optional run ID
            tags: Optional tags
            description: Optional description
            config: Optional configuration with:
                - tracking_uri: MLflow tracking URI
                - auto_start_ui: Auto-start MLflow UI (default: True)
                - ui_port: Port for MLflow UI (default: 5000)
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
        
        # MLflow configuration
        self.tracking_uri = self.config.get('tracking_uri', 'http://localhost:5000')
        self.auto_start_ui = self.config.get('auto_start_ui', True)
        self.ui_port = self.config.get('ui_port', 5000)
        
        # MLflow objects (initialized later)
        self.mlflow = None
        self.experiment_id = None
        self.mlflow_run = None
        self.run_info = None
    
    def _import_mlflow(self):
        """Import and configure MLflow."""
        try:
            import mlflow
            self.mlflow = mlflow
            
            # Set tracking URI
            self.mlflow.set_tracking_uri(self.tracking_uri)
            
            return True
        except ImportError:
            raise ImportError(
                "MLflow not installed. Install with: pip install mlflow"
            )
    
    def initialize(self) -> bool:
        """
        Initialize MLflow backend and start tracking.
        
        Returns:
            True if initialization successful
        """
        try:
            # Import MLflow
            self._import_mlflow()
            
            # Create or get experiment
            experiment_name = self._get_experiment_name()
            
            try:
                self.experiment_id = self.mlflow.create_experiment(experiment_name)
            except Exception:
                # Experiment already exists
                experiment = self.mlflow.get_experiment_by_name(experiment_name)
                if experiment:
                    self.experiment_id = experiment.experiment_id
                else:
                    raise RuntimeError(f"Failed to get experiment: {experiment_name}")
            
            # Start MLflow run
            self.mlflow_run = self.mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=self.run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                description=self.description
            )
            
            self.run_info = self.mlflow_run.info
            self.run_id = self.run_info.run_id
            
            # Set tags
            if self.tags:
                for tag in self.tags:
                    self.mlflow.set_tag(tag, "true")
            
            # Set description as tag
            if self.description:
                self.mlflow.set_tag("description", self.description)
            
            self._initialized = True
            self._run_active = True
            
            # Auto-start UI if configured
            if self.auto_start_ui:
                self._start_mlflow_ui()
            
            return True
        
        except Exception as e:
            print(f"Error initializing MLflow backend: {e}")
            return False
    
    def _get_experiment_name(self) -> str:
        """Get full experiment name combining project and experiment."""
        if self.experiment_name:
            return f"{self.project_name}/{self.experiment_name}"
        return self.project_name
    
    def _start_mlflow_ui(self) -> None:
        """Start MLflow UI server if not already running."""
        try:
            # Check if UI is already running
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('localhost', self.ui_port))
            sock.close()
            
            if result == 0:
                # UI already running
                print(f"MLflow UI already running at http://localhost:{self.ui_port}")
                return
            
            # Start MLflow UI in background
            backend_store = self.tracking_uri.replace('http://localhost:', '')
            
            cmd = [
                'mlflow', 'ui',
                '--port', str(self.ui_port),
                '--host', '0.0.0.0'
            ]
            
            # Start process in background
            if os.name == 'nt':  # Windows
                subprocess.Popen(
                    cmd,
                    creationflags=subprocess.CREATE_NEW_CONSOLE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            else:  # Unix/Linux/Mac
                subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True
                )
            
            # Wait a moment for server to start
            time.sleep(2)
            
            # Open browser
            ui_url = f"http://localhost:{self.ui_port}"
            print(f"MLflow UI started at {ui_url}")
            
            try:
                webbrowser.open(ui_url)
            except Exception:
                pass
        
        except Exception as e:
            print(f"Note: Could not auto-start MLflow UI: {e}")
            print(f"Start manually with: mlflow ui --port {self.ui_port}")
    
    def log_param(self, key: str, value: Any) -> None:
        """Log a single parameter."""
        self._validate_initialized()
        
        # MLflow has restrictions on param values
        if isinstance(value, (list, dict)):
            import json
            value = json.dumps(value)
        
        self.mlflow.log_param(key, value)
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log multiple parameters."""
        self._validate_initialized()
        
        # Convert complex types to strings
        processed_params = {}
        for key, value in params.items():
            if isinstance(value, (list, dict)):
                import json
                processed_params[key] = json.dumps(value)
            else:
                processed_params[key] = value
        
        self.mlflow.log_params(processed_params)
    
    def log_metric(
        self,
        key: str,
        value: float,
        step: Optional[int] = None,
        timestamp: Optional[float] = None
    ) -> None:
        """Log a single metric value."""
        self._validate_initialized()
        
        kwargs = {}
        if step is not None:
            kwargs['step'] = step
        
        self.mlflow.log_metric(key, value, **kwargs)
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        timestamp: Optional[float] = None
    ) -> None:
        """Log multiple metrics."""
        self._validate_initialized()
        
        kwargs = {}
        if step is not None:
            kwargs['step'] = step
        
        self.mlflow.log_metrics(metrics, **kwargs)
    
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
        
        if artifact_path.is_file():
            self.mlflow.log_artifact(str(artifact_path))
        elif artifact_path.is_dir():
            self.mlflow.log_artifacts(str(artifact_path))
    
    def log_model(
        self,
        model: Any,
        artifact_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log a trained model using MLflow model logging.
        
        MLflow provides framework-specific model logging which we'll
        use when possible, otherwise fall back to generic artifact logging.
        """
        self._validate_initialized()
        
        # Try to detect model type and use appropriate MLflow logger
        model_type = type(model).__name__
        
        # PyTorch models
        if 'torch' in str(type(model).__module__):
            try:
                self.mlflow.pytorch.log_model(model, artifact_path)
                return
            except Exception:
                pass
        
        # TensorFlow/Keras models
        if hasattr(model, 'save') and 'tensorflow' in str(type(model).__module__):
            try:
                self.mlflow.tensorflow.log_model(model, artifact_path)
                return
            except Exception:
                pass
        
        # Sklearn models
        if 'sklearn' in str(type(model).__module__):
            try:
                self.mlflow.sklearn.log_model(model, artifact_path)
                return
            except Exception:
                pass
        
        # XGBoost models
        if 'xgboost' in str(type(model).__module__):
            try:
                self.mlflow.xgboost.log_model(model, artifact_path)
                return
            except Exception:
                pass
        
        # Fallback: save as pickle and log as artifact
        import pickle
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
            pickle.dump(model, f)
            temp_path = f.name
        
        try:
            self.mlflow.log_artifact(temp_path, artifact_path)
        finally:
            Path(temp_path).unlink()
    
    def set_tags(self, tags: Dict[str, Any]) -> None:
        """Set tags for the run."""
        self._validate_initialized()
        
        for key, value in tags.items():
            self.mlflow.set_tag(key, str(value))
    
    def log_system_info(self, system_info: Dict[str, Any]) -> None:
        """Log system information as tags."""
        self._validate_initialized()
        
        for key, value in system_info.items():
            self.mlflow.set_tag(f"system.{key}", str(value))
    
    def finish(self, status: str = "completed") -> None:
        """Finish the run."""
        self._validate_initialized()
        
        # Map status to MLflow status
        mlflow_status = "FINISHED"
        if status == "failed":
            mlflow_status = "FAILED"
        elif status == "stopped":
            mlflow_status = "KILLED"
        
        self.mlflow.end_run(status=mlflow_status)
        
        self._run_active = False
    
    def get_run_id(self) -> Optional[str]:
        """Get the MLflow run ID."""
        return self.run_id
    
    def get_run_url(self) -> Optional[str]:
        """Get the URL to view the run in MLflow UI."""
        if not self.run_id or not self.experiment_id:
            return None
        
        return f"http://localhost:{self.ui_port}/#/experiments/{self.experiment_id}/runs/{self.run_id}"
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if MLflow is installed."""
        try:
            import mlflow
            return True
        except ImportError:
            return False
    
    @classmethod
    def get_status(cls) -> BackendStatus:
        """Get MLflow backend status."""
        try:
            import mlflow
            return BackendStatus.AVAILABLE
        except ImportError:
            return BackendStatus.UNAVAILABLE
    
    @classmethod
    def get_backend_type(cls) -> BackendType:
        """Get backend type."""
        return BackendType.MLFLOW
    
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
BackendFactory.register_backend(BackendType.MLFLOW, MLflowBackend)