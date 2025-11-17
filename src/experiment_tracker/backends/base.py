"""
Base backend interface for ML Experiment Tracker.

Defines the abstract interface that all tracking backends must implement,
providing a consistent API regardless of the underlying tracking service.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from enum import Enum


class BackendType(Enum):
    """Enumeration of available backend types."""
    WANDB = "wandb"
    MLFLOW = "mlflow"
    LOCAL = "local"


class BackendStatus(Enum):
    """Backend availability status."""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    NOT_CONFIGURED = "not_configured"
    ERROR = "error"


class BaseBackend(ABC):
    """
    Abstract base class for experiment tracking backends.
    
    All backend implementations must inherit from this class and implement
    all abstract methods to ensure consistent behavior across different
    tracking services.
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
        Initialize backend.
        
        Args:
            project_name: Name of the project
            experiment_name: Optional experiment name for grouping runs
            run_name: Optional custom run name
            run_id: Optional custom run ID
            tags: Optional list of tags
            description: Optional run description
            config: Optional backend-specific configuration
        """
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.run_id = run_id
        self.tags = tags or []
        self.description = description
        self.config = config or {}
        
        self._initialized = False
        self._run_active = False
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the backend and start tracking.
        
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def log_param(self, key: str, value: Any) -> None:
        """
        Log a single parameter.
        
        Args:
            key: Parameter name
            value: Parameter value
        """
        pass
    
    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log multiple parameters at once.
        
        Args:
            params: Dictionary of parameters
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
            artifact_type: Type of artifact (file, model, plot, etc.)
            metadata: Optional metadata about the artifact
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def set_tags(self, tags: Dict[str, Any]) -> None:
        """
        Set tags for the run.
        
        Args:
            tags: Dictionary of tags
        """
        pass
    
    @abstractmethod
    def log_system_info(self, system_info: Dict[str, Any]) -> None:
        """
        Log system information.
        
        Args:
            system_info: Dictionary containing system information
        """
        pass
    
    @abstractmethod
    def finish(self, status: str = "completed") -> None:
        """
        Finish the run and cleanup.
        
        Args:
            status: Final run status (completed, failed, stopped)
        """
        pass
    
    @abstractmethod
    def get_run_id(self) -> Optional[str]:
        """
        Get the backend-specific run ID.
        
        Returns:
            Run ID or None if not available
        """
        pass
    
    @abstractmethod
    def get_run_url(self) -> Optional[str]:
        """
        Get the URL to view the run in the backend's UI.
        
        Returns:
            Run URL or None if not available
        """
        pass
    
    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """
        Check if this backend is available (dependencies installed, etc.).
        
        Returns:
            True if backend can be used, False otherwise
        """
        pass
    
    @classmethod
    @abstractmethod
    def get_status(cls) -> BackendStatus:
        """
        Get the current status of the backend.
        
        Returns:
            BackendStatus enum value
        """
        pass
    
    @classmethod
    @abstractmethod
    def get_backend_type(cls) -> BackendType:
        """
        Get the type of this backend.
        
        Returns:
            BackendType enum value
        """
        pass
    
    @abstractmethod
    def __enter__(self):
        """Context manager entry."""
        pass
    
    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass
    
    def _validate_initialized(self) -> None:
        """Check if backend is initialized, raise error if not."""
        if not self._initialized:
            raise RuntimeError(
                f"{self.__class__.__name__} not initialized. "
                "Call initialize() first or use as context manager."
            )
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}("
            f"project='{self.project_name}', "
            f"experiment='{self.experiment_name}', "
            f"initialized={self._initialized})"
        )


class BackendFactory:
    """
    Factory for creating backend instances based on availability and priority.
    """
    
    _backends: Dict[BackendType, type] = {}
    
    @classmethod
    def register_backend(cls, backend_type: BackendType, backend_class: type) -> None:
        """
        Register a backend implementation.
        
        Args:
            backend_type: Type of backend
            backend_class: Backend class (must inherit from BaseBackend)
        """
        if not issubclass(backend_class, BaseBackend):
            raise TypeError(f"{backend_class} must inherit from BaseBackend")
        
        cls._backends[backend_type] = backend_class
    
    @classmethod
    def create_backend(
        cls,
        backend_type: Union[BackendType, str],
        project_name: str,
        **kwargs
    ) -> BaseBackend:
        """
        Create a backend instance.
        
        Args:
            backend_type: Type of backend to create
            project_name: Project name
            **kwargs: Additional arguments for backend initialization
            
        Returns:
            Backend instance
            
        Raises:
            ValueError: If backend type is unknown
            RuntimeError: If backend is not available
        """
        # Convert string to BackendType if needed
        if isinstance(backend_type, str):
            try:
                backend_type = BackendType(backend_type.lower())
            except ValueError:
                raise ValueError(f"Unknown backend type: {backend_type}")
        
        # Get backend class
        backend_class = cls._backends.get(backend_type)
        if backend_class is None:
            raise ValueError(f"Backend not registered: {backend_type}")
        
        # Check availability
        if not backend_class.is_available():
            raise RuntimeError(
                f"{backend_type.value} backend is not available. "
                f"Status: {backend_class.get_status().value}"
            )
        
        # Create instance
        return backend_class(project_name=project_name, **kwargs)
    
    @classmethod
    def create_with_fallback(
        cls,
        priority: List[Union[BackendType, str]],
        project_name: str,
        **kwargs
    ) -> BaseBackend:
        """
        Create backend with fallback logic.
        
        Tries to create backends in priority order, falling back to the next
        available backend if the preferred one is not available.
        
        Args:
            priority: List of backend types in order of preference
            project_name: Project name
            **kwargs: Additional arguments for backend initialization
            
        Returns:
            First available backend instance
            
        Raises:
            RuntimeError: If no backends are available
        """
        errors = []
        
        for backend_type in priority:
            try:
                backend = cls.create_backend(backend_type, project_name, **kwargs)
                return backend
            except (ValueError, RuntimeError) as e:
                errors.append(f"{backend_type}: {str(e)}")
                continue
        
        # No backends available
        raise RuntimeError(
            f"No backends available. Tried: {', '.join(str(p) for p in priority)}. "
            f"Errors: {'; '.join(errors)}"
        )
    
    @classmethod
    def get_available_backends(cls) -> List[BackendType]:
        """
        Get list of available backends.
        
        Returns:
            List of available backend types
        """
        available = []
        for backend_type, backend_class in cls._backends.items():
            if backend_class.is_available():
                available.append(backend_type)
        return available
    
    @classmethod
    def get_backend_status(cls, backend_type: Union[BackendType, str]) -> BackendStatus:
        """
        Get status of a specific backend.
        
        Args:
            backend_type: Type of backend
            
        Returns:
            Backend status
        """
        if isinstance(backend_type, str):
            backend_type = BackendType(backend_type.lower())
        
        backend_class = cls._backends.get(backend_type)
        if backend_class is None:
            return BackendStatus.UNAVAILABLE
        
        return backend_class.get_status()


# Utility function for easy backend creation
def create_backend(
    backend_type: Union[BackendType, str, None] = None,
    project_name: str = "default_project",
    priority: Optional[List[str]] = None,
    **kwargs
) -> BaseBackend:
    """
    Convenience function to create a backend with smart defaults.
    
    Args:
        backend_type: Specific backend to use, or None for auto-selection
        project_name: Project name
        priority: List of backends in order of preference (for fallback)
        **kwargs: Additional backend configuration
        
    Returns:
        Backend instance
        
    Examples:
        # Use specific backend
        backend = create_backend("mlflow", project_name="my_project")
        
        # Auto-select with fallback
        backend = create_backend(
            priority=["wandb", "mlflow", "local"],
            project_name="my_project"
        )
    """
    if backend_type is not None:
        return BackendFactory.create_backend(backend_type, project_name, **kwargs)
    
    if priority is None:
        priority = ["wandb", "mlflow", "local"]
    
    return BackendFactory.create_with_fallback(priority, project_name, **kwargs)


if __name__ == "__main__":
    # Example of how backends will be registered
    print("Base Backend Interface")
    print("=" * 60)
    print()
    print("This is the abstract base class for all tracking backends.")
    print()
    print("Backend Types:")
    for backend_type in BackendType:
        print(f"  - {backend_type.value}")
    print()
    print("All backends must implement:")
    print("  - initialize()")
    print("  - log_param() / log_params()")
    print("  - log_metric() / log_metrics()")
    print("  - log_artifact()")
    print("  - log_model()")
    print("  - set_tags()")
    print("  - log_system_info()")
    print("  - finish()")
    print("  - get_run_id() / get_run_url()")
    print("  - is_available() / get_status()")
    print()
    print("Factory pattern enables:")
    print("  - Automatic backend selection")
    print("  - Fallback to available backends")
    print("  - Consistent API across all backends")