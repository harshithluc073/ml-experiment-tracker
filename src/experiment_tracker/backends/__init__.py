"""Backend implementations for different tracking services."""

from experiment_tracker.backends.base import (
    BaseBackend,
    BackendType,
    BackendStatus,
    BackendFactory,
    create_backend
)

from experiment_tracker.backends.local_backend import LocalBackend

# MLflow backend - conditionally imported
try:
    from experiment_tracker.backends.mlflow_backend import MLflowBackend
    _mlflow_available = True
except ImportError:
    _mlflow_available = False
    MLflowBackend = None

__all__ = [
    'BaseBackend',
    'BackendType',
    'BackendStatus',
    'BackendFactory',
    'create_backend',
    'LocalBackend',
    'MLflowBackend'
]