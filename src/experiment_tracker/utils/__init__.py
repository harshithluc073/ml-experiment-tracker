"""Utility functions and classes."""

from experiment_tracker.utils.env_detector import (
    detect_environment,
    EnvironmentDetector
)
from experiment_tracker.utils.artifact_manager import (
    ArtifactManager,
    Artifact,
    ArtifactType
)
from experiment_tracker.utils.diff_generator import (
    DiffGenerator,
    RunComparison,
    Diff,
    DiffType
)
from experiment_tracker.utils.html_reporter import HTMLReporter
from experiment_tracker.utils.auto_plotter import AutoPlotter

__all__ = [
    'detect_environment',
    'EnvironmentDetector',
    'ArtifactManager',
    'Artifact',
    'ArtifactType',
    'DiffGenerator',
    'RunComparison',
    'Diff',
    'DiffType',
    'HTMLReporter',
    'AutoPlotter'
]
