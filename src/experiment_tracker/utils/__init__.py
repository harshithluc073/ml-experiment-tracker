"""Utility functions and helpers."""

from experiment_tracker.utils.env_detector import (
    EnvironmentDetector,
    detect_environment,
    show_setup_instructions
)

__all__ = [
    'EnvironmentDetector',
    'detect_environment',
    'show_setup_instructions',
    'ArtifactManager',
    'Artifact',
    'ArtifactType',
    'DiffGenerator',
    'RunComparison',
    'Diff',
    'DiffType'
]