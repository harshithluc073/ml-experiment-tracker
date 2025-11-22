"""
ML Experiment Tracker
~~~~~~~~~~~~~~~~~~~~~

Zero-friction ML experiment tracking with intelligent fallbacks.

Basic usage:

    >>> from experiment_tracker import ExperimentTracker
    >>> 
    >>> with ExperimentTracker(project_name="my_project") as tracker:
    ...     tracker.log_params({"learning_rate": 0.001})
    ...     tracker.log_metrics({"loss": 0.5}, step=1)

:copyright: (c) 2024 by Harshith.
:license: MIT, see LICENSE for more details.
"""

__version__ = "0.1.0"
__author__ = "Harshith"
__email__ = "chitikeshiharshith@gmail.com"

# Import main classes - ORDER MATTERS!
# Import dependencies first
from experiment_tracker.core.config import Config
from experiment_tracker.core.run import Run, RunManager
from experiment_tracker.utils.env_detector import detect_environment, EnvironmentDetector

# Then import tracker (which depends on the above)
from experiment_tracker.core.tracker import ExperimentTracker, track_experiment

__all__ = [
    "__version__",
    "ExperimentTracker",
    "track_experiment",
    "Config",
    "Run",
    "RunManager",
    "detect_environment",
    "EnvironmentDetector",
]