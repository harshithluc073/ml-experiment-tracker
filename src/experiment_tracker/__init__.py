"""
ML Experiment Tracker
~~~~~~~~~~~~~~~~~~~~~

Zero-friction ML experiment tracking with intelligent fallbacks.

Basic usage:

    >>> from experiment_tracker import ExperimentTracker
    >>> tracker = ExperimentTracker(project_name="my_project")
    >>> tracker.log_params({"learning_rate": 0.001})
    >>> tracker.log_metrics({"loss": 0.5}, step=1)
    >>> tracker.finish()

:copyright: (c) 2024 by Harshith.
:license: MIT, see LICENSE for more details.
"""

__version__ = "0.1.0"
__author__ = "Harshith"
__email__ = "chitikeshiharshith@gmail.com"

# Import main classes (will be implemented in later steps)
# from experiment_tracker.core.tracker import ExperimentTracker
# from experiment_tracker.core.config import Config
# from experiment_tracker.core.run import Run

__all__ = [
    "__version__",
    # "ExperimentTracker",
    # "Config",
    # "Run",
]