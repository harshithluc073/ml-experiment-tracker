"""
Pytest configuration and fixtures.
"""

import pytest
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_run_data():
    """Sample run data for testing."""
    return {
        'run_id': 'run_test_001',
        'project_name': 'test_project',
        'experiment_name': 'test_exp',
        'params': {
            'learning_rate': 0.001,
            'batch_size': 32
        },
        'metrics': {
            'loss': [
                {'step': 0, 'value': 1.0},
                {'step': 1, 'value': 0.5},
                {'step': 2, 'value': 0.3}
            ],
            'accuracy': [
                {'step': 0, 'value': 0.5},
                {'step': 1, 'value': 0.8},
                {'step': 2, 'value': 0.92}
            ]
        },
        'system_info': {
            'python_version': '3.10.0',
            'platform': 'Linux'
        },
        'status': 'completed'
    }


@pytest.fixture
def sample_config_dict():
    """Sample configuration dictionary."""
    return {
        'project_name': 'test_project',
        'experiment_name': 'test_exp',
        'backend': {
            'priority': ['local'],
            'local_dir': 'test_logs'
        },
        'logging': {
            'log_system_info': True,
            'log_git_info': False
        }
    }