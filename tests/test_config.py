"""
Unit tests for configuration system.
"""

import pytest
from pathlib import Path
import tempfile
import yaml
import json

from experiment_tracker.core.config import (
    Config,
    BackendConfig,
    LoggingConfig
)


class TestBackendConfig:
    """Tests for BackendConfig."""
    
    def test_default_config(self):
        """Test default backend configuration."""
        config = BackendConfig()
        
        assert config.priority == ["wandb", "mlflow", "local"]
        assert config.local_dir == "experiment_logs"
        assert config.mlflow_tracking_uri == "http://localhost:5000"
        assert config.mlflow_auto_start is True
    
    def test_custom_config(self):
        """Test custom backend configuration."""
        config = BackendConfig(
            priority=["local"],
            local_dir="my_logs",
            mlflow_tracking_uri="http://server:8080"
        )
        
        assert config.priority == ["local"]
        assert config.local_dir == "my_logs"
        assert config.mlflow_tracking_uri == "http://server:8080"


class TestLoggingConfig:
    """Tests for LoggingConfig."""
    
    def test_default_config(self):
        """Test default logging configuration."""
        config = LoggingConfig()
        
        assert config.log_system_info is True
        assert config.log_git_info is True
    
    def test_custom_config(self):
        """Test custom logging configuration."""
        config = LoggingConfig(
            log_system_info=False,
            log_git_info=False
        )
        
        assert config.log_system_info is False
        assert config.log_git_info is False


class TestConfig:
    """Tests for Config class."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = Config(project_name="test_project")
        
        assert config.project_name == "test_project"
        assert config.experiment_name is None
        assert isinstance(config.backend, BackendConfig)
        assert isinstance(config.logging, LoggingConfig)
    
    def test_custom_config(self):
        """Test custom configuration."""
        backend = BackendConfig(priority=["local"])
        logging = LoggingConfig(log_system_info=False)
        
        config = Config(
            project_name="test",
            experiment_name="exp1",
            backend=backend,
            logging=logging
        )
        
        assert config.project_name == "test"
        assert config.experiment_name == "exp1"
        assert config.backend.priority == ["local"]
        assert config.logging.log_system_info is False
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = Config(project_name="test")
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['project_name'] == "test"
        assert 'backend' in config_dict
        assert 'logging' in config_dict
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        config_dict = {
            'project_name': 'test',
            'experiment_name': 'exp1',
            'backend': {
                'priority': ['local']
            },
            'logging': {
                'log_system_info': False
            }
        }
        
        config = Config.from_dict(config_dict)
        
        assert config.project_name == "test"
        assert config.experiment_name == "exp1"
        assert config.backend.priority == ["local"]
        assert config.logging.log_system_info is False
    
    def test_save_yaml(self):
        """Test saving configuration to YAML."""
        config = Config(project_name="test")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            config.save(temp_path)
            
            # Verify file exists
            assert Path(temp_path).exists()
            
            # Verify content
            with open(temp_path) as f:
                data = yaml.safe_load(f)
            
            assert data['project_name'] == "test"
            assert 'backend' in data
        
        finally:
            Path(temp_path).unlink()
    
    def test_load_yaml(self):
        """Test loading configuration from YAML."""
        config_dict = {
            'project_name': 'test',
            'experiment_name': 'exp1'
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_dict, f)
            temp_path = f.name
        
        try:
            config = Config.load(temp_path)
            
            assert config.project_name == "test"
            assert config.experiment_name == "exp1"
        
        finally:
            Path(temp_path).unlink()
    
    def test_save_json(self):
        """Test saving configuration to JSON."""
        config = Config(project_name="test")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            config.save(temp_path)
            
            # Verify file exists
            assert Path(temp_path).exists()
            
            # Verify content
            with open(temp_path) as f:
                data = json.load(f)
            
            assert data['project_name'] == "test"
        
        finally:
            Path(temp_path).unlink()
    
    def test_load_json(self):
        """Test loading configuration from JSON."""
        config_dict = {
            'project_name': 'test',
            'experiment_name': 'exp1'
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_dict, f)
            temp_path = f.name
        
        try:
            config = Config.load(temp_path)
            
            assert config.project_name == "test"
            assert config.experiment_name == "exp1"
        
        finally:
            Path(temp_path).unlink()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])