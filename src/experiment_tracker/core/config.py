"""
Configuration management for ML Experiment Tracker.

Handles loading configuration from YAML, JSON, environment variables,
and provides default configurations with validation.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field, asdict


@dataclass
class BackendConfig:
    """Configuration for tracking backends."""
    
    priority: list = field(default_factory=lambda: ["wandb", "mlflow", "local"])
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_auto_start: bool = True
    wandb_entity: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_mode: str = "online"  # online, offline, disabled
    local_dir: str = "./experiment_logs"


@dataclass
class LoggingConfig:
    """Configuration for logging behavior."""
    
    log_system_info: bool = True
    log_git_info: bool = True
    log_code_snapshot: bool = True
    auto_log_params: bool = True
    auto_log_metrics: bool = True
    metrics_step_key: str = "step"


@dataclass
class ReportingConfig:
    """Configuration for report generation."""
    
    generate_html_report: bool = True
    report_dir: str = "./experiment_reports"
    include_plots: bool = True
    include_diff: bool = True
    plot_format: str = "plotly"  # plotly or matplotlib
    open_report_in_browser: bool = False


@dataclass
class ArtifactConfig:
    """Configuration for artifact management."""
    
    artifact_dir: str = "./experiment_artifacts"
    save_model: bool = True
    save_predictions: bool = True
    compression: bool = False


@dataclass
class Config:
    """Main configuration class for ML Experiment Tracker."""
    
    project_name: str = "default_project"
    experiment_name: Optional[str] = None
    run_name: Optional[str] = None
    tags: list = field(default_factory=list)
    description: str = ""
    
    backend: BackendConfig = field(default_factory=BackendConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)
    
    # Advanced options
    enable_docker: bool = False
    docker_compose_file: Optional[str] = None
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create Config instance from dictionary."""
        # Extract nested configs
        backend_dict = config_dict.pop("backend", {})
        logging_dict = config_dict.pop("logging", {})
        reporting_dict = config_dict.pop("reporting", {})
        artifacts_dict = config_dict.pop("artifacts", {})
        
        return cls(
            backend=BackendConfig(**backend_dict),
            logging=LoggingConfig(**logging_dict),
            reporting=ReportingConfig(**reporting_dict),
            artifacts=ArtifactConfig(**artifacts_dict),
            **config_dict
        )
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "Config":
        """Load configuration from YAML file."""
        yaml_path = Path(yaml_path)
        
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict or {})
    
    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> "Config":
        """Load configuration from JSON file."""
        json_path = Path(json_path)
        
        if not json_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {json_path}")
        
        with open(json_path, "r") as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        config_dict = {}
        
        # Project settings
        if os.getenv("EXP_PROJECT_NAME"):
            config_dict["project_name"] = os.getenv("EXP_PROJECT_NAME")
        if os.getenv("EXP_EXPERIMENT_NAME"):
            config_dict["experiment_name"] = os.getenv("EXP_EXPERIMENT_NAME")
        if os.getenv("EXP_RUN_NAME"):
            config_dict["run_name"] = os.getenv("EXP_RUN_NAME")
        
        # Backend settings
        backend_dict = {}
        if os.getenv("EXP_MLFLOW_URI"):
            backend_dict["mlflow_tracking_uri"] = os.getenv("EXP_MLFLOW_URI")
        if os.getenv("EXP_WANDB_ENTITY"):
            backend_dict["wandb_entity"] = os.getenv("EXP_WANDB_ENTITY")
        if os.getenv("EXP_WANDB_PROJECT"):
            backend_dict["wandb_project"] = os.getenv("EXP_WANDB_PROJECT")
        if os.getenv("EXP_LOCAL_DIR"):
            backend_dict["local_dir"] = os.getenv("EXP_LOCAL_DIR")
        
        if backend_dict:
            config_dict["backend"] = backend_dict
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def load(cls, config_path: Optional[Union[str, Path]] = None) -> "Config":
        """
        Load configuration with priority:
        1. Provided config file (YAML/JSON)
        2. Environment variables
        3. Default values
        """
        # Start with defaults
        config = cls()
        
        # Override with environment variables
        env_config = cls.from_env()
        config = cls._merge_configs(config, env_config)
        
        # Override with file if provided
        if config_path:
            config_path = Path(config_path)
            if config_path.suffix in [".yaml", ".yml"]:
                file_config = cls.from_yaml(config_path)
            elif config_path.suffix == ".json":
                file_config = cls.from_json(config_path)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")
            
            config = cls._merge_configs(config, file_config)
        
        # Validate configuration
        config.validate()
        
        return config
    
    @staticmethod
    def _merge_configs(base: "Config", override: "Config") -> "Config":
        """Merge two configurations, with override taking precedence."""
        base_dict = asdict(base)
        override_dict = asdict(override)
        
        def merge_dicts(base_d: dict, override_d: dict) -> dict:
            """Recursively merge dictionaries."""
            result = base_d.copy()
            for key, value in override_d.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge_dicts(result[key], value)
                else:
                    # Only override if value is not default/empty
                    if value or isinstance(value, bool):
                        result[key] = value
            return result
        
        merged_dict = merge_dicts(base_dict, override_dict)
        return Config.from_dict(merged_dict)
    
    def validate(self) -> None:
        """Validate configuration values."""
        # Validate project name
        if not self.project_name or not isinstance(self.project_name, str):
            raise ValueError("project_name must be a non-empty string")
        
        # Validate backend priority
        valid_backends = {"wandb", "mlflow", "local"}
        if not all(b in valid_backends for b in self.backend.priority):
            raise ValueError(f"Invalid backend in priority list. Valid: {valid_backends}")
        
        # Validate wandb mode
        valid_wandb_modes = {"online", "offline", "disabled"}
        if self.backend.wandb_mode not in valid_wandb_modes:
            raise ValueError(f"Invalid wandb_mode. Valid: {valid_wandb_modes}")
        
        # Validate plot format
        valid_plot_formats = {"plotly", "matplotlib"}
        if self.reporting.plot_format not in valid_plot_formats:
            raise ValueError(f"Invalid plot_format. Valid: {valid_plot_formats}")
        
        # Validate paths (create if needed)
        self._ensure_directories()
    
    def _ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        dirs_to_create = [
            self.backend.local_dir,
            self.reporting.report_dir,
            self.artifacts.artifact_dir,
        ]
        
        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(yaml_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    def to_json(self, json_path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(json_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return (
            f"Config(\n"
            f"  project='{self.project_name}',\n"
            f"  experiment='{self.experiment_name}',\n"
            f"  backend_priority={self.backend.priority},\n"
            f"  generate_reports={self.reporting.generate_html_report}\n"
            f")"
        )


def create_default_config(output_path: Union[str, Path] = "exp_config.yaml") -> None:
    """Create a default configuration file for reference."""
    config = Config()
    
    if str(output_path).endswith(".json"):
        config.to_json(output_path)
    else:
        config.to_yaml(output_path)
    
    print(f"Default configuration saved to: {output_path}")


if __name__ == "__main__":
    # Example usage
    config = Config.load()
    print(config)
    
    # Create sample config file
    create_default_config("sample_config.yaml")