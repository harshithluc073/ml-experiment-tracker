# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-11-26

### Added
- Initial release of ML Experiment Tracker
- Core experiment tracking with ExperimentTracker API
- Configuration system with YAML/JSON support
- Run metadata management
- Environment detection (Colab, Jupyter, VS Code, Terminal)
- Three backend implementations:
  - Local backend (zero dependencies, always available)
  - MLflow backend (self-hosted with web UI)
  - Weights & Biases backend (cloud-based)
- Intelligent backend fallback mechanism
- Auto-logging features:
  - System information (CPU, RAM, GPU, Python, OS)
  - Git information (commit, branch, dirty status)
  - Environment detection
- Artifact management system
- Run comparison with diff generator
- HTML report generation
- Automatic plotting and visualization
- Docker support with docker-compose
- CLI interface with commands:
  - `list` - List experiment runs
  - `show` - Show run details
  - `compare` - Compare multiple runs
  - `report` - Generate HTML reports
  - `plot` - Generate plots
  - `init` - Initialize project
  - `clean` - Clean old runs
  - `version` - Show version
- Comprehensive test suite:
  - Unit tests for core components
  - Integration tests for workflows
  - Test fixtures and utilities
- CI/CD pipeline with GitHub Actions
- Code quality tools:
  - Black for formatting
  - Flake8 for linting
  - pytest for testing
  - Coverage reporting
- Complete documentation:
  - README with quick start
  - Contributing guidelines
  - API documentation
  - Examples and tutorials

### Features
- ✅ Zero-configuration tracking
- ✅ Intelligent backend selection
- ✅ Unified API for all backends
- ✅ Context manager support
- ✅ Dual logging (backend + local)
- ✅ Framework-agnostic model logging
- ✅ Automatic metric plotting
- ✅ Run comparison and diff reports
- ✅ HTML report generation
- ✅ Docker containerization
- ✅ CLI tools
- ✅ Type hints
- ✅ Comprehensive tests

### Technical Details
- Python 3.8+ support
- Dependencies: PyYAML, psutil
- Optional dependencies: mlflow, wandb, matplotlib, numpy
- Development tools: pytest, black, flake8, mypy
- Docker support with multi-stage builds
- GitHub Actions CI/CD

---

## [Unreleased]

### Planned Features
- [ ] Web UI dashboard
- [ ] REST API
- [ ] Experiment templates
- [ ] Hyperparameter tuning integration
- [ ] Model registry
- [ ] Experiment scheduling
- [ ] Notification system
- [ ] Cloud storage backends (S3, GCS, Azure)
- [ ] Experiment search and filtering
- [ ] Team collaboration features

---

## Release Notes Format

### Added
- New features

### Changed
- Changes in existing functionality

### Deprecated
- Soon-to-be removed features

### Removed
- Removed features

### Fixed
- Bug fixes

### Security
- Security improvements