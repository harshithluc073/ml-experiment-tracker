# ML Experiment Tracker

Zero-friction ML experiment tracking with intelligent backend fallbacks.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

✅ **Zero Configuration** - Works out of the box with sensible defaults  
✅ **Intelligent Fallbacks** - Automatically selects best available backend (W&B → MLflow → Local)  
✅ **Unified API** - Same code works with any backend  
✅ **Auto-Logging** - System info, git info, environment detection  
✅ **Multiple Backends** - Local, MLflow, Weights & Biases  
✅ **Comprehensive Tracking** - Parameters, metrics, artifacts, models  
✅ **Beautiful Reports** - HTML reports and automatic plotting  
✅ **CLI Interface** - Command-line tools for managing experiments  
✅ **Docker Support** - Containerized deployment ready  

## Installation

```bash
# Basic installation
pip install ml-experiment-tracker

# With all optional dependencies
pip install ml-experiment-tracker[all]

# With specific backends
pip install ml-experiment-tracker[mlflow]
pip install ml-experiment-tracker[wandb]

# With plotting support
pip install ml-experiment-tracker[plotting]
```

## Quick Start

```python
from experiment_tracker import ExperimentTracker

# That's it! One import, everything works
with ExperimentTracker(project_name="my_project") as tracker:
    # Log hyperparameters
    tracker.log_params({
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 20
    })
    
    # Training loop
    for epoch in range(20):
        train_loss, val_acc = train_epoch()
        
        tracker.log_metrics({
            "train_loss": train_loss,
            "val_accuracy": val_acc
        }, step=epoch)
    
    # Save model
    tracker.log_model(model, "final_model.pth")

# Automatically uses W&B, MLflow, or Local backend
# Logs system info, git info, environment
# Saves local metadata copy
# Handles errors gracefully
```

## CLI Usage

```bash
# List experiments
experiment-tracker list

# Show run details
experiment-tracker show <run_id>

# Compare runs
experiment-tracker compare run1 run2

# Generate HTML report
experiment-tracker report <run_id> --output report.html

# Generate plots
experiment-tracker plot <run_id> --output plots/

# Initialize project
experiment-tracker init --project my_project

# Clean old runs
experiment-tracker clean --days 30

# Show version
experiment-tracker version
```

## Advanced Usage

### With Configuration

```python
from experiment_tracker import ExperimentTracker, Config

config = Config(
    project_name="my_project",
    backend=BackendConfig(
        priority=["wandb", "mlflow", "local"],
        mlflow_tracking_uri="http://localhost:5000"
    )
)

with ExperimentTracker(config=config) as tracker:
    tracker.log_params({"lr": 0.001})
```

### Generate Reports

```python
from experiment_tracker.utils import HTMLReporter, AutoPlotter
import json

# Load run data
with open('run.json') as f:
    run_data = json.load(f)

# Generate HTML report
reporter = HTMLReporter()
reporter.generate_run_report(run_data, 'report.html')

# Generate plots
plotter = AutoPlotter()
plotter.plot_training_curves(run_data, 'curves.png')
plotter.create_dashboard(run_data, './dashboard')
```

### Compare Runs

```python
from experiment_tracker.utils import DiffGenerator

diff_gen = DiffGenerator()
comparison = diff_gen.compare_runs('run1.json', 'run2.json')

# Generate report
report = diff_gen.generate_text_report(comparison)
print(report)

# Or save to file
diff_gen.save_report(comparison, 'diff.md', format='markdown')
```

### Artifact Management

```python
from experiment_tracker.utils import ArtifactManager

artifact_mgr = ArtifactManager('./artifacts')

# Add artifacts
artifact_mgr.add_model('model.pth', metadata={'accuracy': 0.95})
artifact_mgr.add_dataset('data.csv')
artifact_mgr.add_plot('loss_curve.png')

# Export metadata
artifact_mgr.export_metadata('artifacts.json')
```

## Docker Deployment

```bash
# Build image
docker build -t ml-experiment-tracker .

# Run with docker-compose
docker-compose up -d

# Execute training in container
docker-compose exec experiment-tracker python train.py

# Access MLflow UI
# http://localhost:5001
```

## Architecture

```
┌─────────────────────────────────────┐
│     ExperimentTracker (API)         │
│  • Unified interface                │
│  • Auto backend selection           │
│  • Context manager                  │
└──────────────┬──────────────────────┘
               │
    ┌──────────┼──────────┐
    │          │          │
┌───▼──┐  ┌───▼───┐  ┌──▼───┐
│ W&B  │  │MLflow │  │Local │
└──────┘  └───────┘  └──────┘
```

## Components

- **Core Tracker**: Unified API for experiment tracking
- **Backends**: Local, MLflow, Weights & Biases support
- **Config System**: Flexible YAML-based configuration
- **Run Management**: Complete run lifecycle tracking
- **Environment Detection**: Auto-detect Colab, Jupyter, VS Code
- **Artifact Manager**: Comprehensive file tracking
- **Diff Generator**: Compare experiment runs
- **HTML Reporter**: Beautiful static reports
- **Auto-Plotter**: Automatic visualization
- **CLI**: Command-line interface

## Backend Comparison

| Feature | Local | MLflow | W&B |
|---------|-------|--------|-----|
| **Setup** | None | Install | Install + Login |
| **Storage** | Local files | Local/Server | Cloud |
| **UI** | Static HTML | Web UI | Cloud Dashboard |
| **Collaboration** | Manual | Self-hosted | Built-in |
| **Always Available** | ✓ | If installed | If installed |
| **Best For** | Quick tests | Team server | Cloud collab |

## Examples

See the `examples/` directory for complete examples:

- `basic_usage.py` - Simple experiment tracking
- `complete_workflow.py` - Full ML training workflow
- `comparison.py` - Comparing multiple runs
- `reports.py` - Generating HTML reports
- `docker_example.py` - Using with Docker

## Documentation

Full documentation available at: [docs/](./docs/)

- [Getting Started](./docs/getting_started.md)
- [Configuration](./docs/configuration.md)
- [Backends](./docs/backends.md)
- [CLI Reference](./docs/cli.md)
- [API Reference](./docs/api.md)

## Development

```bash
# Clone repository
git clone https://github.com/harshithluc073/ml-experiment-tracker.git
cd ml-experiment-tracker

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=experiment_tracker

# Format code
black src/

# Lint
flake8 src/
```

## Requirements

- Python >= 3.8
- PyYAML >= 5.4.0
- psutil >= 5.8.0

### Optional Dependencies

- mlflow >= 2.0.0 (for MLflow backend)
- wandb >= 0.12.0 (for W&B backend)
- matplotlib >= 3.5.0 (for plotting)
- numpy >= 1.21.0 (for plotting)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{ml_experiment_tracker,
  author = {Harshith},
  title = {ML Experiment Tracker: Zero-friction ML experiment tracking},
  year = {2025},
  url = {https://github.com/harshithluc073/ml-experiment-tracker}
}
```

## Acknowledgments

- Inspired by MLflow, Weights & Biases, and other experiment tracking tools
- Built with ❤️ for the ML community

## Support

- 📧 Email: chitikeshiharshith@gmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/harshithluc073/ml-experiment-tracker/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/harshithluc073/ml-experiment-tracker/discussions)

---

**Made with ❤️ by [Harshith](https://github.com/harshithluc073)**