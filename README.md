# ML Experiment Tracker

> Zero-friction ML experiment tracking with intelligent backend fallbacks

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](https://github.com/harshithluc073/ml-experiment-tracker)
[![Coverage](https://img.shields.io/badge/coverage-80%25-green.svg)](https://github.com/harshithluc073/ml-experiment-tracker)

A production-ready, feature-rich ML experiment tracking system that just works. No complex setup, no configuration files, no headaches - just track your experiments.

## 🌟 Key Features

### Core Capabilities
- ✅ **Zero Configuration** - Works out of the box with sensible defaults
- ✅ **Intelligent Fallbacks** - Automatically tries W&B → MLflow → Local backend
- ✅ **Unified API** - Same code works across all backends
- ✅ **Auto-Logging** - Captures system info, git status, and environment automatically
- ✅ **Context Manager** - Clean, Pythonic API with automatic lifecycle management
- ✅ **Dual Logging** - Always saves local copy regardless of backend

### Backend Support
- 🔵 **Weights & Biases** - Cloud-based with beautiful UI and collaboration
- 🟢 **MLflow** - Self-hosted with web UI and model registry
- 🟡 **Local** - Zero-dependency fallback that always works

### Advanced Features
- 📊 **HTML Reports** - Beautiful, self-contained reports with embedded CSS
- 📈 **Auto-Plotting** - Automatic visualization of training curves
- 🔄 **Run Comparison** - Detailed diff reports between experiments
- 📦 **Artifact Management** - Comprehensive file and model tracking
- 🐳 **Docker Support** - Production-ready containerization
- 💻 **CLI Interface** - Command-line tools for experiment management

---

## 📦 Installation

### Basic Installation

```bash
pip install ml-experiment-tracker
```

### With Optional Dependencies

```bash
# All features
pip install ml-experiment-tracker[all]

# Specific backends
pip install ml-experiment-tracker[mlflow]
pip install ml-experiment-tracker[wandb]

# Plotting support
pip install ml-experiment-tracker[plotting]

# Development tools
pip install ml-experiment-tracker[dev]
```

### From Source

```bash
git clone https://github.com/harshithluc073/ml-experiment-tracker.git
cd ml-experiment-tracker
pip install -e ".[all]"
```

---

## 🚀 Quick Start

### Basic Usage

```python
from experiment_tracker import ExperimentTracker

# That's it! Everything else is automatic
with ExperimentTracker(project_name="my_project") as tracker:
    # Log hyperparameters
    tracker.log_params({
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10
    })
    
    # Training loop
    for epoch in range(10):
        # Your training code here
        train_loss = train_one_epoch()
        val_accuracy = validate()
        
        # Log metrics
        tracker.log_metrics({
            "train_loss": train_loss,
            "val_accuracy": val_accuracy
        }, step=epoch)
    
    # Save model
    tracker.log_model(model, "best_model.pth")
    
    print(f"✓ Experiment tracked: {tracker.get_run_id()}")
```

### What Happens Automatically?

When you run the code above, the tracker automatically:
1. ✅ Tries to connect to W&B (if installed and configured)
2. ✅ Falls back to MLflow (if installed)
3. ✅ Falls back to Local backend (always available)
4. ✅ Captures system info (CPU, RAM, GPU, Python version, OS)
5. ✅ Captures git info (commit, branch, dirty status)
6. ✅ Detects environment (Colab, Jupyter, VS Code, Terminal)
7. ✅ Saves local copy to `experiment_logs/`
8. ✅ Organizes everything by project and run ID

---

## 📖 Detailed Usage

### Configuration

#### YAML Configuration

Create `experiment_config.yaml`:

```yaml
project_name: "image_classification"
experiment_name: "resnet_baseline"
tags:
  - "production"
  - "baseline"

backend:
  priority:
    - "wandb"
    - "mlflow"
    - "local"
  local_dir: "experiment_logs"
  mlflow_tracking_uri: "http://localhost:5000"

logging:
  log_system_info: true
  log_git_info: true
```

Use it:

```python
from experiment_tracker import ExperimentTracker, Config

config = Config.load("experiment_config.yaml")

with ExperimentTracker(config=config) as tracker:
    tracker.log_params({"lr": 0.001})
```

#### JSON Configuration

```json
{
  "project_name": "my_project",
  "backend": {
    "priority": ["local"],
    "local_dir": "my_experiments"
  }
}
```

#### Python Configuration

```python
from experiment_tracker import ExperimentTracker, Config, BackendConfig

config = Config(
    project_name="my_project",
    experiment_name="experiment_1",
    backend=BackendConfig(
        priority=["mlflow", "local"],
        mlflow_tracking_uri="http://localhost:5000"
    )
)

with ExperimentTracker(config=config) as tracker:
    # Your code here
    pass
```

### Logging Parameters

```python
with ExperimentTracker(project_name="demo") as tracker:
    # Single parameter
    tracker.log_param("learning_rate", 0.001)
    
    # Multiple parameters
    tracker.log_params({
        "learning_rate": 0.001,
        "batch_size": 32,
        "optimizer": "adam",
        "model": "resnet50"
    })
```

### Logging Metrics

```python
with ExperimentTracker(project_name="demo") as tracker:
    # Single metric
    tracker.log_metric("loss", 0.5, step=1)
    
    # Multiple metrics
    tracker.log_metrics({
        "train_loss": 0.5,
        "val_loss": 0.6,
        "accuracy": 0.92
    }, step=1)
    
    # Training loop
    for epoch in range(100):
        tracker.log_metrics({
            "train_loss": compute_train_loss(),
            "val_loss": compute_val_loss(),
            "learning_rate": get_lr()
        }, step=epoch)
```

### Logging Artifacts

```python
with ExperimentTracker(project_name="demo") as tracker:
    # Log a file
    tracker.log_artifact("model.pth")
    
    # Log a directory
    tracker.log_artifact("checkpoints/", is_dir=True)
    
    # Log with description
    tracker.log_artifact(
        "results.csv",
        description="Final evaluation results"
    )
```

### Logging Models

```python
import torch

with ExperimentTracker(project_name="demo") as tracker:
    # PyTorch model
    tracker.log_model(model, "final_model.pth")
    
    # With metadata
    tracker.log_model(
        model,
        "best_model.pth",
        metadata={"accuracy": 0.95, "f1_score": 0.93}
    )
```

### Tags and Metadata

```python
with ExperimentTracker(
    project_name="demo",
    experiment_name="baseline",
    tags=["production", "v1"]
) as tracker:
    # Add more tags
    tracker.set_tag("approved")
    tracker.set_tags(["reviewed", "deployed"])
    
    # Add description
    tracker.run.description = "Baseline model with ResNet50"
```

---

## 🎨 Advanced Features

### HTML Reports

Generate beautiful, self-contained HTML reports:

```python
from experiment_tracker.utils import HTMLReporter
import json

# Load run data
with open('experiment_logs/my_project/run_xxx/run.json') as f:
    run_data = json.load(f)

# Generate report
reporter = HTMLReporter()
reporter.generate_run_report(
    run_data,
    output_path='report.html'
)
```

**Features:**
- Self-contained (embedded CSS/JS)
- Responsive design
- Print-friendly
- Overview, parameters, metrics, system info, artifacts

### Automatic Plotting

```python
from experiment_tracker.utils import AutoPlotter

plotter = AutoPlotter(style='seaborn')

# Training curves
plotter.plot_training_curves(
    run_data,
    metrics=['loss', 'accuracy'],
    output_path='curves.png'
)

# Metric comparison across runs
plotter.plot_metric_comparison(
    runs_data=[run1, run2, run3],
    metric_name='accuracy',
    output_path='comparison.png'
)

# Parameter impact analysis
plotter.plot_parameter_impact(
    runs_data,
    param_name='learning_rate',
    metric_name='accuracy',
    output_path='param_impact.png'
)

# Complete dashboard
plotter.create_dashboard(
    run_data,
    output_dir='./dashboard'
)
```

### Run Comparison

Compare multiple runs with detailed diff reports:

```python
from experiment_tracker.utils import DiffGenerator

diff_gen = DiffGenerator()

# Compare two runs
comparison = diff_gen.compare_runs(
    'run1.json',
    'run2.json'
)

# Text report
report = diff_gen.generate_text_report(comparison)
print(report)

# Markdown report (GitHub-compatible)
markdown = diff_gen.generate_markdown_report(comparison)

# Save to file
diff_gen.save_report(
    comparison,
    output_path='diff.md',
    format='markdown'
)
```

**Comparison includes:**
- Parameter differences
- Metric differences (final values)
- System info changes
- Git commit changes
- Summary statistics

### Artifact Management

Comprehensive artifact tracking with metadata:

```python
from experiment_tracker.utils import ArtifactManager
from pathlib import Path

# Initialize
artifacts_dir = Path("./artifacts")
artifact_mgr = ArtifactManager(artifacts_dir)

# Add artifacts
artifact_mgr.add_model(
    "model.pth",
    metadata={"accuracy": 0.95, "size_mb": 100}
)

artifact_mgr.add_dataset(
    "train_data.csv",
    description="Training dataset with 10k samples"
)

artifact_mgr.add_plot("loss_curve.png")
artifact_mgr.add_log("training.log")

# Add checkpoint with epoch info
artifact_mgr.add_checkpoint("checkpoint.pth", epoch=50)

# Query artifacts
models = artifact_mgr.get_artifacts_by_type(ArtifactType.MODEL)
specific = artifact_mgr.get_artifact_by_name("model.pth")

# Get summary
summary = artifact_mgr.get_summary()
print(f"Total artifacts: {summary['total_artifacts']}")
print(f"Total size: {summary['total_size_mb']:.2f} MB")

# Export metadata
artifact_mgr.export_metadata("artifacts.json")
```

---

## 💻 CLI Usage

### List Experiments

```bash
# List all runs
python test_cli.py list

# List by project
python test_cli.py list --project my_project

# Limit results
python test_cli.py list --limit 5

# Filter by status
python test_cli.py list --status completed
```

### Show Run Details

```bash
# Show run information
python test_cli.py show run_20241126_120000_abc123

# Output as JSON
python test_cli.py show run_20241126_120000_abc123 --json
```

### Compare Runs

```bash
# Compare two runs (text format)
python test_cli.py compare run1 run2

# Compare with markdown output
python test_cli.py compare run1 run2 --format markdown

# Save to file
python test_cli.py compare run1 run2 --output diff.md --format markdown
```

### Generate Reports

```bash
# Generate HTML report
python test_cli.py report run_20241126_120000_abc123

# Custom output path
python test_cli.py report run_20241126_120000_abc123 --output report.html
```

### Generate Plots

```bash
# Generate all plots
python test_cli.py plot run_20241126_120000_abc123

# Specific metrics
python test_cli.py plot run_20241126_120000_abc123 --metrics loss accuracy

# Custom output directory
python test_cli.py plot run_20241126_120000_abc123 --output ./plots
```

### Initialize Project

```bash
# Create project configuration
python test_cli.py init --project my_new_project
```

### Clean Old Runs

```bash
# Preview what would be deleted
python test_cli.py clean --days 30 --dry-run

# Delete runs older than 30 days
python test_cli.py clean --days 30

# Clean specific project
python test_cli.py clean --project my_project --days 30
```

### Check Version

```bash
python test_cli.py version
```

---

## 🐳 Docker Usage

### Using Docker Compose

```bash
# Start all services
docker-compose up -d

# This starts:
# - experiment-tracker container
# - MLflow server (http://localhost:5001)
# - Jupyter Lab (http://localhost:8888)

# Run training in container
docker-compose exec experiment-tracker python train.py

# View logs
docker-compose logs -f experiment-tracker

# Stop services
docker-compose down
```

### Building Custom Image

```bash
# Build image
docker build -t ml-experiment-tracker:latest .

# Run container
docker run -it --rm \
  -v $(pwd)/experiments:/workspace/experiments \
  -p 5000:5000 \
  ml-experiment-tracker:latest
```

### Docker Features

- Multi-stage build for smaller images
- Mounted volumes for data persistence
- MLflow server with SQLite backend
- Jupyter Lab for interactive development
- Network isolation
- Environment variables support

---

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────┐
│         ExperimentTracker (API)                 │
│  • Unified interface for all backends           │
│  • Auto backend selection with fallback         │
│  • Context manager for lifecycle                │
│  • Dual logging (backend + local)               │
└────────────────┬────────────────────────────────┘
                 │
      ┌──────────┼──────────┐
      │          │          │
 ┌────▼───┐ ┌───▼────┐ ┌──▼────┐
 │  W&B   │ │ MLflow │ │ Local │
 │ Backend│ │ Backend│ │Backend│
 └────────┘ └────────┘ └───────┘
      │          │          │
      ▼          ▼          ▼
   Cloud     Self-      File
   Service   Hosted    System
```

### Backend Priority

By default: **W&B → MLflow → Local**

The tracker tries each backend in order:
1. **Weights & Biases**: If `wandb` installed and configured
2. **MLflow**: If `mlflow` installed and tracking URI accessible
3. **Local**: Always available as final fallback

### Data Flow

```
1. User Code
   ├─> log_params()
   ├─> log_metrics()
   └─> log_artifact()
         │
         ▼
2. ExperimentTracker
   ├─> Validate inputs
   ├─> Update Run object
   └─> Send to Backend
         │
         ▼
3. Backend (W&B/MLflow/Local)
   ├─> Store in backend
   └─> Return confirmation
         │
         ▼
4. Local Copy (always)
   ├─> Save run.json
   ├─> Save artifacts
   └─> Update index
```

### File Structure

```
experiment_logs/
├── project_1/
│   ├── run_20241126_120000_abc123/
│   │   ├── run.json              # Run metadata
│   │   ├── metrics.json          # Time-series metrics
│   │   ├── artifacts/            # Saved artifacts
│   │   │   ├── model.pth
│   │   │   ├── plots/
│   │   │   └── logs/
│   │   └── report.html           # Generated report
│   └── run_20241126_130000_def456/
│       └── ...
└── project_2/
    └── ...
```

---

## 📊 Backend Comparison

| Feature | Local | MLflow | W&B |
|---------|-------|--------|-----|
| **Setup Required** | None | Install + Server | Install + Login |
| **Dependencies** | 0 | mlflow | wandb |
| **Storage** | Local files | Local/Server | Cloud |
| **Web UI** | Static HTML | ✓ Built-in | ✓ Cloud |
| **Collaboration** | Manual | Self-hosted | ✓ Built-in |
| **Model Registry** | Basic | ✓ Advanced | ✓ Advanced |
| **Real-time Viz** | ✗ | ✓ | ✓ |
| **Always Available** | ✓ | If running | If online |
| **Cost** | Free | Free (self-host) | Free tier |
| **Best For** | Quick tests | Team server | Cloud collab |

---

## 🎯 Complete Examples

### Image Classification

```python
from experiment_tracker import ExperimentTracker
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms

def train_image_classifier():
    # Initialize tracker
    with ExperimentTracker(
        project_name="image_classification",
        experiment_name="resnet50_cifar10",
        tags=["vision", "resnet", "cifar10"]
    ) as tracker:
        
        # Log hyperparameters
        config = {
            "model": "resnet50",
            "dataset": "cifar10",
            "learning_rate": 0.001,
            "batch_size": 128,
            "epochs": 50,
            "optimizer": "adam",
            "weight_decay": 1e-4
        }
        tracker.log_params(config)
        
        # Setup model and training
        model = models.resnet50(pretrained=True)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["learning_rate"]
        )
        
        # Training loop
        for epoch in range(config["epochs"]):
            # Training
            train_loss, train_acc = train_epoch(model, train_loader, optimizer)
            
            # Validation
            val_loss, val_acc = validate(model, val_loader)
            
            # Log metrics
            tracker.log_metrics({
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "learning_rate": optimizer.param_groups[0]['lr']
            }, step=epoch)
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                torch.save(model.state_dict(), f"checkpoint_epoch_{epoch+1}.pth")
                tracker.log_artifact(f"checkpoint_epoch_{epoch+1}.pth")
        
        # Save final model
        torch.save(model.state_dict(), "final_model.pth")
        tracker.log_model(model, "final_model.pth")
        
        print(f"✓ Training complete! Run ID: {tracker.get_run_id()}")

if __name__ == "__main__":
    train_image_classifier()
```

### Natural Language Processing

```python
from experiment_tracker import ExperimentTracker
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def train_text_classifier():
    with ExperimentTracker(
        project_name="nlp",
        experiment_name="bert_sentiment",
        tags=["nlp", "bert", "sentiment"]
    ) as tracker:
        
        # Configuration
        config = {
            "model": "bert-base-uncased",
            "task": "sentiment_analysis",
            "max_length": 128,
            "learning_rate": 2e-5,
            "batch_size": 32,
            "epochs": 3
        }
        tracker.log_params(config)
        
        # Model setup
        tokenizer = AutoTokenizer.from_pretrained(config["model"])
        model = AutoModelForSequenceClassification.from_pretrained(
            config["model"],
            num_labels=2
        )
        
        # Training loop
        for epoch in range(config["epochs"]):
            metrics = train_and_evaluate(model, tokenizer)
            
            tracker.log_metrics({
                "train_loss": metrics["train_loss"],
                "train_accuracy": metrics["train_accuracy"],
                "val_loss": metrics["val_loss"],
                "val_accuracy": metrics["val_accuracy"],
                "val_f1": metrics["val_f1"]
            }, step=epoch)
        
        # Save model
        model.save_pretrained("./bert_sentiment_model")
        tracker.log_artifact("./bert_sentiment_model", is_dir=True)

if __name__ == "__main__":
    train_text_classifier()
```

### Hyperparameter Tuning

```python
from experiment_tracker import ExperimentTracker
from sklearn.model_selection import ParameterGrid

def hyperparameter_search():
    # Define search space
    param_grid = {
        'learning_rate': [0.001, 0.0001, 0.00001],
        'batch_size': [32, 64, 128],
        'dropout': [0.1, 0.3, 0.5]
    }
    
    results = []
    
    # Grid search
    for params in ParameterGrid(param_grid):
        with ExperimentTracker(
            project_name="hyperparameter_search",
            experiment_name=f"lr_{params['learning_rate']}_bs_{params['batch_size']}",
            tags=["grid_search"]
        ) as tracker:
            
            # Log parameters
            tracker.log_params(params)
            
            # Train model
            val_accuracy = train_with_params(params)
            
            # Log final metric
            tracker.log_metrics({
                "final_val_accuracy": val_accuracy
            }, step=0)
            
            results.append({
                'run_id': tracker.get_run_id(),
                'params': params,
                'accuracy': val_accuracy
            })
    
    # Find best run
    best_run = max(results, key=lambda x: x['accuracy'])
    print(f"Best run: {best_run['run_id']}")
    print(f"Best params: {best_run['params']}")
    print(f"Best accuracy: {best_run['accuracy']:.4f}")

if __name__ == "__main__":
    hyperparameter_search()
```

---

## 🧪 Testing

### Run Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=experiment_tracker --cov-report=html

# Specific test file
pytest tests/test_tracker.py -v

# Integration tests only
pytest tests/test_integration.py -v -m integration

# Unit tests only
pytest -v -m unit
```

### Test Structure

```
tests/
├── conftest.py              # Pytest fixtures
├── test_config.py           # Configuration tests
├── test_run.py              # Run management tests
├── test_tracker.py          # ExperimentTracker tests
└── test_integration.py      # End-to-end workflow tests
```

---

## 🛠️ Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/harshithluc073/ml-experiment-tracker.git
cd ml-experiment-tracker

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

### Code Quality

```bash
# Format code
black src/ tests/

# Check linting
flake8 src/ tests/

# Sort imports
isort src/ tests/

# Type checking
mypy src/
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=experiment_tracker --cov-report=html

# View coverage report
# Open htmlcov/index.html
```

### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build docs (if using Sphinx)
cd docs
make html
```

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Quick Contribution Guide

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**
4. **Add tests** for new functionality
5. **Run tests**: `pytest`
6. **Format code**: `black src/ tests/`
7. **Commit**: `git commit -m 'Add amazing feature'`
8. **Push**: `git push origin feature/amazing-feature`
9. **Open a Pull Request**

---

## 📝 Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and release notes.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Inspired by [MLflow](https://mlflow.org/), [Weights & Biases](https://wandb.ai/), and [Neptune.ai](https://neptune.ai/)
- Built with ❤️ for the ML community
- Special thanks to all contributors

---

## 📧 Support & Contact

- **GitHub Issues**: [Report bugs or request features](https://github.com/harshithluc073/ml-experiment-tracker/issues)
- **GitHub Discussions**: [Ask questions or share ideas](https://github.com/harshithluc073/ml-experiment-tracker/discussions)
- **Email**: chitikeshiharshith@gmail.com

---

## 🌟 Star History

If you find this project useful, please consider giving it a star ⭐

---

## 📚 Additional Resources

- **Documentation**: [Full documentation](https://github.com/harshithluc073/ml-experiment-tracker/docs)
- **Examples**: [More examples](https://github.com/harshithluc073/ml-experiment-tracker/examples)
- **Tutorials**: [Video tutorials](https://github.com/harshithluc073/ml-experiment-tracker/tutorials)
- **Blog Posts**: [Blog](https://github.com/harshithluc073/ml-experiment-tracker/blog)

---

## 🎓 Citation

If you use this tool in your research, please cite:

```bibtex
@software{ml_experiment_tracker,
  author = {Harshith, Chitikeshi},
  title = {ML Experiment Tracker: Zero-friction ML experiment tracking},
  year = {2024},
  url = {https://github.com/harshithluc073/ml-experiment-tracker},
  version = {0.1.0}
}
```

---

## 🗺️ Roadmap

### Version 0.2.0 (Planned)
- [ ] Web UI dashboard
- [ ] REST API
- [ ] Experiment templates
- [ ] Advanced search and filtering

### Version 0.3.0 (Planned)
- [ ] Hyperparameter tuning integration
- [ ] Model registry with versioning
- [ ] Experiment scheduling
- [ ] Notification system

### Future
- [ ] Cloud storage backends (S3, GCS, Azure)
- [ ] Team collaboration features
- [ ] A/B testing framework
- [ ] AutoML integration

---

**Made with ❤️ by [Harshith](https://github.com/harshithluc073)**

*Happy Experimenting!* 🚀