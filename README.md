# ML Experiment Tracker

**Zero-friction ML experiment tracking with intelligent fallbacks and offline-first reporting**

## 🎯 Overview

A robust experiment tracking system that embeds directly into model training scripts, enforcing discipline in iterative ML development. Leverages free tools like Weights & Biases or local MLflow with seamless fallbacks to ensure every run is logged without user intervention.

## ✨ Key Features

- **Zero-Config Setup**: Detects runtime environment and provides tailored setup instructions
- **Intelligent Fallbacks**: W&B → MLflow → Local JSON/CSV storage
- **Universal Accessibility**: Works with free tools, no authentication barriers
- **Offline-First**: Static HTML reports for instant analysis without live servers
- **Complete Reproducibility**: Docker Compose integration for full stack deployment
- **Automatic Artifact Preservation**: Models, predictions, hyperparameters all captured
- **Run Continuation**: Links related experiments across sessions
- **Collaborative**: Auto-generates guest-viewable links for sharing

## 🚀 Quick Start
```python
from experiment_tracker import ExperimentTracker

# Initialize tracker (auto-detects best backend)
tracker = ExperimentTracker(project_name="my_ml_project")

# Log parameters
tracker.log_params({"learning_rate": 0.001, "epochs": 10})

# Train your model
for epoch in range(10):
    loss = train_epoch()
    tracker.log_metrics({"loss": loss}, step=epoch)

# Save artifacts
tracker.save_artifact(model, "model.pkl")
tracker.finish()  # Generates HTML report automatically
```

## 📦 Installation
```bash
pip install ml-experiment-tracker
```

## 🏗️ Project Status

**Current Version**: 0.1.0 (In Development)

This project is under active development. Check back for updates!

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Harshith**
- GitHub: [@harshithluc073](https://github.com/harshithluc073)
- Email: chitikeshiharshith@gmail.com

## 🙏 Acknowledgments

Built with inspiration from MLflow, Weights & Biases, and the broader MLOps community.