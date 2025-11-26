"""Setup configuration for ML Experiment Tracker."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read version
version = "0.1.0"

setup(
    name="ml-experiment-tracker",
    version=version,
    author="Harshith",
    author_email="chitikeshiharshith@gmail.com",
    description="Zero-friction ML experiment tracking with intelligent fallbacks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/harshithluc073/ml-experiment-tracker",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pyyaml>=5.4.0",
        "psutil>=5.8.0",
    ],
    extras_require={
        "all": [
            "mlflow>=2.0.0",
            "wandb>=0.12.0",
            "matplotlib>=3.5.0",
            "numpy>=1.21.0",
        ],
        "mlflow": ["mlflow>=2.0.0"],
        "wandb": ["wandb>=0.12.0"],
        "plotting": ["matplotlib>=3.5.0", "numpy>=1.21.0"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
    entry_points={
        "console_scripts": [
            "experiment-tracker=experiment_tracker.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)