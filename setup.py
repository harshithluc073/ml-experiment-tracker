from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ml-experiment-tracker",
    version="0.1.0",
    author="Harshith",
    author_email="chitikeshiharshith@gmail.com",
    description="Zero-friction ML experiment tracking with intelligent fallbacks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/harshithluc073/ml-experiment-tracker",
    project_urls={
        "Bug Tracker": "https://github.com/harshithluc073/ml-experiment-tracker/issues",
        "Documentation": "https://github.com/harshithluc073/ml-experiment-tracker/docs",
        "Source Code": "https://github.com/harshithluc073/ml-experiment-tracker",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "pyyaml>=6.0",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "plotly>=5.0.0",
        "jinja2>=3.0.0",
        "click>=8.0.0",
        "psutil>=5.8.0",
        "gitpython>=3.1.0",
    ],
    extras_require={
        "mlflow": ["mlflow>=2.0.0"],
        "wandb": ["wandb>=0.13.0"],
        "all": [
            "mlflow>=2.0.0",
            "wandb>=0.13.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.12.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.990",
            "isort>=5.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "exp-track=experiment_tracker.cli:main",
        ],
    },
)