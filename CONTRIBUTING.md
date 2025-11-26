# Contributing to ML Experiment Tracker

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Development Setup

1. **Fork and clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/ml-experiment-tracker.git
cd ml-experiment-tracker
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install in development mode**

```bash
pip install -e ".[dev]"
```

4. **Install pre-commit hooks** (optional but recommended)

```bash
pip install pre-commit
pre-commit install
```

## Development Workflow

### Making Changes

1. **Create a new branch**

```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes**

- Write clear, concise code
- Follow PEP 8 style guidelines
- Add docstrings to functions and classes
- Update documentation as needed

3. **Run tests**

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=experiment_tracker

# Run specific tests
pytest tests/test_config.py -v
```

4. **Run linting and formatting**

```bash
# Format code with black
black src/ tests/

# Check with flake8
flake8 src/ tests/

# Type checking (optional)
mypy src/
```

### Commit Guidelines

Use clear, descriptive commit messages:

```
<type>: <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Test additions or changes
- `chore`: Build process or auxiliary tool changes

**Example:**

```
feat: Add support for custom metric aggregation

- Implement MetricAggregator class
- Add min, max, mean, median aggregation
- Update documentation with examples

Closes #123
```

### Pull Request Process

1. **Update documentation**
   - Update README.md if needed
   - Add/update docstrings
   - Update CHANGELOG.md

2. **Add tests**
   - Unit tests for new functionality
   - Integration tests if applicable
   - Aim for >80% code coverage

3. **Create pull request**
   - Use a clear, descriptive title
   - Reference related issues
   - Describe changes in detail
   - Include screenshots if UI-related

4. **Code review**
   - Address reviewer feedback
   - Keep discussions focused and professional
   - Make requested changes promptly

## Testing

### Running Tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/ -m unit

# Integration tests only
pytest tests/ -m integration

# With coverage
pytest --cov=experiment_tracker --cov-report=html

# Specific test file
pytest tests/test_config.py -v
```

### Writing Tests

- Use pytest fixtures for setup/teardown
- Test edge cases and error conditions
- Keep tests isolated and independent
- Use descriptive test names

**Example:**

```python
def test_log_param_with_valid_input():
    """Test logging a parameter with valid input."""
    tracker = ExperimentTracker(project_name="test")
    tracker.log_param("learning_rate", 0.001)
    assert tracker.run.params["learning_rate"] == 0.001
```

## Code Style

### Python Style Guide

- Follow PEP 8
- Use type hints where applicable
- Maximum line length: 100 characters
- Use meaningful variable names

### Docstring Format

Use Google-style docstrings:

```python
def function_name(param1: str, param2: int) -> bool:
    """
    Brief description of function.
    
    Longer description if needed.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: Description of when this is raised
    
    Example:
        >>> function_name("test", 42)
        True
    """
    pass
```

## Documentation

### Updating Documentation

- Update README.md for user-facing changes
- Update docstrings for API changes
- Add examples for new features
- Update guides in docs/ directory

### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation (if using Sphinx)
cd docs
make html
```

## Issue Guidelines

### Reporting Bugs

Include:
- Clear, descriptive title
- Steps to reproduce
- Expected vs actual behavior
- Python version and OS
- Error messages/stack traces
- Code snippet (if applicable)

### Suggesting Features

Include:
- Clear description of the feature
- Use cases and benefits
- Possible implementation approach
- Examples of usage

## Release Process

(For maintainers)

1. **Update version**
   - Update version in `setup.py` or `pyproject.toml`
   - Update CHANGELOG.md

2. **Create release**
   ```bash
   git tag -a v0.2.0 -m "Release v0.2.0"
   git push origin v0.2.0
   ```

3. **Build and publish**
   ```bash
   python -m build
   twine upload dist/*
   ```

## Community

- Be respectful and inclusive
- Help others when you can
- Provide constructive feedback
- Celebrate contributions

## Questions?

- Open an issue for bug reports or feature requests
- Start a discussion for questions
- Email: chitikeshiharshith@gmail.com

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing! 🎉