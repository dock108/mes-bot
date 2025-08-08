# Contributing to MES Bot

Thank you for your interest in contributing to the MES 0DTE Options Trading Bot! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Development Workflow](#development-workflow)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing Requirements](#testing-requirements)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)
- [Documentation](#documentation)
- [Security](#security)

## Code of Conduct

### Our Standards

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Accept responsibility and apologize for mistakes
- Prioritize the community's best interests

### Unacceptable Behavior

- Harassment, discrimination, or offensive comments
- Publishing private information without consent
- Trolling or insulting comments
- Any conduct deemed unprofessional

## Getting Started

1. **Fork the Repository**
   ```bash
   # Fork on GitHub, then:
   git clone https://github.com/YOUR_USERNAME/mes-bot.git
   cd mes-bot
   ```

2. **Set Up Development Environment**
   ```bash
   # Install Poetry
   curl -sSL https://install.python-poetry.org | python3 -

   # Install dependencies
   poetry install --with dev

   # Install pre-commit hooks
   poetry run pre-commit install
   ```

3. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

## Development Setup

### Required Tools

- Python 3.11+
- Poetry for dependency management
- Git for version control
- Docker (optional, for containerized testing)

### Environment Configuration

```bash
# Copy example configuration
cp .env.example .env.development

# Configure for development (paper trading)
TRADE_MODE=paper
LOG_LEVEL=DEBUG
```

### IDE Setup

#### VS Code
```json
{
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length", "100"],
  "python.testing.pytestEnabled": true,
  "editor.formatOnSave": true
}
```

#### PyCharm
- Enable Black formatter with line length 100
- Configure pytest as test runner
- Enable type checking with mypy

## Development Workflow

### 1. Before Starting Work

```bash
# Sync with upstream
git remote add upstream https://github.com/dock108/mes-bot.git
git fetch upstream
git checkout main
git merge upstream/main
```

### 2. During Development

```bash
# Run tests frequently
pytest tests/ -v

# Check code style
poetry run black app/ tests/
poetry run flake8 app/ tests/
poetry run mypy app/

# Or use pre-commit
poetry run pre-commit run --all-files
```

### 3. Commit Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```bash
# Format: <type>(<scope>): <subject>

feat(strategy): add volatility-based entry filter
fix(ib-client): handle connection timeout properly
docs(readme): update installation instructions
test(backtester): add edge case scenarios
refactor(risk): simplify position sizing logic
style(ui): improve dashboard layout
perf(ml): optimize feature calculation
chore(deps): update dependencies
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test additions/changes
- `refactor`: Code refactoring
- `style`: Code style changes
- `perf`: Performance improvements
- `chore`: Maintenance tasks

### 4. Testing Your Changes

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_strategy.py

# Run with coverage
pytest --cov=app --cov-report=html

# Run integration tests
pytest -m integration

# Run performance tests
pytest -m performance
```

## Code Style Guidelines

### Python Style

We use [PEP 8](https://pep8.org/) with the following specifications:

- **Line Length**: 100 characters maximum
- **Imports**: Sorted with `isort`
- **Formatting**: Enforced with `black`
- **Type Hints**: Required for all public functions
- **Docstrings**: Google style for all public modules/classes/functions

Example:
```python
from typing import Optional, List
import numpy as np

class StrategyEngine:
    """Manages trading strategy execution.

    Attributes:
        config: Strategy configuration parameters
        risk_manager: Risk management component
    """

    def calculate_signal(
        self,
        prices: np.ndarray,
        volumes: Optional[np.ndarray] = None
    ) -> float:
        """Calculate trading signal strength.

        Args:
            prices: Array of price data
            volumes: Optional array of volume data

        Returns:
            Signal strength between -1 and 1

        Raises:
            ValueError: If prices array is empty
        """
        if len(prices) == 0:
            raise ValueError("Prices array cannot be empty")

        # Implementation here
        return signal_strength
```

### Error Handling

```python
# Good
try:
    result = risky_operation()
except SpecificError as e:
    logger.error(f"Operation failed: {e}")
    # Handle gracefully
    return default_value

# Bad
try:
    result = risky_operation()
except:
    pass  # Never suppress errors silently
```

### Logging

```python
import logging

logger = logging.getLogger(__name__)

# Use appropriate log levels
logger.debug("Detailed information for debugging")
logger.info("General information")
logger.warning("Warning about potential issues")
logger.error("Error that needs attention")
logger.critical("Critical failure")
```

## Testing Requirements

### Test Coverage

- Minimum coverage: 80%
- New features must include tests
- Bug fixes must include regression tests

### Test Structure

```python
import pytest
from unittest.mock import Mock, patch

class TestStrategyEngine:
    """Test suite for StrategyEngine."""

    @pytest.fixture
    def engine(self):
        """Create strategy engine instance."""
        return StrategyEngine(config=test_config)

    def test_signal_calculation(self, engine):
        """Test signal calculation with valid data."""
        # Arrange
        prices = np.array([100, 101, 102])

        # Act
        signal = engine.calculate_signal(prices)

        # Assert
        assert -1 <= signal <= 1
        assert isinstance(signal, float)

    def test_signal_calculation_empty_prices(self, engine):
        """Test signal calculation raises error for empty prices."""
        with pytest.raises(ValueError, match="empty"):
            engine.calculate_signal(np.array([]))
```

### Performance Tests

```python
@pytest.mark.performance
def test_strategy_performance():
    """Ensure strategy calculations meet performance requirements."""
    engine = StrategyEngine()
    data = generate_large_dataset()

    start = time.time()
    engine.process(data)
    duration = time.time() - start

    assert duration < 0.1  # Must process in under 100ms
```

## Pull Request Process

### 1. Before Submitting

- [ ] Tests pass locally (`pytest`)
- [ ] Code follows style guidelines (`pre-commit run --all-files`)
- [ ] Documentation updated if needed
- [ ] Commit messages follow convention
- [ ] Branch is up-to-date with main

### 2. PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change)
- [ ] New feature (non-breaking change)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings generated
```

### 3. Review Process

1. Automated checks must pass
2. At least one maintainer review required
3. All review comments addressed
4. Final approval from maintainer
5. Squash and merge to main

## Issue Guidelines

### Bug Reports

Include:
- Python version
- Operating system
- Steps to reproduce
- Expected behavior
- Actual behavior
- Error messages/logs
- Relevant configuration

### Feature Requests

Include:
- Use case description
- Proposed solution
- Alternative solutions considered
- Impact on existing functionality

### Good First Issues

Look for issues labeled `good first issue` - these are ideal for newcomers.

## Documentation

### Code Documentation

- All public APIs must have docstrings
- Complex algorithms need inline comments
- Update README.md for user-facing changes
- Update technical docs for architectural changes

### Documentation Style

```python
def calculate_strangle_strikes(
    spot_price: float,
    implied_move: float,
    multipliers: tuple[float, float] = (1.25, 1.5)
) -> tuple[float, float]:
    """Calculate strike prices for strangle position.

    Uses implied move and multipliers to determine optimal
    strike placement for 0DTE strangle positions.

    Args:
        spot_price: Current underlying price
        implied_move: Expected daily move (as decimal)
        multipliers: Distance multipliers for strikes

    Returns:
        Tuple of (put_strike, call_strike)

    Example:
        >>> strikes = calculate_strangle_strikes(5000, 0.01)
        >>> print(strikes)
        (4937.5, 5062.5)
    """
    # Implementation
```

## Security

### Security Issues

**DO NOT** create public issues for security vulnerabilities. Instead:

1. Email security concerns to maintainers
2. Include detailed description
3. Provide steps to reproduce if possible
4. Allow time for patch before disclosure

### Security Best Practices

- Never commit credentials or API keys
- Use environment variables for sensitive data
- Validate all user inputs
- Follow principle of least privilege
- Keep dependencies updated

## Getting Help

- **Discord**: [Join our server](https://discord.gg/mes-bot)
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check [docs/](docs/) folder
- **Examples**: See [examples/](examples/) folder

## Recognition

Contributors are recognized in:
- [CHANGELOG.md](CHANGELOG.md) for their contributions
- GitHub contributors page
- Special thanks in release notes

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to make this project better! 🚀
