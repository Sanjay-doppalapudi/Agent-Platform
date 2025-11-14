# Contributing to Agent Platform

Thank you for your interest in contributing to Agent Platform! 🎉

We welcome contributions of all kinds: bug reports, feature requests, documentation improvements, and code contributions.

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher
- Poetry for dependency management
- Docker for sandbox testing
- Git

### Development Setup

1. **Fork and Clone**

```bash
git clone https://github.com/YOUR_USERNAME/agent-platform.git
cd agent-platform
```

2. **Install Dependencies**

```bash
poetry install
```

3. **Set Up Environment**

```bash
cp .env.example .env
# Add your API keys to .env
```

4. **Run Tests**

```bash
poetry run pytest tests/unit/ -v
```

## 📝 Development Workflow

### Branch Naming Convention

- `feature/your-feature-name` - New features
- `fix/issue-description` - Bug fixes
- `docs/what-you-are-documenting` - Documentation
- `refactor/what-you-are-refactoring` - Code refactoring

### Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Maintenance tasks

**Examples:**
```bash
feat(tools): add WebSearchTool for internet queries
fix(sandbox): prevent memory leak in container cleanup
docs(api): add examples for streaming endpoints
test(agent): add integration tests for multi-step tasks
```

## 🧪 Testing

### Running Tests

```bash
# Unit tests (fast, no API calls)
poetry run pytest tests/unit/ -v

# Integration tests (requires API keys)
export OPENAI_API_KEY=sk-...
poetry run pytest tests/integration/ -v

# Security tests
poetry run pytest tests/security/ -v

# All tests with coverage
poetry run pytest --cov=src --cov-report=html
```

### Test Requirements

- **Unit tests** for all new functions/classes
- **Integration tests** for new features
- **Security tests** for anything touching user input or sandboxing
- Minimum **90% code coverage** for new code

### Writing Tests

```python
import pytest

class TestMyFeature:
    """Test suite for my new feature."""
    
    def test_basic_functionality(self):
        """Test the basic use case."""
        # Arrange
        ...
        
        # Act
        ...
        
        # Assert
        ...
```

## 📐 Code Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length**: 100 characters
- **Imports**: Organized by stdlib, third-party, local
- **Type hints**: Required for all public APIs
- **Docstrings**: Google style for all public functions

### Code Formatting

```bash
# Format code
poetry run black src/ tests/

# Check formatting
poetry run black --check src/ tests/

# Lint
poetry run ruff check src/ tests/

# Type check
poetry run mypy src/
```

### Pre-commit Hooks

```bash
# Install hooks
poetry run pre-commit install

# Run manually
poetry run pre-commit run --all-files
```

## 📚 Documentation

### Docstring Example

```python
def my_function(param1: str, param2: int) -> bool:
    """
    Brief description of what the function does.
    
    More detailed explanation if needed. Can span multiple
    paragraphs.
    
    Args:
        param1: Description of param1.
        param2: Description of param2.
    
    Returns:
        Description of return value.
    
    Raises:
        ValueError: When param1 is empty.
    
    Example:
        >>> my_function("test", 42)
        True
    """
    ...
```

### Documentation Updates

Update relevant documentation when:
- Adding new features
- Changing public APIs
- Adding new configuration options
- Fixing bugs that affect documented behavior

## 🔒 Security

### Reporting Vulnerabilities

**DO NOT** open public issues for security vulnerabilities.

Email: security@agent-platform.dev

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### Security Considerations

When contributing code that:
- Handles user input → Add input validation
- Executes code → Use sandbox
- Accesses files → Validate paths
- Makes network requests → Add timeouts and limits

## 📋 Pull Request Process

### Before Submitting

- [ ] Tests pass locally
- [ ] Code is formatted (black, ruff)
- [ ] Type checks pass (mypy)
- [ ] Documentation updated
- [ ] CHANGELOG.md updated (for features/fixes)
- [ ] Commit messages follow convention

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe testing done

## Checklist
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Code formatted
- [ ] No new warnings
```

### Review Process

1. Automated checks must pass
2. At least one maintainer review required
3. All comments addressed
4. Squash and merge (maintainers will handle this)

## 🎯 Areas We Need Help

### High Priority

- [ ] Additional tool implementations (web_tools, git_tools)
- [ ] WebSocket streaming support
- [ ] React frontend demo app
- [ ] Performance optimizations

### Medium Priority

- [ ] Additional LLM provider integrations
- [ ] More example applications
- [ ] Tutorial videos/guides
- [ ] Benchmark suite expansion

### Documentation

- [ ] API usage examples
- [ ] Architecture deep-dives
- [ ] Best practices guide
- [ ] Troubleshooting guide

## 🤝 Community

- **GitHub Discussions**: Ask questions, share ideas
- **Issues**: Bug reports and feature requests
- **Pull Requests**: Code contributions

## 📜 License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

---

**Thank you for contributing to Agent Platform!** 🙏
