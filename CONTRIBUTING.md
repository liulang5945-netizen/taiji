# Contributing to Taiji

Thank you for your interest in contributing! Taiji is a self-evolving LLM framework, and we welcome contributions of all kinds.

## How to Contribute

### Reporting Issues

- Search [existing issues](https://github.com/taiji-community/taiji/issues) before opening a new one
- Include: Python version, OS, steps to reproduce, expected vs actual behavior
- For training bugs, include: model size, config file, loss curves if available

### Pull Requests

1. Fork the repository and create a feature branch
2. Install development dependencies: `pip install -e ".[dev]"`
3. Make your changes, following the project's code style
4. Add tests for new functionality
5. Run the test suite: `python -m pytest tests/ -x -q`
6. Run the linter: `ruff check .`
7. Submit a PR with a clear description of the change

### Code Style

- Python code follows PEP 8
- Use type hints where practical
- Docstrings in reStructuredText or Google style
- Ruff linting is enforced in CI

### Commit Messages

- Use conventional commits: `feat:`, `fix:`, `chore:`, `docs:`, `refactor:`, `test:`
- Include a brief description of why the change was made

### Testing

- Tests are in the `tests/` directory
- Use the existing `conftest.py` fixtures for temp directories and cache management
- Test files should be named `test_*.py`
- Run `python -m pytest tests/ -v` to see detailed output

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cd frontend && npm install && cd ..
```

## License

By contributing, you agree that your contributions will be licensed under the GPL 3.0 License.
