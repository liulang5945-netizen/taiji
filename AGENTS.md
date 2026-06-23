# Taiji Project Rules

## Code Style
- Use `black` for Python formatting, `ruff` for linting
- Type hints required for all public functions
- Follow existing module structure in `taiji/`, `api/`, `frontend/`

## Testing
- Always run tests after changes: `python -m pytest tests/ -v`
- Write tests for new features in `tests/` directory

## Documentation
- Use Context7 when you need to look up library docs (FastAPI, Vue, PyTorch, etc.)
- Use `gh_grep` to search for code examples on GitHub when unsure about patterns

## MCP Usage
- Use `context7` tools when searching for documentation
- Use `gh_grep` tool when you need to find code examples from GitHub
- Use `sequential-thinking` for complex multi-step problems that need structured reasoning
- Use `memory` to persist important findings across sessions
- Use `filesystem` for secure file operations within the project

## Git
- Follow conventional commits: `feat(scope): description`
- Scopes: `core`, `agent`, `api`, `frontend`, `life`, `train`, `mcp`, `safety`, `docker`

## HuggingFace
- Use `huggingface-ops` skill for model/dataset operations
- Models cached in `model_cache/`
- Training data in `taiji_data/`

## Monitoring
- Use `monitoring` skill for Prometheus/Grafana operations
- Check `/metrics` endpoint for application metrics
- Use `promql` for metric queries
