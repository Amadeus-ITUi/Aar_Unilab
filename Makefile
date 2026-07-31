.PHONY: sync
sync:
	uv sync

.PHONY: setup
setup:
	uv sync
	uv run --no-sync unilab-complete install

.PHONY: install-completion
install-completion:
	uv run --no-sync unilab-complete install

.PHONY: format
format:
	uv run ruff format
	uv run ruff check --fix

.PHONY: lint
lint:
	uv run ruff format --check .
	uv run ruff check .

.PHONY: type
type:
	uv run mypy src/unilab
	uv run pyright

.PHONY: check
check: lint type

.PHONY: test
test:
	uv run pytest -m "not slow"

.PHONY: test-cov
test-cov:
	uv run pytest -m "not slow" --cov=src/unilab --cov-report=term-missing

.PHONY: test-slow
test-slow:
	uv run pytest -m "slow" -v

.PHONY: test-all
test-all: check test-cov

.PHONY: clean
clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov
	rm -f .coverage train_rsl_rl.log
	find src tests scripts -type d -name "__pycache__" -prune -exec rm -rf {} +
	find src tests scripts -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete
