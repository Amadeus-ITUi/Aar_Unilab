CONDA_ENV ?= unilab_cuda
TORCH_VARIANT ?= cu128

.PHONY: setup
setup:
	bash install_conda_environment.txt $(CONDA_ENV) $(TORCH_VARIANT)

.PHONY: install
install:
	python -m pip install --no-build-isolation -e ".[dev]"

.PHONY: install-completion
install-completion:
	python -m unilab.tools.completion install

.PHONY: format
format:
	python -m ruff format
	python -m ruff check --fix

.PHONY: lint
lint:
	python -m ruff format --check .
	python -m ruff check .

.PHONY: type
type:
	python -m mypy src/unilab

.PHONY: check
check: lint type

.PHONY: test
test:
	python -m pytest -m "not slow"

.PHONY: test-cov
test-cov:
	python -m pytest -m "not slow" --cov=src/unilab --cov-report=term-missing

.PHONY: test-slow
test-slow:
	python -m pytest -m "slow" -v

.PHONY: test-all
test-all: check test-cov

.PHONY: clean
clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov
	rm -f .coverage train_rsl_rl.log
	find src tests scripts -type d -name "__pycache__" -prune -exec rm -rf {} +
	find src tests scripts -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete
