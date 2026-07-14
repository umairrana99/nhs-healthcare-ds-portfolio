.PHONY: install test lint format typecheck check clean

install:  ## Install package + dev tooling in editable mode
	python -m pip install -e ".[dev]"
	pre-commit install

test:  ## Run the test suite
	python -m pytest

lint:  ## Ruff lint
	python -m ruff check .

format:  ## Black + Ruff autofix
	python -m black .
	python -m ruff check --fix .

typecheck:  ## MyPy static type check
	python -m mypy

check: lint typecheck test  ## Run all quality gates

clean:  ## Remove caches and build artefacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
