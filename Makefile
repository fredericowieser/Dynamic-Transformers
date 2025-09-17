.PHONY: install test test-fast test-all lint format clean dev-install

install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

test-fast:
	pytest tests/ -v -m "not slow"

test-unit:
	pytest tests/test_layers.py tests/test_routers.py tests/test_blocks.py tests/test_models.py -v

test-integration:
	pytest tests/test_training_integration.py tests/test_evaluation_integration.py -v

test-all:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint:
	ruff check src tests

format:
	ruff format src tests

type-check:
	mypy src --ignore-missing-imports --no-strict-optional

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +

run-tests-parallel:
	pytest tests/ -v -n auto

benchmark:
	pytest tests/ -v -m benchmark --benchmark-only

ci-test:
	@echo "Running CI tests..."
	make lint
	make test-all

pre-commit:
	@echo "Running pre-commit checks..."
	make format
	make lint
	make test-fast