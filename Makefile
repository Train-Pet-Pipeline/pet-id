.PHONY: setup test lint clean test-nogpu

setup:
	python -m pip install -e ".[dev,all]"

test:
	pytest tests/ -v --cov=purrai_core --cov-report=term-missing

test-nogpu:
	pytest tests/ -v -m "not gpu"

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/
	mypy src/purrai_core

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf build/ dist/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
