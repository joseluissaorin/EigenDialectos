.PHONY: all install test test-unit test-integration test-coverage lint typecheck validate clean

all: install test lint validate

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

test-unit:
	pytest tests/ -v --tb=short --ignore=tests/integration

test-integration:
	pytest tests/integration/ -v --tb=long

test-coverage:
	pytest tests/ --cov=src/eigendialectos --cov-report=html

lint:
	ruff check src/ tests/

typecheck:
	mypy src/eigendialectos/

validate:
	python scripts/validate_project.py

clean:
	rm -rf outputs/* data/synthetic/* __pycache__ .pytest_cache

run-exp-%:
	python -m eigendialectos.experiments.runner --experiment=$*
