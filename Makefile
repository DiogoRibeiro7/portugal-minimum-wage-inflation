.PHONY: help install hooks format lint typecheck test coverage check clean

help:
	@echo "install    Install dependencies from poetry.lock"
	@echo "hooks      Install pre-commit git hooks"
	@echo "format     Apply ruff formatting and safe fixes"
	@echo "lint       Ruff check and format verification"
	@echo "typecheck  mypy, strict, over src/"
	@echo "test       pytest"
	@echo "coverage   pytest with a terminal coverage report"
	@echo "check      lint + typecheck + test (what CI runs)"
	@echo "clean      Remove caches and build artefacts"

install:
	poetry install

hooks:
	poetry run pre-commit install

format:
	poetry run ruff check --fix src tests
	poetry run ruff format src tests

lint:
	poetry run ruff check src tests
	poetry run ruff format --check src tests

typecheck:
	poetry run mypy src

test:
	poetry run pytest

coverage:
	poetry run pytest --cov=pt_mw_inflation --cov-report=term-missing

check: lint typecheck test

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml dist build
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
