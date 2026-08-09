.PHONY: install lint typecheck test check

install:
	poetry install

lint:
	poetry run ruff check src tests
	poetry run ruff format --check src tests

typecheck:
	poetry run mypy src

test:
	poetry run pytest

check: lint typecheck test
