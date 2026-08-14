.PHONY: help install hooks format lint typecheck test coverage check paper clean

help:
	@echo "install    Install dependencies from poetry.lock"
	@echo "hooks      Install pre-commit git hooks"
	@echo "format     Apply ruff formatting and safe fixes"
	@echo "lint       Ruff check and format verification"
	@echo "typecheck  mypy, strict, over src/"
	@echo "test       pytest"
	@echo "coverage   pytest with a terminal coverage report"
	@echo "check      lint + typecheck + test (what CI runs)"
	@echo "paper      Rebuild every dataset, regenerate all outputs, compile the manuscript"
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

# The manuscript imports generated figures and tables, so the data pipeline
# has to run before LaTeX does.
paper:
	poetry run ptmw data download-sources
	# Reports raw inputs whose bytes moved since the manifest was frozen.
	# Not strict: an upstream revision should be seen and decided on, not
	# silently block a rebuild.
	poetry run ptmw data verify-inputs
	poetry run ptmw data ine-cpi
	poetry run ptmw data regional-employment
	poetry run ptmw build minimum-wage
	poetry run ptmw build macro
	# The October 2015 round, and a window starting in 2016, so exposure
	# precedes every shock it is used to estimate. The default round post-
	# dates the window and the builder refuses it.
	poetry run ptmw build regional-exposure --bite-period 2015-10 --first-shock-year 2016
	poetry run ptmw analyse macro
	poetry run ptmw analyse pass-through
	poetry run ptmw analyse exposure-design
	poetry run ptmw analyse exposure-robustness
	# LaTeX resolves citations and cross-references across passes, and bibtex
	# runs between them. A single pass leaves the bibliography empty and every
	# citation unresolved, which is how the references silently went missing.
	cd report && pdflatex -interaction=nonstopmode -halt-on-error main.tex
	cd report && bibtex main
	cd report && pdflatex -interaction=nonstopmode -halt-on-error main.tex
	cd report && pdflatex -interaction=nonstopmode -halt-on-error main.tex
	@cd report && ! grep -qE "Citation .* undefined|undefined references" main.log 		|| (echo "ERROR: unresolved citations remain" && exit 1)

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml dist build
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
