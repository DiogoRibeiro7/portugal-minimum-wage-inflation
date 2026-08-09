# Summary

<!-- What does this change, and why? One paragraph. -->

Closes #

## Type of change

- [ ] Bug fix
- [ ] Data correction or new source
- [ ] New feature or analysis
- [ ] Refactor (no behaviour change)
- [ ] Documentation or manuscript
- [ ] Tooling / CI

## Effect on results

- [ ] This change does **not** alter any estimate, table, or figure.
- [ ] This change **does** alter results. Before/after values:

<!--
| Quantity | Before | After |
| -------- | ------ | ----- |
|          |        |       |
Explain which change is responsible.
-->

## Checks

- [ ] `make check` passes locally (ruff, ruff format, mypy strict, pytest).
- [ ] New or changed behaviour is covered by a test.
- [ ] Tests pass with no network access.
- [ ] New tunables live in `config/analysis.yaml`, not hard-coded.
- [ ] New variables are documented in `docs/data_dictionary.md`.
- [ ] New sources are registered in `config/sources.yaml` with provider, URL,
      licence, and checksum.
- [ ] No file under `data/raw/` was edited in place.
- [ ] No generated figure or table was hand-edited or committed.
- [ ] `CHANGELOG.md` updated under `[Unreleased]`.

## Notes for the reviewer

<!-- Anything non-obvious: a modelling judgement call, a source that required
interpretation, a deliberate deviation from the research design. -->
