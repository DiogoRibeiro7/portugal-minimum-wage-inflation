"""Consistency tests between the manuscript and the generated numbers.

The reproducibility rule is that no result is copied into the paper by hand.
These tests enforce it mechanically: the prose may only cite quantities the
pipeline actually defines, and it may not contain a hard-coded figure where a
generated macro belongs.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPORT = Path(__file__).resolve().parents[1] / "report"
SECTIONS = REPORT / "sections"

#: Commands defined by LaTeX, the preamble, or the generated tables, which the
#: sections may use without the macro file defining them.
_STRUCTURAL = re.compile(
    r"^(section|subsection|subsubsection|label|ref|eqref|cite|citep|citet|input|includegraphics"
    r"|begin|end|centering|caption|texttt|emph|textbf|textit|item|qquad|quad|text|pi|,|\\)$"
)

_MACRO_USE = re.compile(r"\\([A-Za-z]+)\{?\}?")
_MACRO_DEF = re.compile(r"\\newcommand\{\\([A-Za-z]+)\}")


def _generated_macros() -> set[str]:
    """Names defined by the generated headline-macro file."""
    path = REPORT / "tables" / "headline_macros.tex"
    if not path.exists():
        pytest.skip("headline macros not generated; run 'ptmw analyse macro'")
    return set(_MACRO_DEF.findall(path.read_text(encoding="utf-8")))


def _section_text() -> dict[str, str]:
    """Return the text of every manuscript section."""
    return {path.name: path.read_text(encoding="utf-8") for path in sorted(SECTIONS.glob("*.tex"))}


def test_every_cited_quantity_is_generated() -> None:
    """A section may not reference a macro the pipeline does not define.

    This is what stops the paper quoting a number that no longer exists after
    the data changes: the reference fails here rather than silently rendering
    as an undefined command.
    """
    defined = _generated_macros()
    unknown: dict[str, set[str]] = {}

    for name, text in _section_text().items():
        used = {
            command
            for command in _MACRO_USE.findall(text)
            # Only capitalised, non-structural commands are candidate quantities.
            if command[:1].isupper() and not _STRUCTURAL.match(command)
        }
        missing = used - defined
        if missing:
            unknown[name] = missing

    assert not unknown, f"sections cite undefined generated quantities: {unknown}"


@pytest.mark.parametrize("section", ["results.tex"])
def test_results_section_uses_macros_not_literals(section: str) -> None:
    """The results prose must not hard-code the numbers it reports.

    A decimal in the results text is almost always a transcribed estimate, which
    is exactly what the reproducibility rules forbid. Years and equation numbers
    are integers and are allowed.
    """
    text = (SECTIONS / section).read_text(encoding="utf-8")
    body = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", "", text, flags=re.DOTALL)
    body = re.sub(r"\\begin\{table\}.*?\\end\{table\}", "", body, flags=re.DOTALL)
    body = re.sub(r"\\begin\{equation\}.*?\\end\{equation\}", "", body, flags=re.DOTALL)
    body = re.sub(r"\$[^$]*\$", "", body)

    decimals = re.findall(r"\b\d+\.\d+\b", body)
    assert not decimals, f"{section} hard-codes numeric results: {decimals}"


def test_results_section_states_what_is_not_established() -> None:
    """The descriptive layer must not be left to read as a causal claim."""
    text = (SECTIONS / "results.tex").read_text(encoding="utf-8")
    assert "not establish" in text or "does not mean" in text


def test_generated_tables_carry_a_do_not_edit_banner() -> None:
    """Generated artefacts must say they are generated."""
    tables = REPORT / "tables"
    if not tables.exists():
        pytest.skip("tables not generated; run 'ptmw analyse macro'")

    generated = list(tables.glob("*.tex"))
    if not generated:
        pytest.skip("no generated tables present")

    for path in generated:
        assert "Do not edit" in path.read_text(encoding="utf-8"), path.name
