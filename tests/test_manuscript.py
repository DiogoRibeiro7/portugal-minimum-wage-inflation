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

from tests.macro_names import definable_macros, written_macro_files

REPORT = Path(__file__).resolve().parents[1] / "report"
SECTIONS = REPORT / "sections"

#: Commands defined by LaTeX, the preamble, or the generated tables, which the
#: sections may use without the macro file defining them.
_STRUCTURAL = re.compile(
    r"^(section|subsection|subsubsection|paragraph|label|ref|eqref|cite|citep|citet|input"
    r"|includegraphics|begin|end|centering|caption|texttt|emph|textbf|textit|item|qquad|quad"
    r"|text|pi|toprule|midrule|bottomrule"
    # Greek letters are capitalised commands too, and are not generated
    # quantities; without them every equation reads as an undefined citation.
    r"|Delta|Gamma|Lambda|Omega|Sigma|Phi|Psi|Theta|Xi"
    r"|,|\\)$"
)

_MACRO_USE = re.compile(r"\\([A-Za-z]+)\{?\}?")
_MACRO_DEF = re.compile(r"\\(?:new|provide)command\{\\([A-Za-z]+)\}")
_BIB_ENTRY = re.compile(r"@\w+\{([^,]+),")
# natbib commands may be capitalised (\Citet) and may carry optional arguments
# before the key list (\citep[see][p. 3]{key}). Missing either form would let a
# citation with no bibliography entry pass the very check meant to catch it.
_CITE = re.compile(r"\\[Cc]ite[a-zA-Z]*(?:\[[^\]]*\])*\{([^}]+)\}")


def _generated_macros(tmp_path: Path) -> set[str]:
    """Names the pipeline's macro writers can define.

    Derived by running the writers, not by reading ``report/tables``. Those are
    build outputs and are not committed, so reading them made this check skip on
    every clean checkout --- which is to say, on every continuous-integration
    run. It passed only where the paper had already been built by hand.

    Running the writers also answers a question the directory could not: a macro
    surviving in a stale local file but no longer emitted by anything is now
    correctly reported as undefined.
    """
    return definable_macros(tmp_path)


def _section_text() -> dict[str, str]:
    """Return the text of every manuscript section."""
    return {path.name: path.read_text(encoding="utf-8") for path in sorted(SECTIONS.glob("*.tex"))}


def test_every_cited_quantity_is_generated(tmp_path: Path) -> None:
    """A section may not reference a macro the pipeline does not define.

    This is what stops the paper quoting a number that no longer exists after
    the data changes: the reference fails here rather than silently rendering
    as an undefined command.
    """
    defined = _generated_macros(tmp_path)
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


@pytest.mark.parametrize("section", ["results.tex", "robustness.tex", "introduction.tex"])
def test_empirical_sections_use_macros_not_literals(section: str) -> None:
    """Empirical prose must not hard-code the numbers it reports.

    Restricting this to one file is how a transcribed estimate got into the
    identification section: the rule was enforced where results were expected
    and not where they actually appeared.

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


#: Ways of asserting that the region-by-industry exposure measure cannot be
#: built. Each is a claim the repository made and then disproved by building it,
#: so the phrasing is pinned rather than the sentiment: a section may say the
#: measure is uninformative, assumption-laden or not predetermined, but it may
#: not say the ingredients do not exist.
_UNAVAILABILITY_CLAIMS = (
    "no Portuguese source publishes",
    "that no source publishes",
    "no source crosses",
    "is not made here",
    "not reported in this version",
    "required by that design is available",
)


def test_no_section_claims_the_exposure_measure_cannot_be_built() -> None:
    """The prose may not say the exposure data is missing while the code uses it.

    This is the defect that recurred: a conclusion drawn from the labour-ministry
    publications alone was generalised to every source, the manuscript was
    written on it, and the manuscript kept saying so after Eurostat's regional
    accounts had been wired into the pipeline and the measure built. Prose and
    pipeline drifted apart silently because nothing compared them.

    The test is deliberately tied to the builder existing. If the shift-share
    construction is ever removed, the claim becomes true again and this stops
    applying.

    Whitespace is collapsed before matching. LaTeX prose is hard-wrapped, so a
    phrase of more than two words is as likely as not to be split by a newline,
    and a raw substring search would pass on exactly the sentences it is meant
    to catch.
    """
    pytest.importorskip("pt_mw_inflation.processing.exposure")
    from pt_mw_inflation.processing.exposure import shift_share_exposure

    assert callable(shift_share_exposure)

    offending: dict[str, list[str]] = {}
    for name, text in _section_text().items():
        flattened = " ".join(text.split())
        # Scoped to the sentence, not the section. Several of these phrasings
        # are ordinary English -- "is not made here" could one day describe an
        # unrelated choice -- and a section-wide match would fail on prose that
        # is perfectly honest about something else.
        for sentence in re.split(r"(?<=\.)\s+", flattened):
            if not re.search(r"exposure|coverage|bite|region|pass-through", sentence, re.I):
                continue
            found = [claim for claim in _UNAVAILABILITY_CLAIMS if claim in sentence]
            if found:
                offending.setdefault(name, []).extend(found)

    assert not offending, (
        "sections assert the exposure measure is unbuildable while the pipeline "
        f"builds it: {offending}"
    )


def test_results_section_states_what_is_not_established() -> None:
    """The descriptive layer must not be left to read as a causal claim."""
    text = (SECTIONS / "results.tex").read_text(encoding="utf-8")
    assert "not establish" in text or "does not mean" in text


def test_generated_tables_carry_a_do_not_edit_banner(tmp_path: Path) -> None:
    """Generated artefacts must say they are generated.

    Checked on what the writers emit rather than on whatever is in the working
    tree, so the rule holds on a clean checkout too.
    """
    for path in written_macro_files(tmp_path):
        assert "Do not edit" in path.read_text(encoding="utf-8"), path.name


def test_every_writer_defines_something() -> None:
    """A writer that emits no macro would silently weaken the check above.

    The guard is only as complete as the set of writers it exercises, so an
    empty one has to be an error rather than a quiet absence.
    """
    import inspect

    from pt_mw_inflation.analysis import outputs

    writers = {
        name
        for name, _ in inspect.getmembers(outputs, inspect.isfunction)
        if name.startswith("write_") and name.endswith("_macros")
    }
    exercised = {
        "write_headline_macros",
        "write_identification_macros",
        "write_exposure_macros",
        "write_exposure_design_macros",
        "write_regional_premium_macros",
        "write_seasonality_macros",
    }
    assert writers == exercised, (
        "tests/macro_names.py does not exercise every macro writer; "
        f"unexercised: {sorted(writers - exercised)}"
    )


def test_every_citation_key_exists_in_the_bibliography() -> None:
    """A cited key with no entry silently drops from the reference list.

    LaTeX reports this as a warning, not an error, so a single-pass build
    produces a paper with an empty bibliography and exit code zero. That is how
    the references went missing here, so the check is made independently of the
    build.
    """
    bibliography = REPORT / "references.bib"
    if not bibliography.exists():
        pytest.skip("no bibliography present")

    defined = set(_BIB_ENTRY.findall(bibliography.read_text(encoding="utf-8")))
    cited: set[str] = set()
    for text in _section_text().values():
        for group in _CITE.findall(text):
            cited |= {key.strip() for key in group.split(",")}

    assert cited, "no section cites anything; the reference list would be empty"
    assert not cited - defined, f"cited but absent from the bibliography: {sorted(cited - defined)}"


def test_bibliography_entries_are_complete() -> None:
    """An entry missing title, author or year renders as a stub."""
    bibliography = REPORT / "references.bib"
    if not bibliography.exists():
        pytest.skip("no bibliography present")

    text = bibliography.read_text(encoding="utf-8")
    incomplete = []
    for block in re.split(r"(?=@)", text):
        match = _BIB_ENTRY.match(block.strip())
        if match is None:
            continue
        for field in ("title", "author", "year"):
            if re.search(field + r"\s*=", block) is None:
                incomplete.append(f"{match.group(1)}:{field}")

    assert not incomplete, f"incomplete bibliography entries: {incomplete}"


#: Which pipeline stage produces each artefact the manuscript imports. Declared
#: rather than discovered, because the failure this guards against is an
#: artefact nothing produces: LaTeX would report it as a missing file on a clean
#: build, but a stale copy in the working tree hides that from whoever made the
#: change.
_ARTEFACT_PRODUCER = {
    "tables/headline_macros": "ptmw analyse macro",
    "tables/regime_summary": "ptmw analyse macro",
    "figures/real_wage_productivity.pdf": "ptmw analyse macro",
    "figures/policy_residual.pdf": "ptmw analyse macro",
    "figures/wage_to_productivity.pdf": "ptmw analyse macro",
    "tables/identification_macros": "ptmw analyse pass-through",
    "tables/regional_design": "ptmw analyse pass-through",
    "tables/premium_macros": "ptmw analyse pass-through",
    "tables/seasonality_macros": "ptmw analyse pass-through",
    "tables/exposure_macros": "ptmw build regional-exposure",
    "tables/exposure_design": "ptmw analyse exposure-design",
    "tables/exposure_design_macros": "ptmw analyse exposure-design",
    "tables/exposure_robustness": "ptmw analyse exposure-robustness",
}

_INPUT = re.compile(r"\\input\{(tables/[a-z_]+)\}")
_GRAPHIC = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{(figures/[a-z_]+\.pdf)\}")


def _imported_artefacts() -> set[str]:
    """Every generated file the manuscript pulls in."""
    text = (REPORT / "main.tex").read_text(encoding="utf-8")
    text += "".join(_section_text().values())
    return set(_INPUT.findall(text)) | set(_GRAPHIC.findall(text))


def test_every_imported_artefact_has_a_declared_producer() -> None:
    """The paper may not import a file no pipeline stage writes."""
    undeclared = _imported_artefacts() - _ARTEFACT_PRODUCER.keys()
    assert not undeclared, (
        f"the manuscript imports artefacts with no declared producer: {sorted(undeclared)}"
    )


def test_the_paper_target_runs_every_producing_stage() -> None:
    """`make paper` must run every stage the manuscript depends on.

    Adding an artefact to the paper and forgetting its stage leaves a build that
    works for whoever added it, because their working tree already holds the
    file, and fails for everyone starting from a clean checkout.
    """
    makefile = (REPORT.parent / "Makefile").read_text(encoding="utf-8")
    target = makefile.split("\npaper:", 1)
    assert len(target) == 2, "the Makefile has no 'paper' target"
    recipe = target[1].split("\n\n", 1)[0]

    required = {_ARTEFACT_PRODUCER[name] for name in _imported_artefacts()}
    missing = sorted(command for command in required if command not in recipe)
    assert not missing, f"'make paper' does not run: {missing}"


def test_no_producer_is_declared_for_an_artefact_nobody_imports() -> None:
    """A stale entry would keep `make paper` running a stage nothing needs."""
    orphaned = _ARTEFACT_PRODUCER.keys() - _imported_artefacts()
    assert not orphaned, f"declared producers for unimported artefacts: {sorted(orphaned)}"
