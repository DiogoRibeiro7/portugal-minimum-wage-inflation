"""Tests for freezing the raw inputs a published result was built from.

`make paper` establishes that the pipeline reproduces itself. It does not
establish that it reproduces the same numbers, because statistical agencies
revise their history: Statistics Portugal rebases the price index and Eurostat
revises national accounts backwards. These tests are about noticing that.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pt_mw_inflation.data.freeze import (
    FreezeError,
    checksum_inputs,
    verify_manifest,
    write_manifest,
)


def _tree(root: Path, files: dict[str, str]) -> Path:
    """Build a raw-data tree of the shape the pipeline produces."""
    raw = root / "data" / "raw"
    for name, content in files.items():
        path = raw / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return raw


def test_a_revised_input_is_reported_as_changed(tmp_path: Path) -> None:
    """The case this exists for: upstream silently revises its history."""
    raw = _tree(tmp_path, {"ine/cpi_1991.json": "original", "dgert/history.html": "stable"})
    manifest = tmp_path / "config" / "publication_inputs.json"
    assert write_manifest(raw, manifest) == 2

    (raw / "ine" / "cpi_1991.json").write_text("revised", encoding="utf-8")
    report = verify_manifest(raw, manifest)

    assert report.changed == ["data/raw/ine/cpi_1991.json"]
    assert report.verified == 1
    assert not report.clean


def test_an_unchanged_tree_verifies_clean(tmp_path: Path) -> None:
    """No drift must report no drift, or the check is noise."""
    raw = _tree(tmp_path, {"ine/cpi.json": "a", "gep/report.pdf": "b"})
    manifest = tmp_path / "config" / "inputs.json"
    write_manifest(raw, manifest)

    report = verify_manifest(raw, manifest)
    assert report.clean
    assert report.verified == 2
    assert not report.added


def test_missing_and_new_inputs_are_distinguished_from_revisions(tmp_path: Path) -> None:
    """A partial download is not an upstream revision and must not read as one."""
    raw = _tree(tmp_path, {"a.json": "one", "b.json": "two"})
    manifest = tmp_path / "config" / "inputs.json"
    write_manifest(raw, manifest)

    (raw / "a.json").unlink()
    (raw / "c.json").write_text("three", encoding="utf-8")
    report = verify_manifest(raw, manifest)

    assert report.missing == ["data/raw/a.json"]
    assert report.added == ["data/raw/c.json"]
    assert report.changed == []
    # A missing input still means the run cannot be reproduced as published.
    assert not report.clean


def test_the_retrieval_manifest_is_not_frozen_against_itself(tmp_path: Path) -> None:
    """`source_manifest.json` records retrieval times, so it changes every run.

    Freezing it would report drift on every download while saying nothing about
    whether the upstream data moved.
    """
    raw = _tree(tmp_path, {"source_manifest.json": "{}", "ine/cpi.json": "data"})
    digests = checksum_inputs(raw)

    assert "data/raw/ine/cpi.json" in digests
    assert not any(name.endswith("source_manifest.json") for name in digests)


def test_freezing_an_empty_tree_is_refused(tmp_path: Path) -> None:
    """An empty manifest would report every later run as clean."""
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    with pytest.raises(FreezeError, match="nothing to freeze"):
        write_manifest(raw, tmp_path / "config" / "inputs.json")


def test_verifying_without_a_manifest_says_how_to_make_one(tmp_path: Path) -> None:
    """The error has to name the command, not just the missing path."""
    raw = _tree(tmp_path, {"a.json": "one"})
    with pytest.raises(FreezeError, match="freeze-inputs"):
        verify_manifest(raw, tmp_path / "config" / "absent.json")


def test_a_corrupt_manifest_is_refused_rather_than_ignored(tmp_path: Path) -> None:
    """Silently treating an unreadable manifest as empty would disable the check."""
    raw = _tree(tmp_path, {"a.json": "one"})
    manifest = tmp_path / "config" / "inputs.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("not json", encoding="utf-8")

    with pytest.raises(FreezeError, match="readable manifest"):
        verify_manifest(raw, manifest)


def test_the_manifest_is_ordered_so_it_diffs_cleanly(tmp_path: Path) -> None:
    """A committed manifest that reorders on every run is unreviewable."""
    raw = _tree(tmp_path, {"z.json": "1", "a.json": "2", "m/n.json": "3"})
    manifest = tmp_path / "config" / "inputs.json"
    write_manifest(raw, manifest)

    inputs = json.loads(manifest.read_text(encoding="utf-8"))["inputs"]
    assert list(inputs) == sorted(inputs)


def test_an_extraction_stamp_does_not_read_as_a_revision(tmp_path: Path) -> None:
    """Responses carry the moment they were fetched, and that is not a change.

    The API returns a list whose first element carries the stamp, which is the
    shape content_digest strips. Statistics Portugal stamps each response with
    an extraction time, so the
    raw bytes differ on every fetch while the data is identical. Digesting the
    bytes directly reported every input as changed on every run, which is not a
    warning but noise, and noise in a check like this trains its reader to
    ignore it.
    """
    raw = _tree(tmp_path, {})
    path = raw / "ine" / "cpi.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([{"DataExtracao": "2026-08-11T10:00:00", "Dados": {"value": 1}}]),
        encoding="utf-8",
    )
    manifest = tmp_path / "config" / "inputs.json"
    write_manifest(raw, manifest)

    # Same data, fetched again a day later.
    path.write_text(
        json.dumps([{"DataExtracao": "2026-08-12T09:30:00", "Dados": {"value": 1}}]),
        encoding="utf-8",
    )
    assert verify_manifest(raw, manifest).clean


def test_a_real_revision_behind_a_new_stamp_is_still_caught(tmp_path: Path) -> None:
    """Stripping the stamp must not strip the signal it was hiding."""
    raw = _tree(tmp_path, {})
    path = raw / "ine" / "cpi.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([{"DataExtracao": "2026-08-11T10:00:00", "Dados": {"value": 1}}]),
        encoding="utf-8",
    )
    manifest = tmp_path / "config" / "inputs.json"
    write_manifest(raw, manifest)

    path.write_text(
        json.dumps([{"DataExtracao": "2026-08-12T09:30:00", "Dados": {"value": 2}}]),
        encoding="utf-8",
    )
    report = verify_manifest(raw, manifest)
    assert report.changed == ["data/raw/ine/cpi.json"]
