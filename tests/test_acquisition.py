"""Tests for source acquisition, integrity checks and provenance.

All tests are offline: responses are constructed in the test, never fetched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests

from pt_mw_inflation.data.http import (
    SourceIntegrityError,
    download_source,
    sha256_bytes,
    verify_payload,
)
from pt_mw_inflation.data.registry import RegistryError, load_source_registry, write_manifest
from pt_mw_inflation.schemas import SourceSpec


class FakeResponse:
    """Minimal stand-in for a requests response."""

    def __init__(self, content: bytes, media_type: str, status_code: int = 200) -> None:
        self.content = content
        self.headers = {"Content-Type": media_type}
        self.status_code = status_code
        self.url = "https://example.invalid/final"

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")


class FakeSession:
    """Session that returns queued responses and records the URLs requested."""

    def __init__(self, *responses: FakeResponse) -> None:
        self._responses = list(responses)
        self.requested: list[str] = []

    def get(self, url: str, **_: object) -> FakeResponse:
        self.requested.append(url)
        return self._responses.pop(0)


def _spec(**overrides: object) -> SourceSpec:
    payload: dict[str, object] = {
        "provider": "Example",
        "kind": "xlsx",
        "url": "https://example.invalid/book.xlsx",
        "destination": Path("data/raw/example/book.xlsx"),
        "description": "example",
        "minimum_bytes": 4,
    }
    payload.update(overrides)
    return SourceSpec.model_validate(payload)


XLSX_MEDIA = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def test_html_served_for_a_spreadsheet_is_rejected() -> None:
    """A dead deep link that redirects to a landing page must fail.

    This is the observed behaviour of the withdrawn GEP bulletin: the request
    succeeds with 200 and returns the site homepage. Without this check the
    HTML would be written under the workbook's name and checksummed, producing
    provenance that looks correct and describes the wrong file.
    """
    response = FakeResponse(b"<html>landing page</html>" * 10, "text/html; charset=UTF-8")
    with pytest.raises(SourceIntegrityError, match="dead"):
        verify_payload("gep_bulletin", _spec(), response)  # type: ignore[arg-type]


def test_truncated_payload_is_rejected() -> None:
    """A response below the declared size floor is not accepted."""
    response = FakeResponse(b"ab", XLSX_MEDIA)
    with pytest.raises(SourceIntegrityError, match="below the"):
        verify_payload("example", _spec(minimum_bytes=1024), response)  # type: ignore[arg-type]


def test_expected_media_type_passes() -> None:
    """A correct payload returns its media type."""
    response = FakeResponse(b"PK\x03\x04payload", XLSX_MEDIA)
    assert verify_payload("example", _spec(), response) == XLSX_MEDIA  # type: ignore[arg-type]


def test_failed_http_status_propagates(tmp_path: Path) -> None:
    """A server error is raised rather than silently written to disk."""
    session = FakeSession(*[FakeResponse(b"", XLSX_MEDIA, status_code=404)] * 3)
    with pytest.raises(requests.HTTPError):
        download_source("example", _spec(), tmp_path, session=session)  # type: ignore[arg-type]
    assert not (tmp_path / "data/raw/example/book.xlsx").exists()


def test_first_download_records_provenance(tmp_path: Path) -> None:
    """A new file is written with checksum, size and media type recorded."""
    payload = b"PK\x03\x04first-version"
    session = FakeSession(FakeResponse(payload, XLSX_MEDIA))
    record = download_source("example", _spec(), tmp_path, session=session)  # type: ignore[arg-type]

    assert record.status == "created"
    assert record.sha256 == sha256_bytes(payload)
    assert record.bytes == len(payload)
    assert record.media_type == XLSX_MEDIA
    assert (tmp_path / "data/raw/example/book.xlsx").read_bytes() == payload


def test_rerunning_without_upstream_change_is_identical(tmp_path: Path) -> None:
    """Re-running against unchanged upstream content produces the same checksum."""
    payload = b"PK\x03\x04stable-version"
    first = download_source(
        "example", _spec(), tmp_path, session=FakeSession(FakeResponse(payload, XLSX_MEDIA))
    )  # type: ignore[arg-type]
    second = download_source(
        "example", _spec(), tmp_path, session=FakeSession(FakeResponse(payload, XLSX_MEDIA))
    )  # type: ignore[arg-type]

    assert second.status == "unchanged"
    assert second.sha256 == first.sha256
    assert second.snapshot_path is None


def test_changed_upstream_content_is_snapshotted_not_overwritten(tmp_path: Path) -> None:
    """Raw data is immutable: a changed upstream file never destroys the old one."""
    original = b"PK\x03\x04original-version"
    revised = b"PK\x03\x04revised-version"

    download_source(
        "example", _spec(), tmp_path, session=FakeSession(FakeResponse(original, XLSX_MEDIA))
    )  # type: ignore[arg-type]
    record = download_source(
        "example", _spec(), tmp_path, session=FakeSession(FakeResponse(revised, XLSX_MEDIA))
    )  # type: ignore[arg-type]

    assert record.status == "changed"
    assert record.previous_sha256 == sha256_bytes(original)
    assert record.snapshot_path is not None

    assert (tmp_path / "data/raw/example/book.xlsx").read_bytes() == revised
    assert (tmp_path / record.snapshot_path).read_bytes() == original


def test_duplicate_destinations_are_rejected(tmp_path: Path) -> None:
    """Two sources writing to one path would make one of them unreachable."""
    registry = tmp_path / "sources.yaml"
    registry.write_text(
        """
sources:
  first:
    provider: A
    kind: html
    url: "https://example.invalid/a"
    destination: "data/raw/shared.html"
    description: first
  second:
    provider: B
    kind: html
    url: "https://example.invalid/b"
    destination: "data/raw/shared.html"
    description: second
""",
        encoding="utf-8",
    )
    with pytest.raises(RegistryError, match="share a destination"):
        load_source_registry(registry)


def test_empty_registry_is_rejected(tmp_path: Path) -> None:
    """An empty registry is a configuration error, not an empty run."""
    registry = tmp_path / "sources.yaml"
    registry.write_text("sources:\n", encoding="utf-8")
    with pytest.raises(RegistryError, match="no sources"):
        load_source_registry(registry)


def test_disabled_source_requires_a_reason() -> None:
    """A withdrawn source must document why it cannot be retrieved."""
    with pytest.raises(ValueError, match="unavailable_reason"):
        _spec(enabled=False)


def test_manifest_is_sorted_and_complete(tmp_path: Path) -> None:
    """The manifest is stable across runs so its diff is meaningful."""
    payload = b"PK\x03\x04manifest-version"
    records = [
        download_source(
            name,
            _spec(destination=Path(f"data/raw/example/{name}.xlsx")),
            tmp_path,
            session=FakeSession(FakeResponse(payload, XLSX_MEDIA)),  # type: ignore[arg-type]
        )
        for name in ("zulu", "alpha", "mike")
    ]

    manifest_path = write_manifest(records, tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert [entry["source_name"] for entry in manifest] == ["alpha", "mike", "zulu"]
    for entry in manifest:
        assert entry["sha256"] and entry["bytes"] > 0
        assert entry["retrieved_at_utc"].endswith("+00:00")
        assert entry["url"] and entry["provider"]


def test_repository_registry_is_valid() -> None:
    """The registry shipped in the repository must parse and be consistent."""
    root = Path(__file__).resolve().parents[1]
    registry = load_source_registry(root / "config/sources.yaml")

    assert "dgert_minimum_wage" in registry
    for name, spec in registry.items():
        assert spec.description.strip(), f"{name} has no description"
        assert spec.licence != "unknown" or not spec.enabled
        if not spec.enabled:
            assert spec.unavailable_reason
