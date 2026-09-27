"""Tests for source acquisition, integrity checks and provenance.

All tests are offline: responses are constructed in the test, never fetched.
"""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path

import pytest
import requests
from dataexcept import DataLoadingError, FileReadError, FileWriteError

from pt_mw_inflation.data import registry as registry_module
from pt_mw_inflation.data.http import (
    SourceIntegrityError,
    download_source,
    sha256_bytes,
    sha256_file,
    verify_payload,
)
from pt_mw_inflation.data.registry import (
    RegistryError,
    download_registry,
    load_source_registry,
    write_manifest,
)
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
    """A failed source retains the HTTP error and does not write raw data."""
    session = FakeSession(*[FakeResponse(b"", XLSX_MEDIA, status_code=404)] * 3)
    with pytest.raises(DataLoadingError) as caught:
        download_source("example", _spec(), tmp_path, session=session)  # type: ignore[arg-type]
    assert caught.value.source == "https://example.invalid/book.xlsx"
    assert isinstance(caught.value.original, requests.HTTPError)
    assert caught.value.__cause__ is caught.value.original
    assert not (tmp_path / "data/raw/example/book.xlsx").exists()


def test_unreadable_existing_source_reports_its_path(tmp_path: Path) -> None:
    """Checksum failures identify the original source rather than the download URL."""
    missing = tmp_path / "absent.xlsx"

    with pytest.raises(FileReadError) as caught:
        sha256_file(missing)

    assert caught.value.path == str(missing)
    assert isinstance(caught.value.__cause__, FileNotFoundError)


def test_unwritable_raw_source_reports_its_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A successful retrieval can still fail when persisting raw bytes."""
    destination = tmp_path / "data/raw/example/book.xlsx"
    original = OSError("read-only filesystem")

    def fail_write(_path: Path, _payload: bytes) -> int:
        raise original

    monkeypatch.setattr(Path, "write_bytes", fail_write)
    session = FakeSession(FakeResponse(b"PK\x03\x04payload", XLSX_MEDIA))

    with pytest.raises(FileWriteError) as caught:
        download_source("example", _spec(), tmp_path, session=session)  # type: ignore[arg-type]

    assert caught.value.path == str(destination)
    assert caught.value.__cause__ is original


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


def test_missing_registry_reports_its_path(tmp_path: Path) -> None:
    """A missing configuration file is distinguishable from an empty registry."""
    registry = tmp_path / "sources.yaml"
    with pytest.raises(FileReadError) as caught:
        load_source_registry(registry)

    assert caught.value.path == str(registry)
    assert isinstance(caught.value.__cause__, FileNotFoundError)


def test_batch_keeps_successful_sources_after_a_download_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One unreachable source must not hide another source's provenance record."""
    registry_path = tmp_path / "sources.yaml"
    registry_path.write_text(
        """sources:
  unavailable:
    provider: Example
    kind: html
    url: https://example.invalid/missing.html
    destination: data/raw/missing.html
    description: Unreachable source
    minimum_bytes: 4
  available:
    provider: Example
    kind: html
    url: https://example.invalid/available.html
    destination: data/raw/available.html
    description: Available source
    minimum_bytes: 4
""",
        encoding="utf-8",
    )
    session = FakeSession(
        *[FakeResponse(b"", "text/html", status_code=404) for _ in range(3)],
        FakeResponse(b"<html>available</html>", "text/html"),
    )
    monkeypatch.setattr(registry_module.requests, "Session", lambda: nullcontext(session))
    monkeypatch.setattr("pt_mw_inflation.data.http.time.sleep", lambda _delay: None)

    with pytest.raises(RuntimeError, match="1 of 2 sources failed") as caught:
        download_registry(registry_path, tmp_path)

    assert "DataLoadingError" in str(caught.value)
    assert len(session.requested) == 4
    assert (tmp_path / "data/raw/available.html").exists()
    manifest = json.loads((tmp_path / "data/raw/source_manifest.json").read_text(encoding="utf-8"))
    assert [record["source_name"] for record in manifest] == ["available"]


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


def test_unwritable_manifest_reports_its_path(tmp_path: Path) -> None:
    """A failed provenance write must not look like a completed download run."""
    (tmp_path / "data").write_text("not a directory", encoding="utf-8")
    manifest = tmp_path / "data/raw/source_manifest.json"

    with pytest.raises(FileWriteError) as caught:
        write_manifest([], tmp_path)

    assert caught.value.path == str(manifest)
    assert isinstance(caught.value.__cause__, OSError)


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


def test_two_revisions_in_the_same_second_keep_both_snapshots(tmp_path: Path) -> None:
    """Rapid successive revisions must not overwrite each other's evidence.

    The snapshot name carries a second-resolution timestamp. Two revisions of
    one source inside the same second would collide on that name alone, and the
    second would silently replace the first -- destroying exactly the retained
    payload this branch exists to preserve. The digest of the superseded bytes
    disambiguates them.
    """
    versions = [b"PK\x03\x04version-one", b"PK\x03\x04version-two", b"PK\x03\x04version-three"]
    snapshots: list[Path] = []

    for payload in versions:
        record = download_source(
            "example",
            _spec(),
            tmp_path,
            session=FakeSession(FakeResponse(payload, XLSX_MEDIA)),  # type: ignore[arg-type]
        )
        if record.snapshot_path is not None:
            snapshots.append(tmp_path / record.snapshot_path)

    assert len(snapshots) == 2
    assert len({path.name for path in snapshots}) == 2, "snapshots collided"
    # Every superseded version is still readable.
    retained = {path.read_bytes() for path in snapshots}
    assert retained == set(versions[:-1])
    assert (tmp_path / "data/raw/example/book.xlsx").read_bytes() == versions[-1]
