"""HTTP download utilities with provenance and integrity metadata."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import requests

from pt_mw_inflation.schemas import DownloadRecord, SourceSpec


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_source(name: str, spec: SourceSpec, root: Path) -> DownloadRecord:
    """Download one configured source and return a reproducibility record.

    Args:
        name: Stable source identifier from the registry.
        spec: Validated source specification.
        root: Repository root used to resolve relative destinations.

    Returns:
        A provenance record containing timestamp, size and checksum.

    Raises:
        requests.HTTPError: If the remote server returns a failed response.
    """
    destination = root / spec.destination
    destination.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(str(spec.url), timeout=90)
    response.raise_for_status()
    destination.write_bytes(response.content)

    return DownloadRecord(
        source_name=name,
        provider=spec.provider,
        url=spec.url,
        destination=spec.destination,
        retrieved_at_utc=datetime.now(UTC).isoformat(),
        sha256=sha256_file(destination),
        bytes=destination.stat().st_size,
    )
