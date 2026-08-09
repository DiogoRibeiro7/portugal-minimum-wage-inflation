"""Source-registry loading and batch download functions."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import requests
import yaml

from pt_mw_inflation.data.http import SourceIntegrityError, download_source
from pt_mw_inflation.schemas import DownloadRecord, SourceSpec

MANIFEST_PATH = Path("data/raw/source_manifest.json")


class RegistryError(ValueError):
    """Raised when the source registry is internally inconsistent."""


def load_source_registry(path: Path) -> dict[str, SourceSpec]:
    """Load and validate the YAML source registry.

    Args:
        path: Path to the registry file.

    Returns:
        Validated specifications keyed by source identifier.

    Raises:
        RegistryError: If the file has no sources, or if two sources would
            write to the same destination, which silently makes one of them
            unreachable.
    """
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw_sources = (payload or {}).get("sources")
    if not raw_sources:
        raise RegistryError(f"{path} defines no sources")

    registry = {name: SourceSpec.model_validate(spec) for name, spec in raw_sources.items()}

    destinations = Counter(spec.destination for spec in registry.values())
    clashes = sorted(str(dest) for dest, count in destinations.items() if count > 1)
    if clashes:
        raise RegistryError(f"multiple sources share a destination: {clashes}")

    return registry


def download_registry(
    registry_path: Path,
    root: Path,
    *,
    include_disabled: bool = False,
) -> list[DownloadRecord]:
    """Download all enabled sources and persist a provenance manifest.

    Sources documented as unreachable stay in the registry with ``enabled:
    false`` so the research record keeps the citation, and are skipped here
    rather than failing every run.

    Args:
        registry_path: Path to the YAML registry.
        root: Repository root used to resolve relative destinations.
        include_disabled: Attempt disabled sources too, to re-check whether a
            provider has restored a link.

    Returns:
        One provenance record per attempted source.

    Raises:
        RuntimeError: If any source fails, after attempting all of them, so a
            single dead link does not hide the state of the rest.
    """
    registry = load_source_registry(registry_path)
    records: list[DownloadRecord] = []
    failures: list[str] = []

    with requests.Session() as session:
        for name, spec in registry.items():
            if not spec.enabled and not include_disabled:
                continue
            try:
                records.append(download_source(name, spec, root, session=session))
            except (requests.RequestException, SourceIntegrityError) as error:
                failures.append(f"{name}: {error}")

    write_manifest(records, root)

    if failures:
        raise RuntimeError(
            f"{len(failures)} of {len(records) + len(failures)} sources failed:\n"
            + "\n".join(failures)
        )

    return records


def write_manifest(records: list[DownloadRecord], root: Path) -> Path:
    """Persist the provenance manifest, sorted for a stable diff.

    Args:
        records: Provenance records from this run.
        root: Repository root.

    Returns:
        Path to the written manifest.
    """
    manifest_path = root / MANIFEST_PATH
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda record: record.source_name)
    manifest_path.write_text(
        json.dumps([record.model_dump(mode="json") for record in ordered], indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path
