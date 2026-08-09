"""Source-registry loading and batch download functions."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from pt_mw_inflation.data.http import download_source
from pt_mw_inflation.schemas import DownloadRecord, SourceSpec


def load_source_registry(path: Path) -> dict[str, SourceSpec]:
    """Load and validate the YAML source registry."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw_sources: dict[str, dict[str, object]] = payload["sources"]
    return {name: SourceSpec.model_validate(spec) for name, spec in raw_sources.items()}


def download_registry(registry_path: Path, root: Path) -> list[DownloadRecord]:
    """Download all registered sources and persist a provenance manifest."""
    registry = load_source_registry(registry_path)
    records = [download_source(name, spec, root) for name, spec in registry.items()]

    manifest_path = root / "data/raw/source_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps([record.model_dump(mode="json") for record in records], indent=2),
        encoding="utf-8",
    )
    return records
