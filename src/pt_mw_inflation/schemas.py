"""Typed data contracts used throughout the research pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, HttpUrl


class SourceSpec(BaseModel):
    """Description of one external research data source."""

    provider: str
    kind: Literal["html", "csv", "xlsx", "json", "pdf", "zip"]
    url: HttpUrl
    destination: Path
    description: str


class DownloadRecord(BaseModel):
    """Provenance record for a downloaded immutable source file."""

    source_name: str
    provider: str
    url: HttpUrl
    destination: Path
    retrieved_at_utc: str
    sha256: str
    bytes: int
