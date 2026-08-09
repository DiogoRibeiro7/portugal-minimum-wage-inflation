"""Typed data contracts used throughout the research pipeline."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, field_serializer, model_validator

SourceKind = Literal["html", "csv", "xlsx", "json", "pdf", "zip"]

#: Media types accepted for each declared source kind. A provider that silently
#: redirects a dead deep link to an HTML landing page is the failure mode this
#: guards against: the response is a valid 200, so only the media type reveals
#: that the payload is not the file that was requested.
EXPECTED_MEDIA_TYPES: dict[SourceKind, tuple[str, ...]] = {
    "html": ("text/html", "application/xhtml+xml"),
    "csv": ("text/csv", "text/plain", "application/csv", "application/octet-stream"),
    "xlsx": (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel",
        "application/octet-stream",
    ),
    "json": ("application/json", "text/json"),
    "pdf": ("application/pdf",),
    "zip": ("application/zip", "application/x-zip-compressed", "application/octet-stream"),
}


class SourceSpec(BaseModel):
    """Description of one external research data source."""

    provider: str
    kind: SourceKind
    url: HttpUrl
    destination: Path
    description: str
    licence: str = Field(
        default="unknown",
        description="Redistribution terms for the retrieved file, as stated by the provider.",
    )
    verified_on: date | None = Field(
        default=None,
        description="Date on which the URL was last confirmed to serve the expected payload.",
    )
    enabled: bool = Field(
        default=True,
        description="Set false to retain a documented but currently unreachable source.",
    )
    unavailable_reason: str | None = Field(
        default=None,
        description="Why a disabled source cannot be retrieved. Required when enabled is false.",
    )
    minimum_bytes: int = Field(
        default=1024,
        ge=0,
        description="Reject payloads smaller than this; truncated or error pages are tiny.",
    )

    @model_validator(mode="after")
    def _require_reason_when_disabled(self) -> SourceSpec:
        """Refuse a disabled source that does not say why it is disabled.

        A field validator is not enough: it does not run when the field is left
        at its default, which is exactly the case being guarded against.
        """
        if not self.enabled and not self.unavailable_reason:
            raise ValueError("unavailable_reason is required when enabled is false")
        return self


class DownloadRecord(BaseModel):
    """Provenance record for a downloaded immutable source file."""

    source_name: str
    provider: str
    url: HttpUrl
    destination: Path
    retrieved_at_utc: str
    sha256: str
    bytes: int
    media_type: str | None = None
    #: "created" on first retrieval, "unchanged" when the checksum matches the
    #: previous run, "changed" when upstream content moved and a dated snapshot
    #: of the previous bytes was retained.
    status: Literal["created", "unchanged", "changed"] = "created"
    previous_sha256: str | None = None
    snapshot_path: Path | None = None

    @field_serializer("destination", "snapshot_path")
    def _serialise_path(self, value: Path | None) -> str | None:
        """Emit forward-slash paths so the manifest is identical on every platform.

        Without this the manifest records native separators, and a run on
        Windows produces a different file from a run on Linux for byte-identical
        downloads.
        """
        return value.as_posix() if value is not None else None


class StatutoryMinimumWage(BaseModel):
    """One statutory minimum-wage regime for a geography, scope and date."""

    geography: str
    effective_date: date
    scope: Literal["general", "agriculture", "domestic_service"]
    minimum_wage_monthly_eur: float = Field(gt=0)
    payments_per_year: int = Field(gt=0)
    annualised_minimum_wage_eur: float = Field(gt=0)
    original_amount: float = Field(gt=0)
    original_currency: Literal["PTE", "EUR"]
    legal_source: str
    national_or_regional: Literal["national", "regional"]
    notes: str = ""
