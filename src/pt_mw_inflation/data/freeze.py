"""Freeze the raw inputs a published result was built from, and detect drift.

`make paper` runs end to end from an empty tree, which establishes that the
pipeline reproduces itself. It does not establish that it reproduces the *same
numbers*, because every stage fetches from a live statistical API and those APIs
revise their history. Statistics Portugal rebases the consumer price index;
Eurostat revises national accounts backwards for years. A rebuild in 2027 could
therefore produce a different figure from the published one with nothing in the
output to say so.

Freezing records a checksum for every raw input behind a published run.
Verifying recomputes them and reports what moved. The point is not to prevent
the data changing --- it should change, statistical agencies improve their
estimates --- but to make the change visible at the moment it happens rather
than after a number in the manuscript has quietly shifted.

The manifest is committed. The raw files are not: they are large, and several
carry redistribution terms this project does not own. What is distributed is
enough to detect drift, not enough to replace retrieval.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from pt_mw_inflation.data.http import sha256_bytes


class FreezeError(RuntimeError):
    """Raised when the frozen manifest cannot be read or written."""


#: Files that are build outputs of the retrieval itself rather than retrieved
#: bytes. Their checksums would change on every run for reasons that say nothing
#: about the upstream data.
_EXCLUDED_NAMES = frozenset({"source_manifest.json", ".gitkeep"})


@dataclass(frozen=True)
class DriftReport:
    """What changed between a frozen manifest and the working tree.

    Attributes:
        changed: Inputs whose bytes differ from the frozen checksum. These are
            upstream revisions and are the reason this exists.
        missing: Inputs recorded in the manifest and absent locally. Usually a
            partial download rather than a revision.
        added: Inputs present locally and not in the manifest. Usually a new
            source that has not been frozen yet.
        verified: How many inputs matched.
    """

    changed: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    verified: int = 0

    @property
    def clean(self) -> bool:
        """Whether every frozen input is present and unchanged."""
        return not self.changed and not self.missing


def checksum_inputs(raw_root: Path) -> dict[str, str]:
    """Checksum every retrieved file under the raw data directory.

    Args:
        raw_root: Directory holding retrieved sources.

    Returns:
        Repository-relative POSIX paths mapped to their SHA-256 digests, sorted
        so the manifest is stable across machines and filesystem orderings.

    Raises:
        FreezeError: If the directory does not exist.
    """
    if not raw_root.is_dir():
        raise FreezeError(f"{raw_root} does not exist; retrieve the sources first")

    digests: dict[str, str] = {}
    for path in sorted(raw_root.rglob("*")):
        if not path.is_file() or path.name in _EXCLUDED_NAMES:
            continue
        relative = path.relative_to(raw_root.parent.parent).as_posix()
        digests[relative] = sha256_bytes(path.read_bytes())
    return dict(sorted(digests.items()))


def write_manifest(raw_root: Path, destination: Path) -> int:
    """Record the checksum of every raw input, for a publishable run.

    Args:
        raw_root: Directory holding retrieved sources.
        destination: Manifest path, which is committed.

    Returns:
        How many inputs were frozen.

    Raises:
        FreezeError: If no input was found, which would freeze an empty tree and
            report every later run as clean.
    """
    digests = checksum_inputs(raw_root)
    if not digests:
        raise FreezeError(f"no inputs found under {raw_root}; nothing to freeze")

    payload = {
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "inputs": digests,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return len(digests)


def verify_manifest(raw_root: Path, manifest: Path) -> DriftReport:
    """Compare the working tree against a frozen manifest.

    Args:
        raw_root: Directory holding retrieved sources.
        manifest: Frozen manifest.

    Returns:
        What changed, went missing, or appeared.

    Raises:
        FreezeError: If the manifest is absent or unreadable.
    """
    if not manifest.is_file():
        raise FreezeError(f"{manifest} not found; run 'ptmw data freeze-inputs' first")

    try:
        frozen = json.loads(manifest.read_text(encoding="utf-8"))["inputs"]
    except (ValueError, KeyError) as error:
        raise FreezeError(f"{manifest} is not a readable manifest: {error}") from error

    current = checksum_inputs(raw_root)

    changed = sorted(
        name for name, digest in frozen.items() if name in current and current[name] != digest
    )
    missing = sorted(name for name in frozen if name not in current)
    added = sorted(name for name in current if name not in frozen)
    verified = sum(1 for name, digest in frozen.items() if current.get(name) == digest)

    return DriftReport(changed=changed, missing=missing, added=added, verified=verified)
