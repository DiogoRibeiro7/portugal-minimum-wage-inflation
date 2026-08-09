"""HTTP download utilities with provenance and integrity metadata."""

from __future__ import annotations

import hashlib
import time
from datetime import UTC, datetime
from pathlib import Path

import requests

from pt_mw_inflation.schemas import EXPECTED_MEDIA_TYPES, DownloadRecord, SourceSpec

#: Identifies the retrieval to providers. Several Portuguese public sites reject
#: requests that do not present a user agent.
_REPOSITORY_URL = "https://github.com/DiogoRibeiro7/portugal-minimum-wage-inflation"
USER_AGENT = f"pt-mw-inflation-research/0.2 (+{_REPOSITORY_URL})"

RETRY_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class SourceIntegrityError(RuntimeError):
    """Raised when a response is technically successful but is not the expected payload."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    """Return the SHA-256 digest of an in-memory payload."""
    return hashlib.sha256(payload).hexdigest()


def _media_type(response: requests.Response) -> str:
    """Return the response media type without parameters such as charset."""
    return response.headers.get("Content-Type", "").split(";")[0].strip().lower()


def verify_payload(name: str, spec: SourceSpec, response: requests.Response) -> str:
    """Check that a successful response actually carries the declared kind of file.

    A dead deep link on a public portal frequently redirects to a site homepage
    and answers 200 with HTML. Writing those bytes to ``bulletin.xlsx`` and
    recording a checksum for them produces provenance that looks impeccable and
    describes the wrong file, so the mismatch is treated as a hard failure.

    Args:
        name: Registry identifier, used in the error message.
        spec: The source specification that was requested.
        response: The successful HTTP response.

    Returns:
        The response media type.

    Raises:
        SourceIntegrityError: If the media type or payload size is inconsistent
            with the declared source kind.
    """
    media_type = _media_type(response)
    allowed = EXPECTED_MEDIA_TYPES[spec.kind]

    if media_type and media_type not in allowed:
        raise SourceIntegrityError(
            f"{name}: declared kind {spec.kind!r} expects one of {allowed}, "
            f"but the server returned {media_type!r}. "
            f"Requested {spec.url}, final URL {response.url}. "
            "The link is most likely dead and redirecting to a landing page."
        )

    if len(response.content) < spec.minimum_bytes:
        raise SourceIntegrityError(
            f"{name}: payload is {len(response.content)} bytes, below the "
            f"{spec.minimum_bytes}-byte floor. A truncated or error response is likely."
        )

    return media_type


def fetch(
    url: str,
    *,
    timeout_seconds: int = 90,
    attempts: int = 3,
    backoff_seconds: float = 2.0,
    session: requests.Session | None = None,
) -> requests.Response:
    """Fetch a URL, retrying only on transient failures.

    Args:
        url: Absolute URL to retrieve.
        timeout_seconds: Per-attempt timeout.
        attempts: Total number of attempts, including the first.
        backoff_seconds: Base delay for linear backoff between attempts.
        session: Optional session, so a caller can reuse connections.

    Returns:
        The successful response.

    Raises:
        requests.HTTPError: If the final attempt returns a failed status.
        requests.RequestException: If the final attempt fails at transport level.
    """
    get = session.get if session is not None else requests.get
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            response = get(url, timeout=timeout_seconds, headers={"User-Agent": USER_AGENT})
            if response.status_code in RETRY_STATUS_CODES and attempt < attempts:
                last_error = requests.HTTPError(f"transient status {response.status_code}")
                time.sleep(backoff_seconds * attempt)
                continue
            response.raise_for_status()
            return response
        except requests.RequestException as error:  # noqa: PERF203 - retry needs the handler
            last_error = error
            if attempt == attempts:
                raise
            time.sleep(backoff_seconds * attempt)

    raise RuntimeError(f"unreachable retry state for {url}") from last_error


def download_source(
    name: str,
    spec: SourceSpec,
    root: Path,
    *,
    session: requests.Session | None = None,
) -> DownloadRecord:
    """Download one configured source and return a reproducibility record.

    Raw files are immutable. When upstream content changes, the previous bytes
    are preserved under a dated snapshot name rather than being overwritten, so
    that a rerun can never silently redefine the inputs of a published estimate.

    Args:
        name: Stable source identifier from the registry.
        spec: Validated source specification.
        root: Repository root used to resolve relative destinations.
        session: Optional shared HTTP session.

    Returns:
        A provenance record containing timestamp, size, media type and checksum.

    Raises:
        requests.HTTPError: If the remote server returns a failed response.
        SourceIntegrityError: If the payload is not the declared kind of file.
    """
    destination = root / spec.destination
    destination.parent.mkdir(parents=True, exist_ok=True)

    response = fetch(str(spec.url), session=session)
    media_type = verify_payload(name, spec, response)

    payload = response.content
    new_digest = sha256_bytes(payload)
    retrieved_at = datetime.now(UTC)

    status: str = "created"
    previous_digest: str | None = None
    snapshot_path: Path | None = None

    if destination.exists():
        previous_digest = sha256_file(destination)
        if previous_digest == new_digest:
            status = "unchanged"
        else:
            status = "changed"
            stamp = retrieved_at.strftime("%Y%m%dT%H%M%SZ")
            snapshot = destination.with_name(f"{destination.stem}.{stamp}{destination.suffix}")
            destination.replace(snapshot)
            snapshot_path = snapshot.relative_to(root)

    if status != "unchanged":
        destination.write_bytes(payload)

    return DownloadRecord(
        source_name=name,
        provider=spec.provider,
        url=spec.url,
        destination=spec.destination,
        retrieved_at_utc=retrieved_at.isoformat(),
        sha256=new_digest,
        bytes=len(payload),
        media_type=media_type or None,
        status=status,  # type: ignore[arg-type]
        previous_sha256=previous_digest if status == "changed" else None,
        snapshot_path=snapshot_path,
    )
