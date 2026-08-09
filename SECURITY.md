# Security Policy

## Scope

This is a research repository. It has no server, no authentication, and no user
data. The realistic risk surface is narrow:

- The data downloaders in `src/pt_mw_inflation/data/` fetch remote files over
  HTTP and parse them (HTML, XML, XLSX, CSV, Parquet).
- Deserialisation or archive-extraction issues in those parsers.
- Vulnerabilities in pinned third-party dependencies.
- Anything that could cause a checksum-verified source to be silently replaced.

Reports outside that scope — for example, findings about the statistical
methodology — belong in a regular issue, not here.

## Supported versions

Only the `main` branch is supported. Fixes land there; there are no maintained
release branches.

| Version | Supported |
| ------- | --------- |
| `main`  | ✅        |
| tagged releases | ❌ (upgrade to `main`) |

## Reporting a vulnerability

**Do not open a public issue for a security report.**

Use GitHub's private reporting form:
[Report a vulnerability](https://github.com/DiogoRibeiro7/portugal-minimum-wage-inflation/security/advisories/new)

If private advisories are unavailable to you, email **diogo_dj@hotmail.com**
with `SECURITY` in the subject line.

Please include:

- A description of the issue and why you believe it is exploitable.
- The affected file, function, or dependency.
- Reproduction steps or a proof of concept.
- Any suggested remediation.

## What to expect

- **Acknowledgement** within 7 days.
- **Initial assessment** within 14 days.
- **Fix or documented mitigation** for confirmed issues as quickly as the
  severity warrants; you will be kept informed if it takes longer.
- **Credit** in the release notes and the advisory, unless you prefer to remain
  anonymous.

This is a single-maintainer academic project, not a funded product. There is no
bug-bounty programme and no financial reward.

## Dependency vulnerabilities

Dependencies are pinned in `poetry.lock` and updated by Dependabot. If you spot
a vulnerable transitive dependency that Dependabot has not flagged, a normal
public issue is fine — advisories for public packages are already public.
