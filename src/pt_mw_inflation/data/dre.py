"""Retrieval of Portuguese legal acts from the Diário da República.

The official gazette's web interface is a client-side application that returns
no content to a plain request, which is why an earlier version of this project
could not cite the acts it depended on. Its documents are, however, addressable
through European Legislation Identifier permalinks, which redirect to the
published PDF on the gazette's file host:

    https://data.dre.pt/eli/{type}/{number}/{year}/{month}/{day}/{jurisdiction}/dre/pt/pdf

That address is stable, official, and citable, so the wage history no longer
rests on a provider's summary table: each value can be read from the act that
set it.

Note the type token. ``dec-lei`` is the national decree-law; regional
legislative decrees are ``declegreg``, not the hyphenated form the rest of the
scheme would suggest, and the other spellings answer 200 with the application
shell rather than a document. The jurisdiction is ``p`` for the Republic, ``m``
for Madeira and ``a`` for the Azores.
"""

from __future__ import annotations

import io
import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Literal

from pypdf import PdfReader

from pt_mw_inflation.data.http import fetch

ELI_BASE = "https://data.dre.pt/eli"

ActType = Literal["dec-lei", "declegreg", "lei", "port"]
Jurisdiction = Literal["p", "m", "a"]

#: Amounts in escudos, printed as "63 800$" or "63.800$00".
_ESCUDO_PATTERN = re.compile(r"(\d{1,3}(?:[ .]\d{3})+)\s*\$(?:\d{2})?")

#: A euro figure, with or without cents.
_MONEY = r"\d{1,3}(?:[ .]\d{3})*(?:,\d{2})?"
#: Acts place the euro sign on either side of the amount, and text extraction
#: routinely mangles the glyph itself. Both orders are matched, and the sign is
#: allowed to appear as the replacement character. A bare capital "E" is
#: accepted only in front of a figure carrying cents, which is how the gazette
#: rendered euro amounts around the changeover; without that restriction it
#: would match ordinary prose.
_EURO_PATTERNS = (
    re.compile(rf"[€�]\s*({_MONEY})"),
    re.compile(rf"({_MONEY})\s*[€�]"),
    re.compile(r"E\s*(\d{1,3}(?:[ .]\d{3})*,\d{2})"),
)

#: The regional supplement, as stated in the article that fixes it.
_SUPPLEMENT_PATTERN = re.compile(r"acrescimo de\s*(\d{1,2}(?:,\d+)?)\s*%", re.IGNORECASE)


@dataclass(frozen=True)
class LegalAct:
    """One retrieved legal act and the amounts it states."""

    act_type: ActType
    number: int
    act_date: date
    jurisdiction: Jurisdiction
    url: str
    text: str

    @property
    def citation(self) -> str:
        """Return the conventional Portuguese citation for the act."""
        label = {
            "dec-lei": "Decreto-Lei",
            "declegreg": "Decreto Legislativo Regional",
            "lei": "Lei",
            "port": "Portaria",
        }[self.act_type]
        suffix = {"p": "", "m": "/M", "a": "/A"}[self.jurisdiction]
        # Portuguese citations abbreviate the year for last-century acts and
        # write it in full from 2000, which is also how the DGERT history
        # renders them; matching that keeps the two comparable.
        if self.act_type == "declegreg" or self.act_date.year >= 2000:
            formatted = f"{self.act_date.year}"
        else:
            formatted = f"{self.act_date.year % 100:02d}"
        return (
            f"{label} n.º {self.number}/{formatted}{suffix} "
            f"de {self.act_date.day} de {_MONTHS[self.act_date.month]} de {self.act_date.year}"
        )


_MONTHS = {
    1: "janeiro",
    2: "fevereiro",
    3: "março",
    4: "abril",
    5: "maio",
    6: "junho",
    7: "julho",
    8: "agosto",
    9: "setembro",
    10: "outubro",
    11: "novembro",
    12: "dezembro",
}


def eli_url(
    act_type: ActType,
    number: int,
    act_date: date,
    *,
    jurisdiction: Jurisdiction = "p",
    fmt: str = "pdf",
) -> str:
    """Build the permalink for one act.

    Args:
        act_type: Kind of act.
        number: Act number within its year.
        act_date: Publication date.
        jurisdiction: Republic, Madeira or the Azores.
        fmt: ``pdf`` for the published document, ``html`` for the shell.

    Returns:
        The permalink.
    """
    return (
        f"{ELI_BASE}/{act_type}/{number}/{act_date.year}/{act_date.month:02d}/"
        f"{act_date.day:02d}/{jurisdiction}/dre/pt/{fmt}"
    )


#: A word split across a line by the typesetter, once the line break has become
#: a space. Requiring a letter on both sides is what keeps it from touching the
#: real hyphens these acts contain: ``Decreto -Lei`` and ``254 -A/2015`` both
#: carry a space *before* the hyphen, not after.
_SOFT_HYPHEN = re.compile(r"(?<=[A-Za-z])- (?=[a-z])")


def _normalise(text: str) -> str:
    """Strip accents, rejoin split words, and collapse whitespace.

    Rejoining matters more than it looks. These acts are read from the gazette's
    own typesetting, which hyphenates freely, so ``retribuicao minima`` may
    reach us as ``retribuicao mi- nima``. A phrase search then fails to match
    with no sign that anything went wrong: the act is retrieved, the text is
    present, and the extractor reports that the act sets no wage.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(char for char in decomposed if not unicodedata.combining(char))
    return _SOFT_HYPHEN.sub("", " ".join(stripped.split()))


def parse_amounts(text: str) -> list[tuple[Decimal, Literal["PTE", "EUR"]]]:
    """Extract every monetary amount stated in an act, in order of appearance.

    Args:
        text: Extracted document text.

    Returns:
        Amounts with the currency they were published in.
    """
    found: list[tuple[int, Decimal, Literal["PTE", "EUR"]]] = []

    for match in _ESCUDO_PATTERN.finditer(text):
        digits = match.group(1).replace(" ", "").replace(".", "")
        found.append((match.start(), Decimal(digits), "PTE"))

    # Patterns overlap: "€ 980,00" is matched by the sign-first rule, and a
    # figure followed by a sign by the second. Keying on the start offset alone
    # would let one amount be recorded twice, which shifts every position that
    # amount_index selects from.
    claimed: list[tuple[int, int]] = []
    for pattern in _EURO_PATTERNS:
        for match in pattern.finditer(text):
            span = match.span()
            if any(span[0] < end and start < span[1] for start, end in claimed):
                continue
            claimed.append(span)
            digits = match.group(1).replace(" ", "").replace(".", "").replace(",", ".")
            found.append((match.start(), Decimal(digits), "EUR"))

    return [(amount, currency) for _, amount, currency in sorted(found)]


def fetch_act(
    act_type: ActType,
    number: int,
    act_date: date,
    *,
    jurisdiction: Jurisdiction = "p",
    timeout_seconds: int = 120,
) -> LegalAct:
    """Retrieve one act and extract its text.

    Args:
        act_type: Kind of act.
        number: Act number within its year.
        act_date: Publication date.
        jurisdiction: Republic, Madeira or the Azores.
        timeout_seconds: Request timeout.

    Returns:
        The act with its extracted text.

    Raises:
        requests.HTTPError: If the request fails.
        ValueError: If the response is not a PDF, which is how a wrong type
            token or a withdrawn act surfaces: the permalink still answers 200,
            but with the application shell.
    """
    url = eli_url(act_type, number, act_date, jurisdiction=jurisdiction)
    # Shared retry policy: the gazette occasionally answers 503, and a build
    # that reads primary law must not fail on one transient response.
    response = fetch(url, timeout_seconds=timeout_seconds)

    media_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
    if media_type != "application/pdf":
        raise ValueError(
            f"{url} returned {media_type!r} rather than a PDF. The act may not exist, "
            "or the type token may be wrong: regional legislative decrees are "
            "'declegreg'."
        )

    reader = PdfReader(io.BytesIO(response.content))
    text = _normalise(" ".join((page.extract_text() or "") for page in reader.pages))

    return LegalAct(
        act_type=act_type,
        number=number,
        act_date=act_date,
        jurisdiction=jurisdiction,
        url=url,
        text=text,
    )


def extract_minimum_wage(act: LegalAct) -> tuple[Decimal, Literal["PTE", "EUR"]]:
    """Read the statutory minimum wage an act sets.

    Acts state several figures, so the first amount after the operative phrase
    is taken rather than the first amount in the document, which is often a
    page number or a cross-reference to the act being replaced.

    Args:
        act: A retrieved act.

    Returns:
        The wage and the currency it was published in.

    Raises:
        ValueError: If no amount can be located, which means the act does not
            set a wage or its wording has changed.
    """
    lowered = act.text.lower()
    # Anchored on the operative wording, most specific first. Anchoring matters:
    # these acts also state pension thresholds and tax bands, and the first
    # amount in the document is frequently a page number or a reference to the
    # act being replaced.
    markers = (
        # Madeira states the wage as the national figure "increased by a regional
        # supplement", naming itself in the same clause. The construction is the
        # region's own and appears in every one of its decrees back to 2010, so
        # it is anchored first: it is the most specific wording available, and
        # unlike the generic markers below it cannot match a neighbouring act on
        # the same gazette page.
        "acrescido de complemento regional, e, na regiao autonoma da madeira, de",
        "valor da retribuicao minima mensal garantida para vigorar",
        "retribuicao minima mensal garantida e de",
        "passam a ser de",
        "e fixado em",
        "no valor de",
    )
    for marker in markers:
        # Every occurrence is considered, not just one. The operative article is
        # usually last, but a revocation clause may quote the superseded value,
        # and silently preferring either position would return a wrong wage with
        # no warning. Disagreement is reported instead of resolved.
        candidates = []
        start = lowered.find(marker)
        while start >= 0:
            amounts = parse_amounts(act.text[start : start + 400])
            if amounts:
                candidates.append(amounts[0])
            start = lowered.find(marker, start + 1)

        distinct = set(candidates)
        if len(distinct) == 1:
            return candidates[0]
        if len(distinct) > 1:
            raise ValueError(
                f"{act.citation} states conflicting amounts after {marker!r}: "
                f"{sorted(distinct)}. The operative article cannot be identified "
                "unambiguously."
            )

    raise ValueError(
        f"no monetary amount found near the operative wording of {act.citation}; "
        "the act may set a legal regime rather than a value"
    )


def extract_regional_supplement(act: LegalAct) -> Decimal:
    """Read the regional supplement percentage an act establishes.

    The Azores do not publish a wage figure. Their statute fixes a proportional
    supplement to whatever the national wage is, so the regional series has to
    be derived from the rule rather than parsed from a value.

    Args:
        act: A retrieved act.

    Returns:
        The supplement as a percentage.

    Raises:
        ValueError: If the act states no supplement.
    """
    lowered = act.text.lower()
    position = lowered.find("acrescimo regional ao salario minimo")
    # Search the same string the anchor was found in: the article heading may be
    # capitalised, and the pension bands preceding it quote other percentages.
    window = lowered[position:] if position >= 0 else lowered
    match = _SUPPLEMENT_PATTERN.search(window)
    if match is None:
        raise ValueError(f"no regional supplement stated in {act.citation}")
    return Decimal(match.group(1).replace(",", "."))
