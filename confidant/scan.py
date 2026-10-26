"""Offline text scanners shared by the memory store, the guard and the assistant.

* :func:`find_secrets` spots credentials (passwords, PINs, payment cards that pass
  the Luhn check, IBANs, API keys and tokens) so they are never stored or sent.
* :func:`find_values` extracts the distinctive values of a text (dates, times,
  numbers, e-mail addresses and capitalised names) under normalised keys, so a
  value is recognised however it is written: "1,180" and "1180", "3 Nov" and
  "November 3rd", "Léa" and "lea".
* :func:`fold` lower-cases, strips accents and collapses whitespace while keeping
  a map back to the original characters.

Everything here is pure and deterministic, and runs on the device.
"""
from __future__ import annotations

import calendar
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, timedelta

MONTHS: dict[str, int] = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sept": 9, "sep": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}
WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_WEEKDAY_SHORT = ("mon", "tue", "tues", "wed", "thu", "thur", "thurs", "fri", "sat", "sun")
CALENDAR_WORDS = frozenset(MONTHS) | frozenset(WEEKDAYS) | frozenset(_WEEKDAY_SHORT)

STOPWORDS = frozenset(
    """
    a about above after again against all also am an and any are as at be because been before
    being below between both but by can could did do does doing done down during each few for
    from further get gets getting give given go going gone got had has have having he her here
    hers herself him himself his how i if in into is it its itself just let lets like me might
    more most much must my myself next no nor not now of off on once only or other our ours
    ourselves out over own please really same see shall she should so some still such than that
    the their theirs them themselves then there these they this those through to too under until
    up upon us very want was we were what whats when where which while who whom why will with
    would yet you your yours yourself s t d ll re ve m
    find search look looking draft write tell show make help know remember note save keep forget
    today tonight tomorrow yesterday day days week weeks weekend month months year upcoming coming
    plate agenda schedule calendar plan plans went moved changed change changes
    """.split()
)


def fold(text: str) -> tuple[str, list[int]]:
    """Case-fold ``text``, strip accents and collapse whitespace runs to one space.

    Returns the folded text and, for each folded character, the index of the
    original character it came from, so matches can be mapped back to spans.
    """
    chars: list[str] = []
    origin: list[int] = []
    for index, char in enumerate(text):
        if char.isspace():
            if chars and chars[-1] != " ":
                chars.append(" ")
                origin.append(index)
            continue
        for piece in unicodedata.normalize("NFKD", char.casefold()):
            if not unicodedata.combining(piece):
                chars.append(piece)
                origin.append(index)
    return "".join(chars), origin


def words(text: str) -> list[str]:
    """Folded word tokens of ``text`` (letters and digits; hyphenated words stay whole)."""
    return re.findall(r"[^\W_]+(?:-[^\W_]+)*", fold(text)[0])


# --- dates -----------------------------------------------------------------

def _alternation(options: Iterable[str]) -> str:
    """Regex alternation of ``options``, longest first so that "november" beats "nov"."""
    return "|".join(sorted(options, key=len, reverse=True))


_MONTH = _alternation(MONTHS)
_WEEKDAY_PREFIX = rf"(?:(?:{_alternation(WEEKDAYS + _WEEKDAY_SHORT)})\.?,?\s+(?:the\s+)?)?"
_ORDINAL = r"(?:st|nd|rd|th)?"
_YEAR = r"(?:,?\s+(?P<year>\d{4}))?"
_DATE_PATTERNS = (
    re.compile(r"\b(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})\b"),
    re.compile(
        rf"\b{_WEEKDAY_PREFIX}(?P<day>\d{{1,2}}){_ORDINAL}\s+(?:of\s+)?(?P<month>{_MONTH})\.?{_YEAR}\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b{_WEEKDAY_PREFIX}(?P<month>{_MONTH})\.?\s+(?P<day>\d{{1,2}}){_ORDINAL}{_YEAR}\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?P<day>\d{1,2})/(?P<month>\d{1,2})/(?P<year>\d{4})\b"),
)


@dataclass(frozen=True)
class DateMatch:
    """A calendar date found in text; ``year`` is None when the text omits it."""

    start: int
    end: int
    month: int
    day: int
    year: int | None

    def resolve(self, today: date) -> date:
        """The date meant, relative to ``today`` when the year is missing.

        A date without a year is taken in the current year, or in the next one
        if that would put it more than six months in the past.
        """
        if self.year is not None:
            return _clamped(self.year, self.month, self.day)
        candidate = _clamped(today.year, self.month, self.day)
        if candidate < today - timedelta(days=183):
            candidate = _clamped(today.year + 1, self.month, self.day)
        return candidate


def _clamped(year: int, month: int, day: int) -> date:
    """``date(year, month, day)``, moving 29 February to the 28th in common years."""
    return date(year, month, min(day, calendar.monthrange(year, month)[1]))


def _spans_overlap(spans: Iterable[DateMatch | Value | Secret], start: int, end: int) -> bool:
    """True when [start, end) overlaps any of the spans already found."""
    return any(start < item.end and item.start < end for item in spans)


def find_dates(text: str) -> list[DateMatch]:
    """Dates written as ISO (2026-11-03), day-month ("Tue 3 Nov 2026", "3rd of November"),
    month-day ("November 3rd, 2026") or day/month/year ("03/11/2026"), in text order."""
    found: list[DateMatch] = []
    for pattern in _DATE_PATTERNS:
        for match in pattern.finditer(text):
            parsed = _parse_date(match)
            if parsed is not None and not _spans_overlap(found, parsed.start, parsed.end):
                found.append(parsed)
    return sorted(found, key=lambda item: item.start)


def _parse_date(match: re.Match[str]) -> DateMatch | None:
    """Turn a date regex match into a DateMatch, or None if it is not a real date."""
    month_text = match.group("month")
    month = int(month_text) if month_text.isdigit() else MONTHS[month_text.lower()]
    day = int(match.group("day"))
    year_text = match.groupdict().get("year")
    year = int(year_text) if year_text else None
    if not 1 <= month <= 12 or not 1 <= day <= calendar.monthrange(year or 2024, month)[1]:
        return None
    return DateMatch(match.start(), match.end(), month, day, year)


# --- values ----------------------------------------------------------------

_TIME_24 = re.compile(r"(?<![\d:.,])(?P<hour>[01]?\d|2[0-3])[:h](?P<minute>[0-5]\d)(?![\d:])", re.I)
_TIME_12 = re.compile(
    r"(?<![\d:.,])(?P<hour>1[0-2]|0?[1-9])(?::(?P<minute>[0-5]\d))?\s?(?P<half>[ap])\.?m\b\.?", re.I
)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_NUMBER = re.compile(r"(?<![\w.,])[€$£]?\+?\d(?:[\d   ,.'’-]*\d)?(?!\w)")
_WORD = re.compile(r"[^\W\d_][\w'’-]*")
_SENTENCE_BREAK = ".!?:;\n•*-"

KIND_PHRASES = {
    "date": "a date",
    "time": "a time",
    "email": "an e-mail address",
    "number": "a number",
    "name": "a name",
}


@dataclass(frozen=True)
class Value:
    """A distinctive value in text: ``kind`` is a key of KIND_PHRASES, ``key`` its
    normalised form (for example "date:11-03", "time:10:30", "number:1180", "name:lea")."""

    kind: str
    key: str
    start: int
    end: int

    @property
    def folded(self) -> str:
        """The normalised value without its kind prefix."""
        return self.key.split(":", 1)[1]


def find_values(text: str) -> list[Value]:
    """Dates, times, e-mail addresses, numbers (three digits or more, years excluded)
    and capitalised names that do not start a sentence, in text order."""
    values = [
        Value("date", f"date:{found.month:02d}-{found.day:02d}", found.start, found.end)
        for found in find_dates(text)
    ]

    def add(kind: str, key: str, start: int, end: int) -> None:
        """Keep a value unless an earlier, more specific one already covers its span."""
        if not _spans_overlap(values, start, end):
            values.append(Value(kind, key, start, end))

    for match in _TIME_12.finditer(text):  # before 24-hour times, so "2:15 pm" is 14:15
        hour = int(match["hour"]) % 12 + (12 if match["half"].lower() == "p" else 0)
        add("time", f"time:{hour:02d}:{match['minute'] or '00'}", *match.span())
    for match in _TIME_24.finditer(text):
        add("time", f"time:{int(match['hour']):02d}:{match['minute']}", *match.span())
    for match in _EMAIL.finditer(text):
        add("email", f"email:{match.group().lower()}", *match.span())
    for match in _NUMBER.finditer(text):
        raw = match.group()
        digits = re.sub(r"\D", "", raw)
        is_year = raw.isdigit() and len(raw) == 4 and 1900 <= int(raw) <= 2100
        if len(digits) >= 3 and not is_year:
            add("number", f"number:{digits}", *match.span())
    for match in _WORD.finditer(text):
        name = _name_at(text, match)
        if name:
            add("name", f"name:{fold(name)[0]}", match.start(), match.start() + len(name))
    return sorted(values, key=lambda value: value.start)


def _name_at(text: str, match: re.Match[str]) -> str | None:
    """The capitalised word at ``match`` if it reads as a name (not a sentence start,
    not a weekday or month, at least three letters), without a trailing "'s"."""
    token = re.sub(r"(?:['’]s|['’-])$", "", match.group())
    if len(token) < 3 or not token[0].isupper() or fold(token)[0] in CALENDAR_WORDS:
        return None
    before = text[: match.start()].rstrip(" \t\"'“‘([")
    if not before or before[-1] in _SENTENCE_BREAK:
        return None
    return token


# --- secrets ---------------------------------------------------------------

@dataclass(frozen=True)
class Secret:
    """A credential-like span of text and a human explanation of what it looks like."""

    start: int
    end: int
    reason: str


_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("a password", re.compile(
        r"\b(?:password|passwd|passphrase|pwd)s?\b"
        r"(?!\s+(?:manager|policy|policies|reset|rules?|hint|strength|field|prompt|change|expiry|generator"
        r"|requirements?|protected)\b)"
        r"(?:[^.\n]{0,30}?\b(?:is|was)\b|\s*[:=])\s*"
        r"(?!(?:strong|weak|wrong|correct|incorrect|expired|required|missing|invalid|too|the|a|an|not|still"
        r"|saved|stored|changed|reset)\b)\S{3,}",
        re.I)),
    ("a PIN or access code", re.compile(
        r"\b(?:pin(?:\s*(?:code|number))?|passcode|door\s*code|access\s*code|security\s*code|cvv2?|cvc)\b"
        r"\D{0,20}?(?<!\d)\d{3,8}(?!\d)",
        re.I)),
    ("an API key or access token", re.compile(
        r"\b(?:sk|pk|rk)-[\w-]{16,}|\btvly-[\w-]{16,}|\bgh[pousr]_[A-Za-z0-9]{20,}|\bgithub_pat_\w{20,}"
        r"|\bxox[abprs]-[A-Za-z0-9-]{10,}|\bAKIA[0-9A-Z]{16}\b|\bAIza[\w-]{30,}|\bhf_[A-Za-z0-9]{20,}"
        r"|\beyJ[\w-]{10,}\.[\w-]{10,}\.[\w-]{10,}")),
    ("a private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("an API key or access token", re.compile(
        r"\b(?:api[ _-]?key|access[ _-]?token|auth[ _-]?token|secret[ _-]?key|client[ _-]?secret"
        r"|bearer|token|secret)\b\W{0,3}(?:is\s+)?[\w\-./+=]{16,}",
        re.I)),
    ("a long random-looking token, possibly a key", re.compile(
        r"(?<![\w/.-])(?=[\w-]*\d)(?=[\w-]*[A-Za-z])[A-Za-z0-9_-]{32,}(?![\w/.-])")),
)
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,3})?\b")
_CARD = re.compile(r"(?<!\d)\d(?:[ -]?\d){12,18}(?!\d)")


def luhn_valid(digits: str) -> bool:
    """True when the digit string passes the Luhn checksum used by payment cards."""
    total = 0
    for position, char in enumerate(reversed(digits)):
        digit = int(char)
        if position % 2:
            digit = digit * 2 - 9 if digit > 4 else digit * 2
        total += digit
    return total % 10 == 0


def iban_valid(iban: str) -> bool:
    """True when ``iban`` (no spaces) has a valid ISO 13616 mod-97 checksum."""
    rearranged = iban[4:] + iban[:4]
    return int("".join(str(int(char, 36)) for char in rearranged)) % 97 == 1


def find_secrets(text: str) -> list[Secret]:
    """Credential-like spans: passwords, PINs and codes, IBANs, payment card numbers
    (Luhn), API keys and tokens, private keys. Spans do not overlap."""
    found: list[Secret] = []

    def add(start: int, end: int, reason: str) -> None:
        """Keep a secret unless an earlier pattern already covers its span."""
        if not _spans_overlap(found, start, end):
            found.append(Secret(start, end, reason))

    for reason, pattern in _SECRET_PATTERNS:
        for match in pattern.finditer(text):
            add(match.start(), match.end(), reason)
    for match in _IBAN.finditer(text):
        compact = match.group().replace(" ", "")
        if 15 <= len(compact) <= 34 and iban_valid(compact):
            add(match.start(), match.end(), "a bank account number (an IBAN with a valid checksum)")
    for match in _CARD.finditer(text):
        if luhn_valid(re.sub(r"\D", "", match.group())):
            add(match.start(), match.end(), "a payment card number (it passes the Luhn check)")
    return sorted(found, key=lambda secret: secret.start)


def detect_secret(text: str) -> str | None:
    """Explain what credential ``text`` seems to contain, or return None if none."""
    secrets = find_secrets(text)
    return secrets[0].reason if secrets else None
