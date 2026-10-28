"""Offline text scanners shared by the memory store, the guard and the assistant.

* :func:`find_secrets` spots credentials (passwords, PINs, payment cards that pass
  the Luhn check, IBANs, API keys and tokens) so they are never stored or sent.
* :func:`find_values` extracts the distinctive values of a text (dates, times,
  numbers, e-mail addresses and capitalised names) under normalised keys, so a
  value is recognised in the forms these scanners read: "1,180", "1 180,00 €",
  "€1,180.00" and "one thousand one hundred and eighty"; "3 Nov", "November the
  3rd", "3 novembre" and "3/11"; "10:30", "10h30", "10.30" and "half past ten";
  "Léa" and "lea". Format characters (zero-width spaces, soft hyphens) are
  ignored and full-width digits read as ASCII before matching. Forms these
  scanners do not read are covered, for protected values only, by
  :mod:`confidant.forms`, which writes each value out ahead of time.
* :func:`fold` lower-cases, strips accents and collapses whitespace while keeping
  a map back to the original characters.

Everything here is pure and deterministic, and runs on the device.
"""
from __future__ import annotations

import calendar
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import date, timedelta

MONTHS: dict[str, int] = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sept": 9, "sep": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}
# Month names in French, Spanish, Italian and German, with and without accents. They
# only count as dates next to a day number, so "mars" or "mai" alone is never a date.
FOREIGN_MONTHS: dict[str, int] = {
    "janvier": 1, "janv": 1, "février": 2, "fevrier": 2, "févr": 2, "fevr": 2, "mars": 3, "avril": 4,
    "avr": 4, "mai": 5, "juin": 6, "juillet": 7, "juil": 7, "août": 8, "aout": 8, "septembre": 9,
    "octobre": 10, "novembre": 11, "décembre": 12, "decembre": 12, "déc": 12,
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
    "septiembre": 9, "setiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
    "gennaio": 1, "febbraio": 2, "aprile": 4, "maggio": 5, "giugno": 6, "luglio": 7, "settembre": 9,
    "ottobre": 10, "dicembre": 12,
    "januar": 1, "februar": 2, "märz": 3, "maerz": 3, "marz": 3, "juni": 6, "juli": 7, "oktober": 10,
    "dezember": 12,
}
WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_WEEKDAY_SHORT = ("mon", "tue", "tues", "wed", "thu", "thur", "thurs", "fri", "sat", "sun")
_FOREIGN_WEEKDAYS = ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche")
CALENDAR_WORDS = (
    frozenset(MONTHS) | frozenset(FOREIGN_MONTHS) | frozenset(WEEKDAYS) | frozenset(_WEEKDAY_SHORT)
    | frozenset(_FOREIGN_WEEKDAYS)
)
# Day and month names as a private message may write them, for masking it (assistant.masked_private): whole or
# abbreviated, in English, French, Spanish, Italian and German, with and without accents.
WEEKDAY_WORDS = frozenset([
    *WEEKDAYS, *_WEEKDAY_SHORT, *_FOREIGN_WEEKDAYS,
    "lunes", "martes", "miércoles", "miercoles", "jueves", "viernes", "sábado", "sabado", "domingo",
    "lunedì", "lunedi", "martedì", "martedi", "mercoledì", "mercoledi", "giovedì", "giovedi", "venerdì", "venerdi",
    "sabato", "domenica",
    "montag", "dienstag", "mittwoch", "donnerstag", "freitag", "samstag", "sonnabend", "sonntag",
])
MONTH_WORDS = frozenset([*MONTHS, *FOREIGN_MONTHS])
_ALL_MONTHS = {**FOREIGN_MONTHS, **MONTHS}


def _day_words() -> dict[str, int]:
    """Days of the month written in words, in English and French ("third", "twenty-first",
    "trois", "vingt et un"), with the first twelve ordinals in German and Spanish."""
    units = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    nth = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth"]
    teens = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
             "nineteen"]
    teens_nth = ["tenth", "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth",
                 "seventeenth", "eighteenth", "nineteenth"]
    words: dict[str, int] = {}
    for i, (card, ordi) in enumerate(zip(units, nth), 1):
        words[card], words[ordi] = i, i
    for i, (card, ordi) in enumerate(zip(teens, teens_nth), 10):
        words[card], words[ordi] = i, i
    words.update(twenty=20, twentieth=20, thirty=30, thirtieth=30)
    for i, (card, ordi) in enumerate(zip(units, nth), 1):
        for sep in ("-", " "):
            words[f"twenty{sep}{card}"] = words[f"twenty{sep}{ordi}"] = 20 + i
    for sep in ("-", " "):
        words[f"thirty{sep}one"] = words[f"thirty{sep}first"] = 31
    fr = ["premier", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf", "dix", "onze", "douze",
          "treize", "quatorze", "quinze", "seize", "dix-sept", "dix-huit", "dix-neuf", "vingt"]
    words.update({word: i for i, word in enumerate(fr, 1)})
    for i, unit in enumerate(["deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf"], 2):
        words[f"vingt-{unit}"] = 20 + i
    words.update({"vingt et un": 21, "vingt-et-un": 21, "trente": 30, "trente et un": 31, "trente-et-un": 31,
                  "1er": 1})
    de = ["erste", "zweite", "dritte", "vierte", "fünfte", "sechste", "siebte", "achte", "neunte", "zehnte",
          "elfte", "zwölfte"]
    for i, word in enumerate(de, 1):
        words[word] = words[word + "n"] = words[word + "r"] = words[word + "s"] = i
    es = ["primero", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez", "once", "doce"]
    words.update({word: i for i, word in enumerate(es, 1)})
    return words


DAY_WORDS = _day_words()

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


class _Normal:
    """``text`` as the scanners read it: format characters (zero-width spaces and
    joiners, soft hyphens) removed, NFKC applied per character (full-width digits
    and letters become ASCII, no-break and thin spaces become spaces) and every
    Unicode decimal digit written as an ASCII digit. :meth:`span` maps a match
    back to the original text."""

    def __init__(self, text: str) -> None:
        chars: list[str] = []
        self.origin: list[int] = []
        for index, char in enumerate(text):
            if unicodedata.category(char) == "Cf":
                continue
            for piece in unicodedata.normalize("NFKC", char):
                digit = unicodedata.decimal(piece, None)
                chars.append(str(digit) if digit is not None else piece)
                self.origin.append(index)
        self.text = "".join(chars)
        self.identity = self.text == text

    def span(self, start: int, end: int) -> tuple[int, int]:
        """The original-text span of the normalised span [start, end)."""
        if self.identity:
            return start, end
        return self.origin[start], self.origin[end - 1] + 1


# --- dates -----------------------------------------------------------------

def _alternation(options: Iterable[str]) -> str:
    """Regex alternation of ``options``, longest first so that "november" beats "nov"."""
    return "|".join(sorted(options, key=len, reverse=True))


_MONTH = _alternation(_ALL_MONTHS)
_WEEKDAY_PREFIX = (
    rf"(?:(?:{_alternation(WEEKDAYS + _WEEKDAY_SHORT + _FOREIGN_WEEKDAYS)})\.?,?\s+(?:the\s+|le\s+)?)?"
)
_ORDINAL = r"(?:st|nd|rd|th|er|ère|ere|ème|eme|e|º|°)?"
_YEAR = r"(?:,?\s+(?P<year>\d{4}))?"
_DATE_PATTERNS = (
    re.compile(r"\b(?P<year>\d{4})(?P<sep>[-/.])(?P<month>\d{1,2})(?P=sep)(?P<day>\d{1,2})\b"),
    re.compile(
        rf"\b{_WEEKDAY_PREFIX}(?P<day>\d{{1,2}}){_ORDINAL}\.?\s+(?:of\s+|de\s+)?(?P<month>{_MONTH})\.?{_YEAR}(?!\w)",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b{_WEEKDAY_PREFIX}(?P<month>{_MONTH})\.?\s+(?:the\s+)?(?P<day>\d{{1,2}}){_ORDINAL}{_YEAR}\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?P<day>\d{1,2})/(?P<month>\d{1,2})/(?P<year>\d{4})\b"),
)
_DAY_WORD = _alternation(DAY_WORDS)
_WORD_DATE_PATTERNS = (
    re.compile(rf"\b{_WEEKDAY_PREFIX}(?:the\s+|le\s+|el\s+|am\s+)?(?P<dayword>{_DAY_WORD})(?:\s+of|\s+de)?\s+"
               rf"(?P<month>{_MONTH})\.?{_YEAR}(?!\w)", re.IGNORECASE),
    re.compile(rf"\b{_WEEKDAY_PREFIX}(?P<month>{_MONTH})\.?\s+(?:the\s+)?(?P<dayword>{_DAY_WORD}){_YEAR}(?!\w)",
               re.IGNORECASE),
)
# Only for value keys (the guard), not for reading periods: a day and month with no year
# ("3/11", "3.11."), read both ways after a slash since "3/11" may be written either way.
_SHORT_NUMERIC_DATE = re.compile(
    r"(?<![\d/.:,-])(?P<first>\d{1,2})(?P<sep>[/.-])(?P<second>\d{1,2})(?:(?P=sep)(?P<year>\d{2}|\d{4}))?"
    r"(?![\d/]|[.,]\d)"
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
    """Dates written as ISO (2026-11-03), day-month ("Tue 3 Nov 2026", "3rd of November",
    "mardi 3 novembre"), month-day ("November 3rd, 2026", "November the 3rd") or
    day/month/year ("03/11/2026"), in text order. Month names are read in English,
    French, Spanish, Italian and German; spans refer to the original text."""
    normal = _Normal(text)
    found: list[DateMatch] = []
    for pattern in (*_DATE_PATTERNS, *_WORD_DATE_PATTERNS):
        for match in pattern.finditer(normal.text):
            parsed = _parse_date(match)
            if parsed is None:
                continue
            start, end = normal.span(parsed.start, parsed.end)
            if not _spans_overlap(found, start, end):
                found.append(replace(parsed, start=start, end=end))
    return sorted(found, key=lambda item: item.start)


def _parse_date(match: re.Match[str]) -> DateMatch | None:
    """Turn a date regex match into a DateMatch, or None if it is not a real date."""
    month_text = match.group("month")
    month = int(month_text) if month_text.isdigit() else _ALL_MONTHS[month_text.lower()]
    groups = match.groupdict()
    day = int(groups["day"]) if groups.get("day") else DAY_WORDS[re.sub(r"\s+", " ", groups["dayword"].lower())]
    year_text = match.groupdict().get("year")
    year = int(year_text) if year_text else None
    if year is not None and not 1 <= year <= 9998:  # year 0 is no date; 9999 leaves no room for date arithmetic
        return None
    if not 1 <= month <= 12 or not 1 <= day <= calendar.monthrange(year or 2024, month)[1]:
        return None
    return DateMatch(match.start(), match.end(), month, day, year)


# --- values ----------------------------------------------------------------

_TIME_24 = re.compile(
    r"(?<![\d:.,])(?P<hour>[01]?\d|2[0-3]) ?[:h] ?(?P<minute>[0-5]\d)(?::[0-5]\d(?:\.\d+)?)?(?![\d:])", re.I
)
_HRS = re.compile(r"(?<![\d:.,])(?P<hour>[01]?\d|2[0-3]) ?(?:hrs?|hours?) ?(?P<minute>[0-5]\d)(?!\d)", re.I)
_TIME_12 = re.compile(
    r"(?<![\d:.,])(?P<hour>1[0-2]|0?[1-9])(?::(?P<minute>[0-5]\d))?\s?(?P<half>[ap])\.?m\b\.?", re.I
)
_UNIT = r"(?:€|\$|£|%|(?:eur|euros?|usd|dollars?|gbp|pounds?|k)\b)"
_TIME_DOT = re.compile(rf"(?<![\d:.,])(?P<hour>[01]?\d|2[0-3])\.(?P<minute>[0-5]\d)(?![\d.,:])(?!\s?{_UNIT})", re.I)
_HOUR_ONLY = re.compile(
    r"(?<![\d:.,])(?P<hour>[01]?\d|2[0-3]) ?(?:h|heures?)(?:\s+(?P<part>et demie|et quart|moins le quart))?(?!\w)",
    re.I,
)
_OCLOCK = re.compile(r"(?<![\d:.,])(?P<hour>1[0-2]|0?[1-9]) ?o['’]? ?clock\b", re.I)
_UHR = re.compile(r"(?<![\d:.,])(?P<hour>[01]?\d|2[0-3]) ?Uhr(?: (?P<minute>[0-5]\d))?(?!\w)", re.I)
_EN_HOURS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "noon": 12, "midday": 12, "midnight": 0,
}
_FR_HOURS = {
    "une": 1, "un": 1, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5, "six": 6, "sept": 7, "huit": 8,
    "neuf": 9, "dix": 10, "onze": 11, "douze": 12, "treize": 13, "quatorze": 14, "quinze": 15,
    "seize": 16, "dix-sept": 17, "dix-huit": 18, "dix-neuf": 19, "vingt": 20, "vingt et une": 21,
    "vingt-deux": 22, "vingt-trois": 23, "midi": 12, "minuit": 0,
}
_EN_MINUTES = {"o'clock": 0, "o’clock": 0, "oclock": 0, "fifteen": 15, "thirty": 30, "forty-five": 45,
               "forty five": 45, "twenty": 20, "forty": 40, "fifty": 50}
_FR_MINUTES = {"et demie": 30, "et quart": 15, "trente": 30, "quinze": 15, "quarante-cinq": 45,
               "quarante cinq": 45, "vingt": 20, "dix": 10, "moins le quart": -15}
_EN_HOUR = _alternation(_EN_HOURS)
_FR_HOUR = _alternation(_FR_HOURS)
_WORD_TIMES = (
    (re.compile(rf"\bhalf past (?P<hour>{_EN_HOUR})\b", re.I), 30),
    (re.compile(rf"\b(?:a )?quarter past (?P<hour>{_EN_HOUR})\b", re.I), 15),
    (re.compile(rf"\b(?:a )?quarter to (?P<hour>{_EN_HOUR})\b", re.I), -15),
    (re.compile(rf"\b(?P<hour>{_EN_HOUR})[ -](?P<minute>{_alternation(_EN_MINUTES)})(?!\w)", re.I), None),
    (re.compile(rf"\b(?P<hour>{_FR_HOUR}) heures?(?: (?P<minute>{_alternation(_FR_MINUTES)}))?(?!\w)", re.I), None),
)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_NUMBER = re.compile(r"(?<![\w.,])[€$£]?\+?\d(?:[\d   ,.'’-]*\d)?(?!\w)")
_WORD = re.compile(r"[^\W\d_][\w'’-]*")
_SENTENCE_BREAK = ".!?:;\n•*-"

KIND_PHRASES = {
    "date": "a date",
    "day": "a day of the month",
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


# A day of the month without its month ("paid on the 5th of each month", "on the fifth"): a value of its own,
# so a private payment day typed back ("rent on the 5th") is withheld like a date. Found by Nemotron's round 9.
_ORDINAL_DAY_WORDS = sorted(
    (word for word in DAY_WORDS if re.fullmatch(r"[a-z]+(?:-[a-z]+)?", word) and word.endswith(("st", "nd", "rd", "th"))),
    key=len, reverse=True,
)
_MONTH_DAY = re.compile(
    r"(?<![\w/.:-])(?P<num>[1-9]|[12]\d|3[01])(?:st|nd|rd|th)(?![\w-])"
    rf"|\bthe\s+(?P<word>{'|'.join(map(re.escape, _ORDINAL_DAY_WORDS))})(?![\w-])"
    r"|\b(?:le|el)\s+(?:d[ií]a\s+)?(?P<lat>[1-9]|[12]\d|3[01])\s+(?:du|de\s+chaque|de\s+cada)\s+(?:mois|mes)\b",
    re.I,
)


def find_values(text: str) -> list[Value]:
    """Dates, days of the month written alone ("the 5th"), times, e-mail addresses, amounts and other numbers
    (three digits or more, years excluded) and capitalised names that do not start a sentence, in text order.

    A span that can be read several ways ("3/11", "10.30", "half past ten") carries one
    value per reading, so the guard catches a private value whichever was meant.
    """
    normal = _Normal(text)
    norm = normal.text
    values = [
        Value("date", f"date:{found.month:02d}-{found.day:02d}", found.start, found.end)
        for found in find_dates(text)
    ]

    def add(kind: str, keys: Iterable[str], start: int, end: int, *, alongside: bool = False) -> None:
        """Keep the readings of a (normalised) span unless an earlier, more specific value
        overlaps it; with ``alongside``, also add them next to earlier values of exactly that span."""
        start, end = normal.span(start, end)
        overlapping = [value for value in values if value.start < end and start < value.end]
        if overlapping and not (alongside and all((v.start, v.end) == (start, end) for v in overlapping)):
            return
        known = {value.key for value in overlapping}
        values.extend(Value(kind, key, start, end) for key in dict.fromkeys(keys) if key not in known)

    for match in _SHORT_NUMERIC_DATE.finditer(norm):  # "3/11", "03/11/2026" read both ways
        first, second = int(match["first"]), int(match["second"])
        readings = [(second, first)] + ([(first, second)] if match["sep"] in "/-" else [])
        keys = [f"date:{month:02d}-{day:02d}" for month, day in readings if _real_day(month, day)]
        if keys:
            add("date", keys, *match.span(), alongside=True)
    for match in _TIME_12.finditer(norm):  # before 24-hour times, so "2:15 pm" is 14:15
        hour = int(match["hour"]) % 12 + (12 if match["half"].lower() == "p" else 0)
        add("time", [f"time:{hour:02d}:{match['minute'] or '00'}"], *match.span())
    for match in _TIME_24.finditer(norm):
        add("time", [f"time:{int(match['hour']):02d}:{match['minute']}"], *match.span())
    for match in _HRS.finditer(norm):  # "10 hrs 30"
        add("time", [f"time:{int(match['hour']):02d}:{match['minute']}"], *match.span())
    for match in _TIME_DOT.finditer(norm):  # "10.30": a time, and also a short date if it reads as one
        add("time", [f"time:{int(match['hour']):02d}:{match['minute']}"], *match.span(), alongside=True)
    for match in _HOUR_ONLY.finditer(norm):  # "11h", "11 heures", "10 h et demie"
        hour, minute = int(match["hour"]), _FR_MINUTES.get((match["part"] or "").lower(), 0)
        add("time", [_clock(hour, minute)], *match.span())
    for match in _OCLOCK.finditer(norm):
        hour = int(match["hour"])
        add("time", [_clock(hour, 0), _clock(hour + 12, 0)], *match.span())
    for match in _UHR.finditer(norm):  # "10 Uhr 30", "11 Uhr"
        add("time", [_clock(int(match["hour"]), int(match["minute"] or 0))], *match.span())
    for pattern, minute in _WORD_TIMES:
        for match in pattern.finditer(norm):
            hour_word = match["hour"].lower()
            hour = _EN_HOURS.get(hour_word, _FR_HOURS.get(hour_word, 0))
            spoken = (match.groupdict().get("minute") or "").lower()
            minutes = minute if minute is not None else _EN_MINUTES.get(spoken, _FR_MINUTES.get(spoken, 0))
            keys = [_clock(hour, minutes)] + ([_clock(hour + 12, minutes)] if hour < 12 else [])
            add("time", keys, *match.span())
    for match in _MONTH_DAY.finditer(norm):  # after dates and times: "5th November" is a date
        raw = match["num"] or match["lat"]
        add("day", [f"day:{int(raw) if raw else DAY_WORDS[match['word'].lower()]}"], *match.span())
    for match in _EMAIL.finditer(norm):
        add("email", [f"email:{match.group().lower()}"], *match.span())
    for match in _THOUSANDS.finditer(norm):  # "1.18k", "6k"
        amount = round(float(match["amount"].replace(",", ".")) * 1000)
        if amount >= 100:
            add("number", [f"number:{amount}"], *match.span())
    for match in _NUMBER.finditer(norm):
        raw = match.group()
        key = _number_key(raw)
        is_year = raw.isdigit() and len(raw) == 4 and 1900 <= int(raw) <= 2100
        if key is not None and not is_year:
            whole = key.split(".")[0]  # "1,180.50" is also read as 1180, close enough to give a private 1,180 away
            add("number", [f"number:{key}"] + ([f"number:{whole}"] if "." in key and len(whole) >= 3 else []),
                *match.span())
    for start, end, amount in _number_words(norm):
        if amount >= 100 and not 1900 <= amount <= 2100:
            add("number", [f"number:{amount}"], start, end)
    for match in _WORD.finditer(norm):
        name = _name_at(norm, match)
        if name:
            add("name", [f"name:{fold(name)[0]}"], match.start(), match.start() + len(name))
    return sorted(values, key=lambda value: (value.start, value.key))


def _real_day(month: int, day: int) -> bool:
    """True when ``day`` exists in ``month`` (29 February counts)."""
    return 1 <= month <= 12 and 1 <= day <= calendar.monthrange(2024, month)[1]


def _clock(hour: int, minute: int) -> str:
    """The time key of ``hour``:``minute``, wrapping negative minutes ("quarter to") to the hour before."""
    if minute < 0:
        hour, minute = hour - 1, 60 + minute
    return f"time:{hour % 24:02d}:{minute:02d}"


_AMOUNT = re.compile(r"(?P<whole>\d{1,3}(?:(?P<sep>[ ,.'’])\d{3})+|\d+)(?:(?P<point>[.,])(?P<cents>\d{1,2}))?")
_THOUSANDS = re.compile(r"(?<![\w.,])(?P<amount>\d{1,3}(?:[.,]\d{1,2})?) ?k(?!\w)", re.I)


def _number_key(raw: str) -> str | None:
    """The key of a number as written: an amount ("1 180,00 €", "€1,180.00", "1.180") is
    keyed by its value, with cents only when they are not zero ("11,80" → "11.80");
    anything else (a phone number, "06 12 34 56 78") by all its digits. None when it
    has fewer than three digits."""
    body = raw.lstrip("€$£+").strip()
    match = _AMOUNT.fullmatch(body)
    if match and not (match["sep"] and match["point"] and match["sep"] == match["point"]):
        whole = re.sub(r"\D", "", match["whole"])
        cents = (match["cents"] or "").ljust(2, "0") if match["cents"] else ""
        if len(whole + cents.rstrip("0")) < 3 and len(whole) < 3:
            return None
        return f"{int(whole)}.{cents}" if cents and int(cents) else str(int(whole))
    digits = re.sub(r"\D", "", raw)
    return digits if len(digits) >= 3 else None


_EN_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_FR_NUMBERS = {
    "un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5, "six": 6, "sept": 7, "huit": 8,
    "neuf": 9, "dix": 10, "onze": 11, "douze": 12, "treize": 13, "quatorze": 14, "quinze": 15,
    "seize": 16, "vingt": 20, "vingts": 20, "trente": 30, "quarante": 40, "cinquante": 50,
    "soixante": 60, "quatre-vingt": 80, "quatre-vingts": 80,
}
_HUNDREDS = {"hundred": 100, "cent": 100, "cents": 100}
_SCALES = {"thousand": 1000, "mille": 1000, "million": 1_000_000, "millions": 1_000_000}
_NUMBER_WORD = _alternation([*_EN_NUMBERS, *_FR_NUMBERS, *_HUNDREDS, *_SCALES])
_NUMBER_RUN = re.compile(
    rf"\b(?:{_NUMBER_WORD})(?:(?:[ -]|[ -]?(?:and|et)[ -])(?:{_NUMBER_WORD}))*\b", re.I
)


def _number_words(text: str) -> list[tuple[int, int, int]]:
    """Numbers written in English or French words, as (start, end, value): "one thousand
    one hundred and eighty", "eleven hundred and eighty", "mille cent quatre-vingts"."""
    found = []
    for match in _NUMBER_RUN.finditer(text):
        phrase = re.sub(r"quatre-vingts?", "quatre-vingt", match.group().lower())
        tokens = [token for token in re.split(r"[ -]+", phrase.replace("quatre-vingt", "quatre_vingt"))
                  if token not in ("and", "et")]
        total = current = 0
        for token in tokens:
            token = token.replace("quatre_vingt", "quatre-vingt")
            if token in _HUNDREDS:
                current = (current or 1) * 100
            elif token in _SCALES:
                total += (current or 1) * _SCALES[token]
                current = 0
            else:
                current += _EN_NUMBERS.get(token, _FR_NUMBERS.get(token, 0))
        found.append((match.start(), match.end(), total + current))
    return found


_CUE_WORDS = (frozenset(fold(word)[0] for word in [*CALENDAR_WORDS, *DAY_WORDS]) | frozenset(
    ["noon", "midday", "midnight", "midi", "minuit", "hundred", "thousand", "million", "cent", "mille", "o'clock",
     "oclock", "uhr"]))


def has_value_cues(text: str) -> bool:
    """True when ``text`` holds anything that may be a date, a time or an amount, however it is
    written: a digit, a month or weekday name, a day or number word. Broader than find_values
    on purpose: used to keep such messages away from a cloud router."""
    normal = _Normal(text).text
    if re.search(r"\d", normal):
        return True
    tokens = re.findall(r"[^\W\d_]+(?:['’-][^\W\d_]+)*", fold(normal)[0])
    return any(token in _CUE_WORDS or token in _EN_NUMBERS or token in _FR_NUMBERS for token in tokens)


# --- digit signatures --------------------------------------------------------
# A value can also be written as bare digits ("1030", "20261103", "2026/11/03") or digit by
# digit in words ("one one eight zero"). The guard compares such runs with the digit
# signatures of each protected value, whatever separators or words carry them.

_DIGIT_WORDS = {
    "zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "zéro": "0", "un": "1", "une": "1", "deux": "2", "trois": "3",
    "quatre": "4", "cinq": "5", "sept": "7", "huit": "8", "neuf": "9", "null": "0", "eins": "1",
    "zwei": "2", "drei": "3", "vier": "4", "fünf": "5", "fuenf": "5", "sechs": "6", "sieben": "7",
    "acht": "8", "neun": "9", "cero": "0", "uno": "1", "dos": "2", "tres": "3", "cuatro": "4",
    "cinco": "5", "seis": "6", "siete": "7", "ocho": "8", "nueve": "9",
}
_DIGIT_RUN = re.compile(r"(?<![\w.,])\d+(?:[ ./:-]\d+)*(?![\w])")
_DIGIT_WORD = _alternation(_DIGIT_WORDS)
_DIGIT_WORD_RUN = re.compile(rf"\b(?:{_DIGIT_WORD})(?:[ ,-]+(?:{_DIGIT_WORD})){{2,}}\b", re.I)


def digit_runs(text: str) -> list[tuple[int, int, str]]:
    """Runs of digits in ``text`` (with single separators between groups) and runs of three
    or more digit words, as (start, end, digits), spans in the original text."""
    normal = _Normal(text)
    runs = []
    for match in _DIGIT_RUN.finditer(normal.text):
        runs.append((*normal.span(*match.span()), re.sub(r"\D", "", match.group())))
    for match in _DIGIT_WORD_RUN.finditer(normal.text):
        digits = "".join(_DIGIT_WORDS[word.lower()] for word in re.split(r"[ ,-]+", match.group()) if word)
        runs.append((*normal.span(*match.span()), digits))
    return runs


def signatures(text: str) -> set[str]:
    """Digit strings of four or more digits that stand for the dates, times and whole amounts
    in ``text``: 3 Nov 2026 gives 1103, 0311, 20261103, 03112026 and 11032026; 10:30 gives 1030."""
    found: set[str] = set()
    for day in find_dates(text):
        md, dm = f"{day.month:02d}{day.day:02d}", f"{day.day:02d}{day.month:02d}"
        found |= {md, dm}
        if day.year:
            found |= {f"{day.year}{md}", f"{dm}{day.year}", f"{md}{day.year}"}
    for value in find_values(text):
        if value.kind == "time":
            found.add(value.folded.replace(":", ""))
        elif value.kind == "number" and "." not in value.folded:
            found.add(value.folded)
    return {digits for digits in found if len(digits) >= 4}


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


# Words that introduce a PIN or a numeric code, in English, French, German and Spanish. The code itself may be
# written with gaps ("4 8 2 1", "12-34-56") or digit by digit in words ("four eight two one").
_CODE_WORDS = (  # not "phone code" (a country's calling code) nor "pass code" ("must pass code review")
    # "pin" the verb is not one: "PIN the slides", "pin it", "a PIN-up poster"
    r"pin(?:\s*(?:code|number)|(?![\s-]+(?:the|a|an|this|that|these|those|it|them|up|down)\b))|passcode|sim\s*pin"
    r"|puk(?:\s*code)?"
    r"|(?:door|gate|alarm|garage|access|security|unlock|entry|keypad|building|intercom)\s*(?:opener\s+|remote\s+)?code"
    r"|(?:debit|credit|bank|atm)?\s*card\s*(?:code|pin)|(?:cash\s*machine|cash\s*point|atm)\s+code"
    r"|(?:four|six|4|6)[ -]digits?\s+(?:for|of)\s+(?:the\s+|my\s+)?(?:bank\s+|debit\s+|credit\s+)?card"
    r"|code\s+(?:for|of|to)\s+(?:the\s+|my\s+)?(?:front\s+|back\s+|main\s+)?(?:door|gate|building|garage|alarm"
    r"|safe|lock|locker|padlock|(?:bank\s+|debit\s+|credit\s+)?card)"
    # not a business named after it: "the code for the alarm company is ACME-2026"
    r"(?!\s+(?:company|companies|firm|shop|store|installer|provider|engineer|technician|repairs?|maker|service"
    r"|services|people|man|guy)\b)"
    r"|lock\s*(?:screen\s*)?code|cvv2?|cvc|cryptogramme(?:\s+visuel)?|digicode"
    r"|code\s+(?:pin|secret|confidentiel|d['’]acc[eè]s|de\s+(?:la\s+)?(?:porte|carte)|de\s+s[ée]curit[ée])"
    r"|geheimzahl|pin-?nummer|zugangscode|t[üu]rcode|c[óo]digo\s+(?:pin|de\s+acceso|de\s+seguridad|secreto)"
    # Italian and Portuguese: "il codice della mia carta", "il codice del bancomat", "o código do meu cartão"
    r"|codice\s+(?:pin|segreto|di\s+accesso|di\s+sblocco|del\s+(?:bancomat|portone|lucchetto|cancello|citofono)"
    r"|della\s+(?:mia\s+)?(?:carta|porta|cassaforte)|dell['’]allarme)"
    r"|c[óo]digo\s+(?:de\s+acesso|de\s+desbloqueio|do\s+(?:meu\s+)?(?:cart[ãa]o|alarme|port[ãa]o|cadeado)"
    r"|da\s+(?:minha\s+)?(?:porta|fechadura))"
    r"|(?:safe|lock|locker|padlock|suitcase|bike\s*lock)\s+(?:combination|combo|code)"
    r"|combination\s+(?:to|for|of)\s+(?:the|my)\s+(?:safe|lock|locker|padlock|suitcase|bike\s*lock)"
)
# Digits in words for a code, Italian and Portuguese included ("quattro otto due uno"); only here, since "due" or "sei"
# alone is an ordinary word elsewhere
_CODE_DIGIT_WORD = _alternation({*_DIGIT_WORDS, "uno", "due", "tre", "quattro", "cinque", "sei", "sette", "otto",
                                 "nove", "um", "uma", "dois", "duas", "três", "tres", "quatro", "sete", "oito"})
_CODE_VALUE = (  # gaps of a space or a dash; not dots, which make versions, decimals and times ("2.1.0", "9.30")
    r"(?<!\d)\d(?:[ -]?\d){2,7}(?!\d|[.,]\d)"
    rf"|\b(?:{_CODE_DIGIT_WORD})(?:[ ,-]+(?:{_CODE_DIGIT_WORD})){{2,7}}\b"
)
# "is" in the languages the code words are read in, Italian and Portuguese included ("è", "é")
_COPULA = r"is|was|est|ist|es|lautet|era|è|é"
_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("a password", re.compile(
        r"\b(?:password|passwd|passphrase|pwd|pw|mots?\s+de\s+passe|mdp|passwort|kennwort|contrase[ñn]a|senha"
        r"|parola\s+d['’]ordine|palavra-?passe"
        r"|wi-?fi\s+(?:key|code)|cl[ée]\s+(?:wi-?fi|wpa)|code\s+wi-?fi)s?\b"
        r"(?!\s+(?:manager|policy|policies|reset|rules?|hints?|strength|field|prompt|change|expiry|generator"
        r"|requirements?|protected|tips?|advice|training|guide|ideas?|reminders?|rotation|best\s+practices?)\b)"
        # "is", or a colon right after the keyword or after a few words with no digit ("pw for the router:"),
        # never one inside a time ("training on Tuesday at 14:00")
        r"(?:[^.\n]{0,30}?\b(?:is|was|est|ist|lautet|es|era|è|é)\b|[^.\n:=\d]{0,30}?[:=])\s*"
        r"(?!(?:strong|weak|wrong|correct|incorrect|expired|required|missing|invalid|too|the|a|an|not|still"
        r"|written|on|in|at|under|behind|taped|printed|kept|somewhere|available|shared"
        r"|saved|stored|changed|reset|new|trop|tr[eè]s|faible|fort|oubli[ée]e?|perdu|chang[ée]|expir[ée]|le|la"
        r"|les|un|une|zu|sehr|falsch|vergessen|ge[äa]ndert|abgelaufen|neu|der|die|das|ein|eine|muy|demasiado"
        r"|olvidada|incorrecta|nueva|el|una)\b)\S{3,}",
        re.I)),
    ("a PIN or access code", re.compile(rf"\b(?:{_CODE_WORDS})\b\D{{0,20}}?(?:{_CODE_VALUE})", re.I)),
    # letters inside, right after "is" or a colon: "the digicode is 12A45"
    ("a PIN or access code", re.compile(
        rf"\b(?:{_CODE_WORDS})\b[^\n\d.]{{0,20}}?(?:\b(?:{_COPULA})\b|[:=])\s*"
        r"(?<![\w-])(?=[A-Za-z0-9]{4,10}(?![\w-]))(?=(?:[A-Za-z]*\d){3})(?=[0-9]*[A-Za-z])[A-Za-z0-9]+", re.I)),
    # dotted, right after "is" or a colon: "my PIN is 4.8.2.1", "alarm code: 45.12" (not "pin the 2.1.0 notes")
    ("a PIN or access code", re.compile(
        rf"\b(?:{_CODE_WORDS})\b[^\n\d.]{{0,20}}?(?:\b(?:{_COPULA})\b|[:=])\s*"
        r"(?<!\d)\d{1,4}(?:\.\d{1,4}){1,7}(?!\d|\.\d)", re.I)),
    # a longer description between the word and "is": "the PIN for the Barclays card is 4821", "my bank PIN, the one I
    # chose at the branch last spring, is 4821"; no sentence break, digit or other "is" in between, so "the gate code
    # is on the fridge, my flight number is 4821" is not one
    ("a PIN or access code", re.compile(
        rf"\b(?:{_CODE_WORDS})\b(?:(?!\b(?:{_COPULA})\b)[^\n\d.;:!?]){{0,60}}?\b(?:{_COPULA})\b\s*(?:{_CODE_VALUE})",
        re.I)),
    # single digits joined by underscores or slashes, right after "is" or a colon: "my PIN is 4_8_2_1", "4/8/2/1"
    # (not a date: "expires on 12/2027")
    ("a PIN or access code", re.compile(
        rf"\b(?:{_CODE_WORDS})\b[^\n\d.;:!?]{{0,40}}?(?:\b(?:{_COPULA})\b|[:=])\s*(?<![\d/_])\d(?:[_/]\d){{2,7}}(?![\d/_])",
        re.I)),
    # in number words, three or more right after "is" or a colon: "forty-eight twenty-one", "four thousand eight
    # hundred twenty one", "forty eight, twenty one" (not "arrives in forty-eight hours", not "twenty-one")
    ("a PIN or access code", re.compile(
        rf"\b(?:{_CODE_WORDS})\b(?:(?!\b(?:{_COPULA})\b)[^\n\d.;:!?]){{0,40}}?(?:\b(?:{_COPULA})\b|[:=])\s*"
        rf"(?:{_NUMBER_WORD})(?:[ ,-]+(?:and[ -]+)?(?:{_NUMBER_WORD})){{2,11}}\b", re.I)),
    # the value first: "4821 is my bank PIN", "Remember 4821, it's my bank PIN", "4821 - that's the door code",
    # "le 4821, c'est mon code de carte"; only a few words may sit between "my" or "the" and the code word, so "1500 is
    # the time the door code changes" is not one
    ("a PIN or access code", re.compile(
        rf"(?<![\w.,])(?:{_CODE_VALUE})\s*(?:[,:;—–-]\s*)?"
        r"(?:(?:is|was|it['’]s|that['’]s|this\s+is|c['’]est|est|ist|es|è|é)\s+)?"
        r"(?:my|the|our|his|her|your|mon|ma|le|la|notre|mein|meine|die|der|mi|el|il\s+mio|la\s+mia|o\s+meu"
        r"|a\s+minha)\s+"
        r"(?:(?:bank|banking|debit|credit|atm|sim|phone|mobile|card|new|old|current|usual|front|back|main|work"
        r"|office|home|own|secret|real|actual|(?:four|six|4|6)-digit)\s+){0,2}"
        rf"(?:{_CODE_WORDS})\b", re.I)),
    ("a login and its password", re.compile(
        r"\b(?:login|log-in|logon|sign-?in|credentials|creds|identifiants?|zugangsdaten|anmeldedaten)\b"
        r"(?:[^.\n]{0,30}?\b(?:is|are|est|sont|ist|sind)\b|[^.\n:=\d]{0,30}?[:=])\s*"
        r"(?=[^\s/]*[A-Za-z@])[^\s/]+\s*/\s*\S{3,}",
        re.I)),
    ("a payment card number", re.compile(
        r"\b(?:card|carte|tarjeta|karte)\s*(?:number|no\.?|num[ée]ro|n°|#|nummer|n[úu]mero)"
        r"\D{0,20}?(?<!\d)\d(?:[ -]?\d){11,18}(?!\d)",
        re.I)),
    ("a card security code", re.compile(
        # only about a card: "the 3 digits on the back of my card", "the CB expires 09/28 and the 3 digits on
        # the back are 123"; not a race bib
        r"\b(?:card|carte|cb|visa|mastercard|amex)\b[^.\n]{0,60}?\b(?:digits|numbers|chiffres)\s+(?:on|at|au)\s+"
        r"(?:the\s+)?(?:back|dos)\b[^.\n]{0,40}?(?<!\d)\d{3,4}(?!\d)"
        r"|\b(?:digits|numbers|chiffres)\s+(?:on|at|au)\s+(?:the\s+)?(?:back|dos)\s+(?:of|de)\s+(?:my|the|your|ma|la)\s+"
        r"(?:\w+\s+)?(?:card|carte|cb|visa|mastercard|amex)\b[^.\n]{0,40}?(?<!\d)\d{3,4}(?!\d)",
        re.I)),
    ("a wallet recovery phrase", re.compile(
        r"\b(?:recovery|seed|mnemonic|secret|backup|wallet)\s+(?:phrase|words|seed)\b[^\n]{0,20}?"
        r"(?:\b[a-z]{3,8}\b[ ,]+){11,23}\b[a-z]{3,8}\b",
        re.I)),
    ("an identity number", re.compile(
        r"\b(?:ssn|social\s+security(?:\s+(?:number|no\.?))?|national\s+insurance(?:\s+number)?"
        r"|passport\s+(?:number|no\.?)|num[ée]ro\s+de\s+(?:s[ée]curit[ée]\s+sociale|s[ée]cu|passeport)|tax\s+id)\b"
        # the number itself: digits with single gaps, at least six, after at most two letters ("AB 12 34 56")
        r"\D{0,20}?(?<![\w-])(?:[A-Za-z]{1,2}[ -]?)?\d(?:[ -]?\d){5,15}(?![\w-])",
        re.I)),
    # a passport number with letters inside: "12AB34567", "19FV12345"
    ("an identity number", re.compile(
        r"\b(?:passport\s+(?:number|no\.?|n°|#)|num[ée]ro\s+de\s+passeport)\D{0,20}?(?<![\w-])"
        r"(?=[A-Za-z0-9]{6,12}(?![\w-]))(?=(?:[A-Za-z]*\d){4})[A-Za-z0-9]+",
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

    normal = _Normal(text)
    for reason, pattern in _SECRET_PATTERNS:
        for match in pattern.finditer(normal.text):
            add(*normal.span(*match.span()), reason)
    for match in _IBAN.finditer(normal.text):
        compact = match.group().replace(" ", "")
        if 15 <= len(compact) <= 34 and iban_valid(compact):
            add(*normal.span(*match.span()), "a bank account number (an IBAN with a valid checksum)")
    for match in _CARD.finditer(normal.text):
        if luhn_valid(re.sub(r"\D", "", match.group())):
            add(*normal.span(*match.span()), "a payment card number (it passes the Luhn check)")
    return sorted(found, key=lambda secret: secret.start)


def detect_secret(text: str) -> str | None:
    """Explain what credential ``text`` seems to contain, or return None if none."""
    secrets = find_secrets(text)
    return secrets[0].reason if secrets else None
