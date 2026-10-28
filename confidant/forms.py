"""Written forms of protected values, generated instead of parsed.

:mod:`confidant.scan` *reads* values in the forms it was written for. This module works the other way
round: for each value the guard protects (the day and month of a date, a time of day, an amount from 100 to
999,999) it *writes* the value out in the forms people use: the CLDR date and time formats of 82 locales
(through Babel), number words in the 44 languages of num2words, and spoken idioms ("half ten", "eleven
sharp", "six grand", "eleven-eighty", "halb elf", "a las diez y media", "03NOV", "3.XI.2026"). The guard then
looks for those exact forms in outbound text. Forms without a letter (``10:30``, ``03/11``) are left to the
scanners and digit signatures.

Generating is the inversion that makes the coverage wide: a parser has to understand every way of writing
every value, while the guard only protects a handful of values, each of which can be written out in a few
hundred to a thousand ways ahead of time. Matching keeps to whole words, stops at sentence ends, and skips a
form that is part of a larger number ("vingt-trois novembre", "seis mil quinientos") or of another time
("huit heures et demie"). Very short forms that are everyday words in some language ("3 set", "mil", "bin")
are not generated. A form that is not generated here still gets through; the leak test
(``docs/leak-test.md``) keeps the score.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from calendar import monthrange
from collections.abc import Iterable
from datetime import date, time
from functools import lru_cache

try:  # declared dependencies; without them the guard keeps the scanners and says so (see AVAILABLE)
    from babel.dates import format_date, format_skeleton, format_time
    from num2words import CONVERTER_CLASSES, num2words
except ImportError:  # pragma: no cover - exercised only on an incomplete install
    CONVERTER_CLASSES = {}
    AVAILABLE = False
else:
    AVAILABLE = True

_log = logging.getLogger(__name__)

# CLDR locales whose date and time formats are written out, with a regional variant only where its formats
# differ from the language's default.
LOCALES = tuple(
    "af am ar az be bg bn bs ca cs cy da de de_AT el en en_GB en_IN es es_MX et eu fa fi fil fr fr_CA ga gl gu "
    "he hi hr hu hy id is it ja ka kk km kn ko ky lo lt lv mk ml mn mr ms my nb ne nl pa pl pt pt_PT ro ru si "
    "sk sl sq sr sr_Latn sv sw ta te th tr uk ur uz vi zh zh_Hant zu".split()
)
# num2words 0.5.14's Amharic converter never returns for numbers of seven digits or more.
_SKIP_NUMBER_LANGUAGES = frozenset({"am"})
NUMBER_LANGUAGES = tuple(sorted(set(CONVERTER_CLASSES) - _SKIP_NUMBER_LANGUAGES))
MAX_SPOKEN = 999_999  # larger amounts are not written out in words (nobody says a phone number as one number)

_DASHES = frozenset("-‐‑‒–—")
_SPACES = frozenset("_,'’`\"«»„“”()[]{}/\\|")
_HARD_STOPS = frozenset("!?;¡¿…\n\r  ")
BOUNDARY = "\n"  # a sentence or line end in canonical text; no form crosses it
_NO_SPACE_SCRIPTS = ("CJK", "HIRAGANA", "KATAKANA", "HANGUL", "THAI", "LAO", "KHMER", "MYANMAR")
_CJK_NUMERALS = "0-9〇一二三四五六七八九十百千万萬億亿兆两零"
_ROMAN = ("i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii")
_EN_MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
_EN_MONTH_TOKENS = frozenset(_EN_MONTHS) | {"sept", "june", "july"}
# Words that follow an hour in a longer time, so "huit heures" in "huit heures et demie" is not 8:00.
_MINUTE_CONTINUATIONS = frozenset(
    "et y e und and past quart demie demi media cuarto mezza mezzo meia kwart half".split()
)
# Articles that are also the word for one; never treated as part of a neighbouring number.
_ARTICLES = frozenset("one ein eine einer uno una une uma een".split())


@lru_cache(maxsize=1)
def _warn_once() -> None:
    _log.warning("babel or num2words is missing: protected values are matched by the scanners only")


def canon(text: str) -> tuple[str, list[int]]:
    """``text`` as forms are matched: format characters dropped, NFKC, every decimal digit in ASCII,
    case-folded, accents stripped, dashes as ``-``, other punctuation and whitespace runs as one space,
    and a sentence or line end (``!``, ``?``, ``;``, ``: `` , a line break, or a full stop after a word and
    before a capital) as :data:`BOUNDARY`; with, for each output character, the index of the original
    character it came from."""
    chars: list[str] = []
    origin: list[int] = []

    def push(char: str, index: int) -> None:
        if char == " ":
            if chars and chars[-1] not in (" ", BOUNDARY):
                chars.append(" ")
                origin.append(index)
        elif char == BOUNDARY:
            if chars and chars[-1] == " ":
                chars[-1] = BOUNDARY
                origin[-1] = index
            elif chars and chars[-1] != BOUNDARY:
                chars.append(BOUNDARY)
                origin.append(index)
        else:
            chars.append(char)
            origin.append(index)

    for index, char in enumerate(text):
        if unicodedata.category(char) == "Cf":
            continue
        for piece in unicodedata.normalize("NFKC", char):
            digit = unicodedata.decimal(piece, None)
            if digit is not None:
                push(str(digit), index)
            elif piece in _HARD_STOPS:
                push(BOUNDARY, index)
            elif piece == ":":
                push(BOUNDARY if text[index + 1: index + 2].isspace() else " ", index)
            elif piece == ".":
                after = text[index + 1: index + 3]
                sentence_end = (chars and chars[-1].isalpha() and len(after) == 2 and after[0].isspace()
                                and after[1].isupper())
                push(BOUNDARY if sentence_end else " ", index)
            elif piece in _DASHES:
                push("-", index)
            elif piece.isspace() or piece in _SPACES:
                push(" ", index)
            else:
                for part in unicodedata.normalize("NFKD", piece.casefold()):
                    if not unicodedata.combining(part):
                        push(part, index)
    return "".join(chars), origin


def _words(number: int, lang: str, **options: str) -> str | None:
    """``number`` in words in ``lang``, or None where num2words cannot say it."""
    if lang in _SKIP_NUMBER_LANGUAGES or number > MAX_SPOKEN:
        return None
    try:
        return num2words(number, lang=lang, **options)
    except Exception:  # noqa: BLE001 - num2words raises many kinds of errors for unsupported cases
        return None


def _en(number: int) -> str:
    return num2words(number, lang="en")


def _letters(form: str) -> list[str]:
    """The letter runs of a canonical form ("03nov" gives ["nov"])."""
    return re.findall(r"[^\W\d_]+", form)


# --- dates ---------------------------------------------------------------------------------------------

def _date_forms(month: int, day: int, years: Iterable[int]) -> set[str]:
    """A day of a month written out: CLDR formats in every locale (with and without the year and the
    weekday), the day in words, compact military forms ("03NOV") and Roman-numeral months with a year
    ("3.XI.2026")."""
    forms: set[str] = set()
    leap = 2024 if (month, day) == (2, 29) else 2026
    base = date(leap, month, day)
    for locale in LOCALES:
        language = locale.split("_")[0]
        for skeleton in ("MMMd", "MMMMd"):
            written = format_skeleton(skeleton, base, locale=locale)
            forms.add(written)
            if language not in CONVERTER_CLASSES:
                continue
            for word in {_words(day, language), _words(day, language, to="ordinal")} - {None}:
                spelled = re.sub(rf"(?<!\d){day}(?!\d)", word, written, count=1)
                forms.add(spelled)
                if language in ("ja", "ko", "zh"):  # "十一月三日", "십일월 삼일"
                    month_word = _words(month, language)
                    if month_word:
                        forms.add(re.sub(rf"(?<!\d){month}(?!\d)", month_word, spelled, count=1))
    for year in sorted(set(years)):
        try:
            day_of_year = date(year, month, day)
        except ValueError:
            continue
        for locale in LOCALES:
            for skeleton in ("MMMEd", "MMMMEEEEd", "yMMMd", "yMMMMd", "yMMMEd", "yMMMMEEEEd"):
                forms.add(format_skeleton(skeleton, day_of_year, locale=locale))
            for length in ("medium", "long", "full"):
                forms.add(format_date(day_of_year, length, locale=locale))
    english, roman = _EN_MONTHS[month - 1], _ROMAN[month - 1]
    for number in {str(day), f"{day:02d}"}:
        forms |= {f"{number}{english}", f"{english}{number}"}
        for year in set(years):
            forms |= {f"{number}{english}{year % 100:02d}", f"{number}{english}{year}",
                      f"{number}.{roman}.{year}", f"{number} {roman} {year}", f"{number}/{roman}/{year}"}
    return {form for form in forms if _specific_date(canon(form)[0])}


def _specific_date(form: str) -> bool:
    """False for a date form whose words are all short and are everyday words somewhere ("3 set", "3 mac",
    "15 pro"): at least one word must have four letters or more, be an English month abbreviation, be
    written in a script without spaces, or be a Roman-numeral month next to a four-digit year."""
    words = _letters(form)
    if _unbounded(form) or any(len(word) >= 4 or word in _EN_MONTH_TOKENS for word in words):
        return True
    return bool(re.search(r"\d{4}", form)) and any(word in _ROMAN for word in words)


# --- times ---------------------------------------------------------------------------------------------

def _time_forms(hour: int, minute: int) -> set[str]:
    """A time of day written out: CLDR short and medium formats in every locale that writes a word or an
    am/pm marker, and spoken forms in English, French, German, Spanish, Italian, Portuguese and Dutch."""
    forms = {format_time(time(hour, minute), length, locale=locale)
             for locale in LOCALES for length in ("short", "medium")}
    h12, after = hour % 12 or 12, (hour + 1) % 12 or 12
    for speaker in (_english_times, _french_times, _german_times, _spanish_times, _italian_times,
                    _portuguese_times, _dutch_times):
        forms |= speaker(hour, minute, h12, after)
    return forms


def _english_times(hour: int, minute: int, h12: int, after: int) -> set[str]:
    """"half ten", "ten thirty", "ten and a half", "eleven sharp", "quarter to eleven", "eleven hundred hours"."""
    say = _en(h12)
    forms = {f"{hour:02d}{minute:02d} hours", f"{hour:02d}{minute:02d}h", f"{hour:02d}{minute:02d} hrs"}
    if minute == 0:
        forms |= {f"{say} sharp", f"{say} on the dot", f"{say} o'clock sharp", f"{say} oclock",
                  f"{_en(hour)} hundred hours", f"{_en(hour)} hundred hrs",
                  f"{say} am" if hour < 12 else f"{say} pm",
                  f"{say} in the morning" if hour < 12 else f"{say} in the evening" if hour >= 18 else
                  f"{say} in the afternoon"}
        forms |= {"noon", "midday", "twelve noon"} if hour == 12 else set()
        forms |= {"midnight", "twelve midnight"} if hour == 0 else set()
    elif minute == 30:
        forms |= {f"half past {say}", f"half {say}", f"{say} thirty", f"{say} and a half", f"half after {say}"}
    elif minute == 15:
        forms |= {f"quarter past {say}", f"a quarter past {say}", f"quarter after {say}", f"{say} fifteen"}
    elif minute == 45:
        forms |= {f"quarter to {_en(after)}", f"a quarter to {_en(after)}", f"quarter of {_en(after)}",
                  f"{say} forty-five"}
    else:
        forms |= {f"{say} {_en(minute)}", f"{_en(minute)} past {say}"}
        forms |= {f"{say} oh {_en(minute)}"} if minute < 10 else set()
        forms |= {f"{_en(60 - minute)} to {_en(after)}"} if minute > 30 else set()
    return forms


def _french_times(hour: int, minute: int, h12: int, after: int) -> set[str]:
    """"dix heures et demie", "une heure", "vingt et une heures", "onze heures moins le quart", "midi"."""
    def hours(n: int) -> str:
        words = re.sub(r"(^|[ -])un$", r"\1une", _words(n, "fr") or str(n))  # une heure, vingt et une heures
        return f"{words} heure" if n == 1 else f"{words} heures"

    said = {hours(hour), hours(h12)}
    if minute == 0:
        forms = said | {f"{h} pile" for h in said}
        forms |= {"midi", "midi pile"} if hour == 12 else {"minuit"} if hour == 0 else set()
    elif minute == 30:
        forms = {f"{h} et demie" for h in said} | {f"{h} trente" for h in said}
        forms |= {"midi et demi"} if hour == 12 else {"minuit et demi"} if hour == 0 else set()
    elif minute == 15:
        forms = {f"{h} et quart" for h in said} | {f"{h} quinze" for h in said}
    elif minute == 45:
        forms = {f"{hours(n)} moins le quart" for n in (hour + 1, after) if n <= 23}
        forms |= {f"{h} quarante-cinq" for h in said}
    else:
        forms = {f"{h} {_words(minute, 'fr')}" for h in said}
    return forms


def _german_times(hour: int, minute: int, h12: int, after: int) -> set[str]:
    """"zehn Uhr dreißig", "halb elf" (10:30), "viertel vor elf", "punkt elf", "ein Uhr"."""
    def say(n: int) -> str:
        return "ein" if n == 1 else _words(n, "de") or str(n)

    clocks = {f"{say(hour)} uhr", f"{say(h12)} uhr"}
    if minute == 0:
        return clocks | {f"punkt {say(h12)}", f"punkt {say(hour)} uhr"} | {f"{c} genau" for c in clocks}
    if minute == 30:
        return {f"halb {say(after)}"} | {f"{c} dreißig" for c in clocks} | {f"{c} dreissig" for c in clocks}
    if minute == 15:
        return {f"viertel nach {say(h12)}", f"viertel {say(after)}"} | {f"{c} fünfzehn" for c in clocks}
    if minute == 45:
        return {f"viertel vor {say(after)}", f"dreiviertel {say(after)}"} | {f"{c} fünfundvierzig" for c in clocks}
    return {f"{c} {_words(minute, 'de')}" for c in clocks}


def _spanish_times(hour: int, minute: int, h12: int, after: int) -> set[str]:
    """"a las diez y media", "a la una", "las once en punto", "las once menos cuarto"."""
    def say(n: int) -> str:
        return "una" if n == 1 else _words(n, "es") or str(n)

    def the(n: int) -> str:
        return f"la {say(n)}" if n == 1 else f"las {say(n)}"

    hours = {hour, h12}
    if minute == 0:
        return ({f"a {the(n)}" for n in hours} | {f"a {the(n)} en punto" for n in hours}
                | {f"{the(n)} en punto" for n in hours} | {f"{say(n)} en punto" for n in hours})
    if minute == 30:
        return {f"{the(n)} y media" for n in hours} | {f"{say(n)} y media" for n in hours} | {f"{the(n)} y treinta" for n in hours}
    if minute == 15:
        return {f"{the(n)} y cuarto" for n in hours} | {f"{say(n)} y cuarto" for n in hours} | {f"{the(n)} y quince" for n in hours}
    if minute == 45:
        return {f"{the(after)} menos cuarto", f"{say(after)} menos cuarto"}
    return {f"{the(n)} y {_words(minute, 'es')}" for n in hours}


def _italian_times(hour: int, minute: int, h12: int, after: int) -> set[str]:
    """"le dieci e mezza", "all'una", "alle undici", "le undici in punto", "le undici meno un quarto"."""
    def say(n: int) -> str:
        return "una" if n == 1 else _words(n, "it") or str(n)

    def the(n: int) -> str:
        return "l'una" if n == 1 else f"le {say(n)}"

    def at(n: int) -> str:
        return "all'una" if n == 1 else f"alle {say(n)}"

    hours = {hour, h12}
    if minute == 0:
        return {f"{the(n)} in punto" for n in hours} | {f"{say(n)} in punto" for n in hours} | {at(n) for n in hours}
    if minute == 30:
        return ({f"{the(n)} e mezza" for n in hours} | {f"{say(n)} e mezza" for n in hours}
                | {f"{say(n)} e mezzo" for n in hours} | {f"{the(n)} e trenta" for n in hours})
    if minute == 15:
        return {f"{the(n)} e un quarto" for n in hours} | {f"{say(n)} e un quarto" for n in hours} | {f"{the(n)} e quindici" for n in hours}
    if minute == 45:
        return {f"{the(after)} meno un quarto", f"{say(after)} meno un quarto"}
    return {f"{the(n)} e {_words(minute, 'it')}" for n in hours}


def _portuguese_times(hour: int, minute: int, h12: int, after: int) -> set[str]:
    """"dez e meia", "às onze horas", "uma hora", "duas horas", "onze em ponto"."""
    def say(n: int) -> str:
        return {1: "uma", 2: "duas"}.get(n) or _words(n, "pt") or str(n)

    def hours_of(n: int) -> str:
        return f"{say(n)} hora" if n == 1 else f"{say(n)} horas"

    hours = {hour, h12}
    if minute == 0:
        return {f"{say(n)} em ponto" for n in hours} | {f"às {hours_of(n)}" for n in hours} | {f"{hours_of(n)} em ponto" for n in hours}
    if minute == 30:
        return {f"{say(n)} e meia" for n in hours} | {f"{hours_of(n)} e meia" for n in hours} | {f"{hours_of(n)} e trinta" for n in hours}
    if minute == 15:
        return {f"{say(n)} e quinze" for n in hours} | {f"{say(n)} e um quarto" for n in hours}
    if minute == 45:
        return {f"um quarto para as {say(after)}"} | {f"{hours_of(n)} e quarenta e cinco" for n in hours}
    return {f"{hours_of(n)} e {_words(minute, 'pt')}" for n in hours}


def _dutch_times(hour: int, minute: int, h12: int, after: int) -> set[str]:
    """"half elf" (10:30), "kwart over tien", "elf uur" ("een uur" is left out: it also means "an hour")."""
    def say(n: int) -> str:
        return _words(n, "nl") or str(n)

    hours = {n for n in (hour, h12) if n != 1}
    if minute == 0:
        return {f"{say(n)} uur" for n in hours}
    if minute == 30:
        return {f"half {say(after)}"} | {f"{say(n)} uur dertig" for n in hours}
    if minute == 15:
        return {f"kwart over {say(h12)}"}
    if minute == 45:
        return {f"kwart voor {say(after)}"}
    return {f"{say(n)} uur {_words(minute, 'nl')}" for n in hours}


def _specific_time(form: str) -> bool:
    """False for a time form without digits whose only word is short (none are generated today; kept as
    a guard for future idioms)."""
    words = _letters(form)
    return bool(re.search(r"\d", form)) or len(words) >= 2 or any(len(word) >= 4 for word in words)


# --- amounts -------------------------------------------------------------------------------------------

_TURKISH_PARTS = sorted(
    "bir iki üç dört beş altı yedi sekiz dokuz on yirmi otuz kırk elli altmış yetmiş seksen doksan yüz bin "
    "milyon".split(), key=len, reverse=True,
)


def _turkish_spaced(words: str) -> str:
    """num2words writes Turkish numbers as one word ("binyüzseksen"); people write "bin yüz seksen"."""
    parts, rest = [], words
    while rest:
        part = next((p for p in _TURKISH_PARTS if rest.startswith(p)), None)
        if part is None:
            return words
        parts.append(part)
        rest = rest[len(part):]
    return " ".join(parts)


def _number_forms(number: int) -> set[str]:
    """An amount in words in every num2words language, and the English ways of saying it ("eleven-eighty",
    "eleven hundred and eighty", "six grand", "6 grand", "six thou")."""
    if number > MAX_SPOKEN:
        return set()
    forms = set()
    for language in NUMBER_LANGUAGES:
        words = _words(number, language)
        if words:
            forms.add(words)
            if language == "tr":
                forms.add(_turkish_spaced(words))
    english = _en(number)
    forms |= {english.replace(" and ", " "), english.replace(",", "")}
    if english.startswith("one thousand"):
        forms.add("a" + english[3:])
    high, low = divmod(number, 100)
    if 1100 <= number <= 9999 and low and high % 10:
        forms |= {f"{_en(high)} {_en(low)}", f"{_en(high)} hundred and {_en(low)}", f"{_en(high)} hundred {_en(low)}"}
        forms |= {f"{_en(high)} oh {_en(low)}"} if low < 10 else set()
    if 1100 <= number <= 9900 and not low and high % 10:
        forms.add(f"{_en(high)} hundred")
    if number >= 1000 and number % 1000 == 0:
        thousands = number // 1000
        forms |= {f"{_en(thousands)} grand", f"{thousands} grand", f"{thousands} thou", f"{_en(thousands)} thou"}
    return {form for form in forms if _specific_number(canon(form)[0])}


def _specific_number(form: str) -> bool:
    """False for a one-word amount shorter than six letters ("mil", "bin", "min", "cent", "mille"): those are
    everyday words in some language, or too common inside other numbers."""
    words = _letters(form)
    return _unbounded(form) or bool(re.search(r"\d", form)) or len(words) >= 2 or any(len(w) >= 6 for w in words)


@lru_cache(maxsize=1)
def number_words() -> frozenset[str]:
    """Words that are part of numbers in some num2words language, for telling "trois novembre" from
    "vingt trois novembre" (articles that also mean one are left out)."""
    tokens: set[str] = set()
    for language in NUMBER_LANGUAGES:
        for number in [*range(0, 101), 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1_000_000]:
            words = _words(number, language) if number <= MAX_SPOKEN else None
            if words:
                tokens.update(word for word in _letters(canon(words)[0]) if len(word) >= 3)
    tokens |= {"hundred", "thousand", "million", "grand", "thou"}
    return frozenset(tokens - _ARTICLES - {"and", "und"})


# --- matching ------------------------------------------------------------------------------------------

def _keep(form: str) -> str | None:
    """The matchable form of ``form``: canonical, on one line, with at least one letter."""
    folded = canon(form)[0].replace(BOUNDARY, " ").strip(" -")
    if not any(unicodedata.category(char).startswith("L") for char in folded):
        return None
    return folded if len(folded) >= (2 if _unbounded(folded) else 3) else None


def _unbounded(form: str) -> bool:
    """True for forms in scripts written without spaces between words, which match without word boundaries."""
    return any(unicodedata.name(char, "").startswith(_NO_SPACE_SCRIPTS) for char in form)


@lru_cache(maxsize=4096)
def surface_forms(key: str, years: tuple[int, ...] = ()) -> frozenset[str]:
    """Every generated, canonical written form of a scanner key ("date:11-03", "time:10:30",
    "number:1180"); ``years`` adds year- and weekday-bearing date forms."""
    if not AVAILABLE:
        _warn_once()
        return frozenset()
    kind, _, value = key.partition(":")
    if kind == "date":
        month, day = (int(part) for part in value.split("-"))
        raw = _date_forms(month, day, years)
    elif kind == "time":
        hour, minute = (int(part) for part in value.split(":"))
        raw = {form for form in _time_forms(hour, minute) if _specific_time(canon(form)[0])}
    elif kind == "number":
        whole = value.split(".")[0]
        raw = _number_forms(int(whole)) if whole.isdigit() and 100 <= int(whole) <= MAX_SPOKEN else set()
    else:
        return frozenset()
    return frozenset(kept for form in raw if (kept := _keep(form)))


def _spaced_pattern(form: str) -> str:
    """A whole-word pattern for a spaced form, where a space and a dash are interchangeable."""
    return "[ -]".join(re.escape(part) for part in re.split(r"[ -]+", form))


def _numeral_edges(form: str) -> tuple[bool, bool]:
    """Whether an unspaced form starts and ends with a digit or a CJK numeral."""
    numeral = re.compile(f"[{_CJK_NUMERALS}]")
    return bool(numeral.match(form[0])), bool(numeral.match(form[-1]))


@lru_cache(maxsize=4096)
def _patterns(key: str, years: tuple[int, ...]) -> tuple[re.Pattern[str], ...]:
    """Regexes for the forms of ``key``: whole words for spaced forms (a dash on either side is not a
    boundary), and for unspaced ones no digit or numeral just outside a numeral edge."""
    forms = surface_forms(key, years)
    spaced = sorted((f for f in forms if not _unbounded(f)), key=len, reverse=True)
    patterns = []
    if spaced:
        patterns.append(re.compile(r"(?<![^\W_])(?<!-)(?:" + "|".join(map(_spaced_pattern, spaced))
                                   + r")(?![^\W_])(?!-)"))
    groups: dict[tuple[bool, bool], list[str]] = {}
    for form in sorted((f for f in forms if _unbounded(f)), key=len, reverse=True):
        groups.setdefault(_numeral_edges(form), []).append(form)
    for (starts, ends), members in groups.items():
        before = f"(?<![{_CJK_NUMERALS}])" if starts else ""
        after = f"(?![{_CJK_NUMERALS}])" if ends else ""
        patterns.append(re.compile(before + "(?:" + "|".join(map(re.escape, members)) + ")" + after))
    return tuple(patterns)


class FormText:
    """An outbound text prepared once for form matching against many keys."""

    def __init__(self, text: str) -> None:
        self._text = text
        self.canon, self._origin = canon(text)

    def spans(self, key: str, years: tuple[int, ...] = ()) -> list[tuple[int, int]]:
        """Original-text spans where ``key`` appears in one of its generated forms, except where the form
        is part of a larger number or of a longer time."""
        kind = key.partition(":")[0]
        found = []
        for pattern in _patterns(key, years):
            for match in pattern.finditer(self.canon):
                start, end = match.span()
                if end <= start or self._inside_larger(start, end, kind):
                    continue
                found.append((self._origin[start], self._origin[end - 1] + 1))
        return found

    def _inside_larger(self, start: int, end: int, kind: str) -> bool:
        """True when the word just before or after [start, end) is a number word, or, for a time, when the
        word after continues it ("et demie", "y media", "and a half")."""
        before, after = self._neighbour(start - 1, -1), self._neighbour(end, 1)
        numbers = number_words()
        if before in numbers or after in numbers:
            return True
        return kind == "time" and after in _MINUTE_CONTINUATIONS

    def _neighbour(self, index: int, step: int) -> str:
        """The word next to canonical position ``index`` in direction ``step``, across spaces and dashes
        but not across a comma or a sentence end."""
        text = self.canon
        while 0 <= index < len(text) and text[index] in " -":
            if self._text[self._origin[index]] == ",":
                return ""
            index += step
        if not 0 <= index < len(text) or text[index] == BOUNDARY:
            return ""
        edge = index
        while 0 <= edge < len(text) and text[edge].isalnum():
            edge += step
        return text[index:edge] if step > 0 else text[edge + 1: index + 1]


# --- every date and time, whatever the memories hold ------------------------------------------------------

GRID_MINUTES = 5  # times of day are looked for on this grid: "half ten", "Viertel nach drei", "eleven sharp"


@lru_cache(maxsize=1)
def calendar_keys() -> tuple[str, ...]:
    """Every day of the year ("date:11-03") and every time of day on a :data:`GRID_MINUTES` grid
    ("time:10:30"), as scanner keys."""
    dates = tuple(f"date:{month:02d}-{day:02d}" for month in range(1, 13)
                  for day in range(1, monthrange(2024, month)[1] + 1))
    times = tuple(f"time:{hour:02d}:{minute:02d}" for hour in range(24) for minute in range(0, 60, GRID_MINUTES))
    return dates + times


@lru_cache(maxsize=4096)
def _calendar_hits(text: str) -> tuple[tuple[int, int, str], ...]:
    """(start, end, key) of every generated form of any date or time in ``text``, longest first at a start."""
    if not AVAILABLE or not any(char.isalpha() for char in text):
        return ()
    form_text = FormText(text)
    return tuple(sorted(((start, end, key) for key in calendar_keys() for start, end in form_text.spans(key)),
                        key=lambda hit: (hit[0], -hit[1], hit[2])))


def calendar_form_spans(text: str) -> list[tuple[int, int, str]]:
    """Where ``text`` writes a date or a time of day in one of its generated forms ("3 de novembro", "11月3日",
    "03NOV", "half ten", "halb elf"), as (start, end, kind) with kind "date" or "time", in order and without
    overlaps. Every date and every time on the grid is looked for, not only the ones private memories hold:
    masking a typed form only when it equals a private value would tell the receiver that it does."""
    kept: list[tuple[int, int, str]] = []
    cursor = 0
    for start, end, key in _calendar_hits(text):
        if start >= cursor:
            kept.append((start, end, key.partition(":")[0]))
            cursor = end
    return kept


def calendar_form_keys(text: str) -> frozenset[str]:
    """The keys of every date and time ``text`` writes in a generated form ("half ten" gives time:10:30 and
    time:22:30): what a shared memory says of its own time, so that it is not taken for a private one's."""
    return frozenset(key for _, _, key in _calendar_hits(text))


def warm_up() -> None:
    """Generate and compile the forms of every date and time once (about ten seconds), so that the first message
    does not wait for it."""
    calendar_form_spans("warm up: 3 de novembro, half ten")


def years_of(text: str, today: date | None = None) -> tuple[int, ...]:
    """The years a date in ``text`` may belong to: those written in it, this year and next year."""
    current = (today or date.today()).year
    written = {int(year) for year in re.findall(r"(?<!\d)(20\d\d)(?!\d)", text)}
    return tuple(sorted(written | {current, current + 1}))
