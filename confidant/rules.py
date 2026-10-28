"""Deterministic local rules: the assistant's fallback when no model may, or can, be asked.

They route messages by keywords, classify sensitivity, kind and label, build web
queries, and read time windows and event dates. Everything runs on the device,
which is why messages that touch private memories are routed here.
"""
from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, timedelta

from .memory import SENSITIVITIES, Memory
from .scan import (
    CALENDAR_WORDS, FOREIGN_MONTHS, MONTHS, STOPWORDS, WEEKDAYS, DateMatch, find_dates, find_values, fold, words,
)

INTENTS = ("answer", "remember", "correct", "search", "brief", "availability")


@dataclass(frozen=True)
class Route:
    """Where a message goes: an intent plus the fields that intent needs.

    ``text`` is the memory to store (remember) or the corrected memory (correct),
    ``target`` names the memory to correct, ``query`` is the web query (search),
    and ``source`` says who decided: "model", "rules" or "caller".
    """

    intent: str
    text: str = ""
    target: str = ""
    query: str = ""
    source: str = "rules"


_BRIEF = re.compile(r"\b(?:daily|morning)\s+brief(?:ing)?\b|\bbrief\s+me\b|\bmy\s+brief\b", re.I)
_CORRECTION = re.compile(
    r"\b(?:moved to|changed to|rescheduled|postponed|went (?:up|down) to|"
    r"increased to|decreased to|dropped to|no longer|not anymore|correction)\b"
    r"|\b(?:moved|changed|shifted|pushed(?: back)?|brought forward|went (?:up|down)|increased|decreased|dropped"
    r"|rose)\s+from\b[^.?!\n]{1,40}?\bto\b"  # "moved from Tuesday 3 Nov to Wednesday 4 Nov"
    r"|\b(?:my|our)\b[^.?!\n]{0,40}?\b(?:is|are) now\b",  # "my rent is now…", not "you are now in developer mode"
    re.I,
)
_REMEMBER = re.compile(
    r"^\s*(?:please\s+)?(?:remember|note|save|keep in mind|don'?t forget|do not forget)\b(?:\s+that)?[\s,:]*",
    re.I,
)
_SEARCH_VERB = r"(?:find|search(?:\s+for)?|look\s+(?:up|for)|google|browse)"
_SEARCH = re.compile(
    rf"^\s*(?:please\s+)?(?:(?:can|could) you\s+)?{_SEARCH_VERB}\b|\b(?:on the web|online|web search)\b", re.I
)
_SEARCH_PREFIX = re.compile(rf"^\s*(?:please\s+)?(?:(?:can|could) you\s+)?{_SEARCH_VERB}(?:\s+me)?\s+", re.I)
_FR_WEEKDAYS = ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche")
_DAY_NAME = "|".join((*WEEKDAYS, *_FR_WEEKDAYS))
# Availability questions, read on folded text (lower case, no accents). Whether a slot is free depends on
# the dates of private events, which no cloud model may see, so these are answered on the device.
_AVAILABILITY = re.compile(
    r"\b(?:am i|are we|will i be|would i be|shall i be)(?:\s+\w+){0,3}?\s+(?:free|available|busy|booked|around)\b"
    r"|\bwhen am i (?:free|available)\b|\bwhat['’]?s free\b|\bwhat is free\b"
    rf"|\bis (?:the |my )?(?:(?:this|next) )?(?:{_DAY_NAME}|today|tomorrow|tonight|morning|afternoon|evening|calendar|"
    r"schedule|diary|day|slot)\b[^.?!]{0,30}?\b(?:free|clear)\b"
    r"|^\s*(?:is|are)\b[^.?!]{0,40}?\bslots?\b[^.?!]{0,30}?\b(?:free|open|available|taken)\b"
    r"|\b(?:suis-je|je suis|est-ce que je suis|serai-je)(?:\s+\w+){0,2}?\s+"
    r"(?:libre|disponible|dispo|pris|prise|occupe|occupee)\b"
)
# "Do I have anything on Tuesday?" is about availability only when it names a time.
_HAVE_ANYTHING = re.compile(
    r"\b(?:do i have|have i got|is there)\s+"
    r"(?:anything|something|any plans|plans|an? (?:appointment|meeting|call)|time)\b"
    r"|\b(?:got anything|anything (?:on|planned|scheduled)|any (?:gaps?|openings?|free (?:time|slots?)))\b"
    r"|\bcan i (?:fit|squeeze|book|schedule|plan)\b"
    r"|\b(?:ai-je|est-ce que j['’]ai|j['’]ai)\s+(?:quelque chose|qqch|un rendez-vous|un rdv|des rendez-vous|le temps)\b"
)
# "free" about a price, not a person: "free entry", "free parking", "free for kids".
_FREE_OF_CHARGE = re.compile(
    r"\bfree\s+(?:entry|entrance|admission|parking|shipping|delivery|wi-?fi|trial|tickets?|of charge|"
    r"for (?:kids|children|"
    r"students|members|everyone)|to (?:visit|enter|use|download|attend))\b|\bfor free\b|\bgratuit\w*\b"
)
# An explicit request to look something up wins over an availability reading.
_WEB_MARKERS = re.compile(
    r"\b(?:on the web|online|web search|search the web|look (?:it|this|that) up|google it|sur (?:le )?web|en ligne)\b",
    re.I,
)
_PARTS_OF_DAY = (  # (pattern, name, first minute, end minute): an item belongs to the part its start time falls in
    (re.compile(r"\b(?:morning|mornings|matin|matinee)\b"), "morning", 6 * 60, 12 * 60),
    (re.compile(r"\b(?:afternoon|afternoons|apres-midi|apres midi|aprem)\b"), "afternoon", 12 * 60, 18 * 60),
    (re.compile(r"\b(?:lunch|lunchtime|noon|midday|midi)\b"), "lunchtime", 11 * 60 + 30, 14 * 60),
    (re.compile(r"\b(?:evening|evenings|tonight|night|soir|soiree)\b"), "evening", 18 * 60, 24 * 60),
)
AROUND_MINUTES = 60  # "free at 11:00?": items starting within an hour either side count, since durations are unknown
_POSSESSIVE = re.compile(r"\b(?:for me|my|our|me)\b\s*", re.I)

_HEALTH_TERMS = (
    r"doctor|gp|nurse|dermatolog\w*|dentist|physio\w*|therap\w*|psycholog\w*|psychiatr\w*|diagnos\w*|"
    r"medication|medicine|prescription|pills?|symptoms?|clinic|hospital|blood test|surgery|pregnan\w*|"
    r"colonoscop\w*|endoscop\w*|biops\w*|mammogra\w*|ultrasound|mri|x-?rays?|scan results?|lab results?|"
    r"test results?|chemo\w*|cancer|tumou?rs?|diabet\w*|asthma|epilep\w*|hiv|aids|sti|std|hepatitis|"
    r"antidepressants?|ssris?|anxiety|depress\w*|adhd|bipolar|rehab|addict\w*|sobriety|ivf|fertility|"
    r"miscarr\w*|abortion|contracepti\w*|gyn(?:a)?ecolog\w*|urolog\w*|oncolog\w*|cardiolog\w*|"
    r"neurolog\w*|illness|disease|sick leave|medical|counsell?ors?|counsell?ing|grief|bereave\w*|"
    r"alzheimer\w*|dementia|parkinson\w*|strokes?|seizures?|heart attack|heart condition|autis\w*|asperger\w*|"
    r"disabilit\w*|disabled|wheelchair|care home|nursing home|hospice|palliative|terminal(?:ly)? ill|"
    r"alcoholi\w*|drinking problem|(?:aa|na) meetings?|alcoholics anonymous|relaps\w*|overdos\w*|"
    r"lupus|crohn\w*|arthritis|multiple sclerosis|fibromyalgia|endometriosis|migraines?|blood pressure|"
    r"eating disorder|anorexi\w*|bulimi\w*|self-harm|suicid\w*|"
    r"medecin|ordonnance|maladie|enceinte|grossesse|hopital|clinique|prise de sang|arret maladie|mutuelle"
)
_MONEY_TERMS = (
    r"rent|salary|salaries|wage|bank|loan|debt|mortgage|savings|tax|taxes|income|invoice|payslip|"
    r"credit|inheritance|heritage|housing costs?|monthly payments?|(?:utility|phone|electricity|energy|water|gas) bills?|overdraft|paycheck|pension|"
    r"insurance|bankrupt\w*|insolven\w*|repossess\w*|evict\w*|foreclos\w*|arrears|bailiffs?|gambl\w*|"
    r"loyer|salaire|impots?|banque|faillite|"
    r"pret|dette|epargne"
)
_JOB_TERMS = (
    r"interview\w*|recruiter|job offer|resign\w*|notice period|severance|redundan\w*|laid off|layoffs?|fired|"
    r"unemploy\w*|burn-?out|salary negotiation|hiring managers?|headhunters?|job (?:search|hunt\w*|applications?)|"
    r"applying for|applied for|(?:a |another )?new job|leaving the company|counter-?offers?|"
    r"entretien d'embauche|recruteur|demission|licenciement|chomage"
)
_LEGAL_TERMS = (
    r"confidential|lawyer|court|lawsuit|divorce|police|passport|visa|residence permit|id card|social security|"
    r"arrest\w*|prison|jail\w*|convict\w*|criminal|probation|parole|custody|restraining order|fraud|sued|suing|"
    r"deport\w*|asylum|immigration|"
    r"avocat|tribunal|plainte|titre de sejour|carte d'identite|garde a vue"
)
_INTIMATE_TERMS = (
    r"sexual\w*|sexuality|religio\w*|political party|union member\w*|gay|lesbian|bisexual|queer|trans|"
    r"transgender|non-binary|coming out|came out|closeted|cheat(?:s|ed|ing)?|affair|unfaithful|"
    r"separat(?:ed|ing|ion)|broke up|breaking up|my ex|ex-(?:wife|husband|partner)|"
    r"trying for a baby|church|mosque|synagogue|mass|prayers?|ramadan|"
    r"must not know|mustn'?t know|must never know|should not know|shouldn'?t know|don'?t tell|do not tell|"
    r"not to be mentioned|keep (?:it|this|that) (?:quiet|secret|between us)|between you and me|"
    r"(?:nobody|no one) knows|ha(?:s|ve)n'?t told|ha(?:s|ve) not told|secret(?:ly)?|behind (?:his|her|their|my) back"
)
_PRIVATE_TERMS = re.compile(
    rf"\b(?:{_HEALTH_TERMS}|{_MONEY_TERMS}|{_JOB_TERMS}|{_LEGAL_TERMS}|{_INTIMATE_TERMS})\b"
)
_MONEY = re.compile(
    r"[€$£]\s?\d|\b\d[\d,. ]*\s?(?:[€$£]|(?:eur|euros?|usd|dollars?|gbp|pounds?)\b)"
)
# Category labels for private memories: a label travels with every placeholder, so it
# names the kind of thing only ("health appointment"), never a word of the memory itself.
_PRIVATE_CATEGORIES = (
    (re.compile(rf"\b(?:{_HEALTH_TERMS})\b"), "health"),
    (re.compile(rf"\b(?:{_MONEY_TERMS})\b"), "finances"),
    (re.compile(rf"\b(?:{_JOB_TERMS})\b"), "career"),
    (re.compile(rf"\b(?:{_LEGAL_TERMS})\b"), "legal or official matter"),
    (re.compile(rf"\b(?:{_INTIMATE_TERMS})\b"), "personal matter"),
)
_PERSONAL_TERMS = re.compile(
    r"\b(?:sister|brother|mother|mom|mum|father|dad|parents?|partner|wife|husband|boyfriend|girlfriend|"
    r"friends?|family|birthday|anniversary|kids?|son|daughter|prefers?|favou?rite|loves?|hates?|likes?|"
    r"dinner|holiday|vacation|trip|race|marathon|run|running|gym|home|weekend)\b"
)
# What makes a new fact clearly everyday: a taste, or a plan or pastime. A family word alone does not, since
# facts about family are often the sensitive ones ("My brother lost his job").
# What makes a new fact clearly everyday: the taste and plan words the rules used before 28 Oct, without the family
# words (a family word alone is not everyday: "my brother was arrested"). Wider lists made sensitive facts
# shareable again ("Brunch with my sponsor after Sunday's meeting", "I enjoy a bottle of wine every night").
_EVERYDAY_TERMS = re.compile(
    r"\b(?:prefers?|favou?rite|loves?|hates?|likes?|rather|usually|always|never|birthday|anniversary|friends?|"
    r"dinner|holiday|vacation|trip|race|marathon|run|running|gym|home|weekend)\b"
)
_PUBLIC_TERMS = re.compile(
    r"\b(?:team|tools?|workflow|process|guidelines?|checklist|documentation|docs|standup|sprint|"
    r"design system|public)\b"
)
_SKILL_TERMS = re.compile(r"\b(?:how to|checklist|steps|template|playbook|procedure|recipe)\b")
_PREFERENCE_TERMS = re.compile(r"\b(?:prefers?|favou?rite|loves?|hates?|likes?|rather|usually|always|never)\b")
_EVENT_TERMS = re.compile(r"\b(?:appointment|meeting|interview|dinner|party|flight|deadline|call|race)\b")
_AGENDA_TERMS = re.compile(r"\b(?:plate|agenda|schedule|calendar|upcoming|coming up|plans|busy)\b")
_OPEN_ITEM = re.compile(r"^(?:to ?do|todo)\b|\b(?:need to|remember to|follow up|pending)\b")


def rule_route(message: str) -> Route:
    """Route a message by keywords: brief, correct, remember, availability, search, else answer."""
    if _BRIEF.search(message):
        return Route("brief")
    if _CORRECTION.search(message):
        return Route("correct", text=memory_text_from(message))
    if _REMEMBER.search(message):
        return Route("remember", text=memory_text_from(message))
    if _WEB_MARKERS.search(message):
        return Route("search", query=web_query_from(message))
    if is_availability_question(message):
        return Route("availability")
    if _SEARCH.search(message):
        return Route("search", query=web_query_from(message))
    return Route("answer")


def is_availability_question(message: str) -> bool:
    """True for "Am I free on Tuesday morning?", "Is Friday evening free?", "Do I have anything tomorrow?",
    "Suis-je libre mardi matin ?": questions whose answer depends on when events happen."""
    folded = fold(message)[0]
    if _FREE_OF_CHARGE.search(folded):
        return False
    if _AVAILABILITY.search(folded):
        return True
    return bool(_HAVE_ANYTHING.search(folded)) and (
        period_asked(message, _ANY_DAY) is not None or part_of_day(message) is not None
    )


_ANY_DAY = date(2026, 1, 5)  # whether a message names a period does not depend on the day it is read


def memory_text_from(message: str) -> str:
    """The memory a message states, as a sentence: "Remember that my X is Y" → "X is Y."."""
    text = _REMEMBER.sub("", message, count=1).strip()
    text = re.sub(r"^(?:my|our)\s+", "", text, flags=re.I)
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    return text if text[-1] in ".!?" else text + "."


def web_query_from(message: str) -> str:
    """A web query from a request: drops the search verb, possessives and a leading article."""
    query = _POSSESSIVE.sub("", _SEARCH_PREFIX.sub("", message, count=1))
    query = re.sub(r"^(?:an?|the)\s+", "", query.strip(), flags=re.I)
    return " ".join(query.rstrip(" .?!").split())


def classify_sensitivity(text: str) -> str:
    """Sensitivity by keywords: private for health, money, legal, job-search or intimate topics and
    for secrecy cues ("must not know"), public for work tooling without personal context, personal
    otherwise."""
    folded = fold(text)[0]
    if _PRIVATE_TERMS.search(folded) or _MONEY.search(folded):
        return "private"
    if _PUBLIC_TERMS.search(folded) and not _PERSONAL_TERMS.search(folded):
        return "public"
    return "personal"


def classify_new_fact(text: str) -> str:
    """Sensitivity of a fact the user asks Confidant to keep. Keyword lists cannot name every
    illness, debt or dispute, so a fact is private unless the rules see something clearly
    everyday (a taste, a plan or pastime) or work-public; any sensitive or secrecy cue wins
    over the everyday words, and a family word alone is not everyday. When in doubt, private."""
    level = classify_sensitivity(text)
    if level != "personal":
        return level
    return "personal" if _EVERYDAY_TERMS.search(fold(text)[0]) else "private"


def classify_kind(text: str) -> str:
    """Kind by keywords: skill, preference, event (a date, a time or an event word) or fact."""
    folded = fold(text)[0]
    if _SKILL_TERMS.search(folded):
        return "skill"
    if _PREFERENCE_TERMS.search(folded):
        return "preference"
    has_time = any(value.kind == "time" for value in find_values(text))
    if find_dates(text) or has_time or _EVENT_TERMS.search(folded):
        return "event"
    return "fact"


def make_label(text: str, sensitivity: str | None = None) -> str:
    """A short lower-case label, safe to show next to a placeholder.

    A private memory gets a category label ("health appointment", "finances",
    "career"), since its label is sent wherever its value is not. Others get up to
    three topic words that never contain a value (no names, numbers or dates).
    """
    if (sensitivity or classify_sensitivity(text)) == "private":
        folded = fold(text)[0]
        category = next((name for pattern, name in _PRIVATE_CATEGORIES if pattern.search(folded)), "")
        if not category:
            category = "finances" if _MONEY.search(folded) else "private matter"
        dated = bool(find_dates(text) or any(value.kind == "time" for value in find_values(text)))
        if dated and category in ("health", "career"):
            return f"{category} appointment"
        return category
    value_words = {word for value in find_values(text) for word in words(text[value.start:value.end])}
    picked = [
        word for word in words(text)
        if len(word) > 2 and word not in STOPWORDS and word not in CALENDAR_WORDS
        and word not in value_words and not any(char.isdigit() for char in word)
    ]
    return " ".join(list(dict.fromkeys(picked))[:3]) or "note"


def stricter(first: str, second: str) -> str:
    """The more restrictive of two sensitivity levels."""
    return max(first, second, key=SENSITIVITIES.index)


def stem(word: str) -> str:
    """Crude singular form, enough to match "appointments" with "appointment"."""
    return word[:-1] if len(word) > 3 and word.endswith("s") else word


def label_mentioned(label: str, message: str) -> bool:
    """True when every topic word of ``label`` appears in ``message`` (accents, case
    and plurals ignored), e.g. "dermatologist appointment" in "My dermatologist
    appointment moved"."""
    label_words = [word for word in words(label) if len(word) > 2 and word not in STOPWORDS]
    message_stems = {stem(word) for word in words(message)}
    return bool(label_words) and all(stem(word) in message_stems for word in label_words)


def time_window(message: str, today: date) -> tuple[date, date] | None:
    """The period a question for the cloud model asks about, if any: an explicit date, today,
    tomorrow, this or next week, the weekend, "next N days", or two weeks for agenda words.

    Its start date goes to the model in the period note, so it reads only these periods, the ones
    the leak rounds cover; :func:`period_asked` also reads weekdays, for checks made on the device.
    """
    dates = find_dates(message)
    if dates:
        day = dates[0].resolve(today)
        return day, day
    folded = fold(message)[0]
    if re.search(r"\b(?:today|tonight)\b", folded):
        return today, today
    if re.search(r"\btomorrow\b", folded):
        return today + timedelta(days=1), today + timedelta(days=1)
    if re.search(r"\bnext week\b", folded):
        monday = today + timedelta(days=7 - today.weekday())
        return monday, monday + timedelta(days=6)
    if re.search(r"\bthis week\b", folded):
        return today, today + timedelta(days=6 - today.weekday())
    if re.search(r"\bweekend\b", folded):
        if today.weekday() == 6:
            return today, today
        saturday = today + timedelta(days=5 - today.weekday())
        return saturday, saturday + timedelta(days=1)
    days = re.search(r"\bnext (\d{1,2}) days\b", folded)
    if days:
        return today, today + timedelta(days=int(days.group(1)))
    if _AGENDA_TERMS.search(folded):
        return today, today + timedelta(days=14)
    return None


_NEXT_WEEK = re.compile(r"\bnext week\b|\bsemaine prochaine\b")
_THIS_WEEK = re.compile(r"\bthis week\b|\bcette semaine\b")
_WEEKDAY = re.compile(rf"\b(?:(?P<this>this|ce) |(?P<next>next) )?(?P<day>{_DAY_NAME})(?P<prochain> prochain)?\b")
_WEEKDAY_RANGE = re.compile(
    rf"\b(?:between|from|entre|du)?\s*(?P<first>{_DAY_NAME})\s*(?:and|to|through|until|till|-|–|et|au|a|jusqu['’]au)\s*"
    rf"(?:next |le )?(?P<last>{_DAY_NAME})\b"
)
_DAY_OF_MONTH = re.compile(
    rf"\b(?:(?:{_DAY_NAME}),?\s+)?(?:the\s+|le\s+)(?P<num>[12]?\d|3[01])(?:st|nd|rd|th|er)?\b(?!\s*(?:h\b|:|\d))"
    rf"|\b(?:{_DAY_NAME}),?\s+(?P<num2>[12]?\d|3[01])(?:st|nd|rd|th|er)?\b(?!\s*(?:h\b|:|\d|am\b|pm\b))"
    r"|\b(?P<num3>[12]?\d|3[01])(?:st|nd|rd|th)\b"
)
_TODAY = re.compile(
    r"\b(?:today|tonight|this (?:morning|afternoon|evening)|aujourd['’]hui|ce (?:matin|soir)|cet apres-midi)\b"
)
_UNREAD_PERIOD = re.compile(r"\b(?:months?|mois|years?|annee|an prochain|week after next)\b|\d")


def _weekday_after(today: date, index: int, *, next_week: bool, this: bool) -> date:
    """A weekday by its index: in next week ("next Tuesday"), from today on ("this Tuesday"), or the next one
    after today (a bare "Tuesday")."""
    if next_week:
        return today + timedelta(days=7 - today.weekday() + index)
    if this:
        return today + timedelta(days=(index - today.weekday()) % 7)
    return today + timedelta(days=(index - today.weekday() - 1) % 7 + 1)


def _next_day_of_month(today: date, number: int) -> date | None:
    """The first date from today on whose day of the month is ``number`` ("the 12th")."""
    year, month = today.year, today.month
    for _ in range(13):
        if number <= calendar.monthrange(year, month)[1]:
            candidate = date(year, month, number)
            if candidate >= today:
                return candidate
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return None


def _date_readings(message: str, today: date) -> list[date]:
    """Every reading of the dates written in a message. A numeric date ("3/11", "11/03/2026") is read day
    first and month first, since either may be meant."""
    found = find_dates(message)
    readings = [match.resolve(today) for match in found]
    years = {match.year for match in found if match.year}
    for value in find_values(message):
        if value.kind != "date" or not re.search(r"\d\s*[/-]\s*\d", message[value.start:value.end]):
            continue
        month, day = (int(part) for part in value.folded.split("-"))
        for year in years or {None}:
            try:
                readings.append(date(year, month, day) if year else DateMatch(0, 0, month, day, None).resolve(today))
            except ValueError:
                continue
    return readings


def period_asked(message: str, today: date) -> tuple[date, date] | None:
    """The period a message names, if any, for checks made on the device (nothing of it is sent): a date
    (a numeric one read both ways, the next one from today preferred), today, tomorrow, the day after
    tomorrow, a weekday ("Tuesday", "next Tuesday", "mardi"), a day of the month ("Thursday the 12th"),
    a range of weekdays ("between Tuesday and Thursday"), this or next week, the weekend, or
    "next N days", in English or French; two dates joined as a range ("from Monday 9 Nov to Wednesday 11
    Nov") give every day between them."""
    found = find_dates(message)
    gap = fold(message[found[0].end:found[1].start])[0].strip() if len(found) == 2 else ""
    if len(found) == 2 and (_joins_a_range(message, found[0], found[1]) or gap in ("and", "et")):
        first, last = found[0].resolve(today), found[1].resolve(today)
        if first < last <= first + timedelta(days=MAX_RANGE_DAYS):
            return first, last
    readings = _date_readings(message, today)
    if readings:
        ahead = sorted(day for day in readings if day >= today)
        day = ahead[0] if ahead else max(readings)
        return day, day
    folded = fold(message)[0]
    if re.search(r"\b(?:day after tomorrow|apres-demain|apres demain)\b", folded):
        return today + timedelta(days=2), today + timedelta(days=2)
    if _TODAY.search(folded):
        return today, today
    if re.search(r"\b(?:tomorrow|demain)\b", folded):
        return today + timedelta(days=1), today + timedelta(days=1)
    order = WEEKDAYS + _FR_WEEKDAYS
    next_week = bool(_NEXT_WEEK.search(folded))
    span = _WEEKDAY_RANGE.search(folded)
    if span:
        first = _weekday_after(today, order.index(span["first"]) % 7, next_week=next_week, this=False)
        last = first + timedelta(days=(order.index(span["last"]) % 7 - first.weekday()) % 7)
        return first, last
    numbered = _DAY_OF_MONTH.search(folded)
    if numbered:
        day = _next_day_of_month(today, int(numbered["num"] or numbered["num2"] or numbered["num3"]))
        if day is not None:
            return day, day
    weekday = _WEEKDAY.search(folded)
    if weekday:
        day = _weekday_after(
            today, order.index(weekday["day"]) % 7, next_week=bool(weekday["next"] or weekday["prochain"] or next_week),
            this=bool(weekday["this"] or _THIS_WEEK.search(folded)),
        )
        return day, day
    monday = today + timedelta(days=7 - today.weekday())  # of next week
    if next_week:
        return monday, monday + timedelta(days=6)
    if _THIS_WEEK.search(folded):
        return today, today + timedelta(days=6 - today.weekday())
    if re.search(r"\bweek-?end\b", folded):
        if today.weekday() == 6:
            return today, today
        saturday = today + timedelta(days=5 - today.weekday())
        return saturday, saturday + timedelta(days=1)
    days = re.search(r"\bnext (\d{1,2}) days\b", folded)
    if days:
        return today, today + timedelta(days=int(days.group(1)))
    return None


def unread_period(message: str) -> bool:
    """True when a message seems to name a day that :func:`period_asked` cannot read ("next month", "the
    week after next", a date in a form it does not know): better to ask than to check another day."""
    if period_asked(message, _ANY_DAY) is not None:
        return False
    folded = fold(message)[0]
    if any(word in _MONTH_NAMES for word in words(message)):
        return True
    clock = part_of_day(message)
    stripped = _CLOCK_IN_QUESTION.sub(" ", folded) if clock else folded
    return bool(_UNREAD_PERIOD.search(stripped))


_EN_HOUR_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12,
}
# Times in a question the value scanners do not take as times: "the 1030 slot", "at 1100", "at 11", "at eleven".
_CLOCK_IN_QUESTION = re.compile(
    r"\b(?:at|@)\s*(?P<mil>(?:[01]\d|2[0-3])[0-5]\d)\b|\b(?P<mil2>(?:[01]\d|2[0-3])[0-5]\d)\s+slot\b"
    r"|\bat\s+(?P<hour>[01]?\d|2[0-3])\b(?![:.h]\d|\s*(?:am|pm|h)\b)"
    rf"|\bat\s+(?P<word>{'|'.join(_EN_HOUR_WORDS)})\b"
)


def part_of_day(message: str) -> tuple[str, int, int] | None:
    """The part of a day a message asks about, as ``(name, first minute, end minute)``: a time written in it
    ("at 11:00", "the 1030 slot", "at eleven": an hour either side), else morning, lunchtime, afternoon or
    evening; None for a whole day. A bare hour from 1 to 6 ("at 3") is taken in the afternoon."""
    minute: int | None = None
    times = [value for value in find_values(message) if value.kind == "time"]
    if times:
        minute = clock_minutes(times[0].key)
    else:
        folded = fold(message)[0]
        clock = _CLOCK_IN_QUESTION.search(folded)
        if clock:
            military = clock["mil"] or clock["mil2"]
            if military:
                minute = int(military[:2]) * 60 + int(military[2:])
            else:
                hour = int(clock["hour"]) if clock["hour"] else _EN_HOUR_WORDS[clock["word"]]
                minute = (hour + 12 if 1 <= hour <= 6 else hour) * 60
    if minute is not None:
        shown = f"{minute // 60:02d}:{minute % 60:02d}"
        return f"around {shown}", max(0, minute - AROUND_MINUTES), min(24 * 60, minute + AROUND_MINUTES)
    folded = fold(message)[0]
    for pattern, name, first, end in _PARTS_OF_DAY:
        if pattern.search(folded):
            return name, first, end
    return None


_ORDINAL_DAY = re.compile(r"\b(?:[12]?\d|3[01])(?:st|nd|rd|th)\b|\b(?:le|du|au)\s+(?:[12]?\d|3[01]|1er)\b")
_RELATIVE_TIME = re.compile(
    r"\b(?:today|tonight|tomorrow|yesterday|weeks?|week-?ends?|months?|mornings?|afternoons?|evenings?|nights?|noon|"
    r"midday|lunch\w*|schedule|calendar|diary|agenda|aujourd['’]hui|demain|apres-demain|hier|semaines?|mois|matin\w*|"
    r"apres-midi|soir\w*|midi)\b"
)


def names_time(message: str) -> bool:
    """True when a message refers to a time: a date or clock time in any form the scanners read, a weekday or
    month, an ordinal day ("the 12th"), or a relative word ("tomorrow", "next week", "this morning").

    Such a message must not pick private memories by the words they share with it: "Tuesday" matching
    "Dermatologist appointment on Tuesday 3 Nov" would tell a cloud model which day that item is."""
    if find_dates(message) or any(value.kind in ("date", "time") for value in find_values(message)):
        return True
    if any(word in CALENDAR_WORDS for word in words(message)):
        return True
    folded = fold(message)[0]
    return bool(_ORDINAL_DAY.search(folded) or _RELATIVE_TIME.search(folded))


def clock_minutes(key: str) -> int:
    """Minutes since midnight of a time key such as "time:10:30"."""
    hours, minutes = key.split(":", 1)[1].split(":")
    return int(hours) * 60 + int(minutes)


# When an item starts: a clock time after "at" ("at 10:30", "at 2:30pm", "at 6.30pm", "à 10h30"), never a pace,
# a goal or a day of the month ("an easy 5:45 per km", "under 1:50", "from 3 Nov").
_MONTH_NAMES = {fold(name)[0] for name in (*MONTHS, *FOREIGN_MONTHS)}
_MONTH_AHEAD = r"\s*(?:" + "|".join(sorted(map(re.escape, _MONTH_NAMES), key=len, reverse=True)) + r")\b"
_AT_TIME = re.compile(
    r"\b(?P<prep>at|from|to|around|a|vers|des)\s+(?P<hour>[01]?\d|2[0-3])"
    r"(?:\s?(?P<sep>[:h.])\s?(?P<minute>[0-5]\d)?)?\s?(?:(?P<half>[ap])\.?m\b\.?)?"
    rf"(?!\d|{_MONTH_AHEAD}|-(?:day|week|month|km)|\s?(?:km|mi|%|percent|per)\b)"
)
_RESCHEDULED = re.compile(
    r"\b(?:moved|rescheduled|postponed|changed|pushed|brought forward|is now|now at|deplace\w*|reporte\w*)\b"
)


# When an item ends: a clock time right after its start, after "to", "until", "-", "and" ("between 9:00 and
# 12:30") or "jusqu'à"; a bare range ("9:00-12:30"); or a duration ("for 2 hours", "a 90-minute call").
_END_TIME = re.compile(
    r"\s*(?:[-–—]|\b(?:to|until|till|til|through|and|jusqu['’]?\s?a|a)\b)\s*"
    r"(?P<hour>[01]?\d|2[0-3])(?:\s?(?P<sep>[:h.])\s?(?P<minute>[0-5]\d)?)?\s?(?:(?P<half>[ap])\.?m\b\.?)?"
    rf"(?!\d|{_MONTH_AHEAD}|\s?(?:km|mi|%|percent|per)\b)"
)
_BARE_RANGE = re.compile(
    r"(?<![\d:.])(?P<h1>[01]?\d|2[0-3])[:h.](?P<m1>[0-5]\d)\s?[-–—]\s?(?P<h2>[01]?\d|2[0-3])(?:[:h.](?P<m2>[0-5]\d))?"
    r"(?!\d)"
)
_COUNT_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
_FOR_DURATION = re.compile(  # at most three digits: "for 999 minutes" is the longest duration read
    r"\bfor\s+(?:about\s+|around\s+)?(?P<n>\d{1,3}(?:[.,]\d{1,2})?|an?|one|two|three|four|five|six)\s+"
    r"(?P<unit>hours?|hrs?|minutes?|mins?)\b"
)
_ADJECTIVE_DURATION = re.compile(r"\b(?P<n>\d{1,3}(?:[.,]\d{1,2})?)-(?P<unit>hour|hr|minute|min)\b")
# Between two dates of a range ("from 9 Nov to 11 Nov", "du 9 au 11 novembre"). "and" joins a range only after
# "between": "physio on Wednesday 11 Nov and Friday 13 Nov" is two days, not three.
_RANGE_GAP = re.compile(r"\s*(?:[-–—]|to|until|till|through|thru|au|a|jusqu['’]?\s?au|bis|al|hasta)\s*")
_BETWEEN = re.compile(r"\b(?:between|entre)\s*$")
_DAY_BEFORE_DATE = re.compile(r"(?<![\d/.:])(?P<day>[1-9]|[12]\d|3[01])\s?(?:[-–—]|to|au)\s?$")
MAX_RANGE_DAYS = 31  # "from 9 Nov to 11 Nov" is a range; two dates further apart are two mentions
ONGOING_MINUTES = 180  # an item with no end noted may still be going on this long after it starts
DAY_MINUTES = 24 * 60


def _minutes(hour: str, minute: str | None, half: str | None) -> int:
    """Minutes since midnight of a clock time read from text, with "am"/"pm" applied."""
    value, extra = int(hour), int(minute or 0)
    if half == "p" and value < 12:
        value += 12
    elif half == "a" and value == 12:
        value = 0
    return value * 60 + extra


def _clock_times(folded: str) -> list[tuple[int, int]]:
    """Clock times written after "at", "from", "to"... as (minutes, end of match), in text order."""
    times: list[tuple[int, int]] = []
    for match in _AT_TIME.finditer(folded):
        french = match["prep"] in ("a", "vers", "des")
        if not (match["sep"] or match["half"]) and (french or match["prep"] in ("from", "to")):
            continue  # "a 10-day plan", "from 3 to 5": only "at 9" may leave out the minutes
        if match["sep"] == "." and not match["minute"]:
            continue
        times.append((_minutes(match["hour"], match["minute"], match["half"]), match.end()))
    return times


def _end_minute(folded: str, start: int, after: int) -> int | None:
    """When an item that starts at ``start`` (its time written up to ``after``) ends, if the text says."""
    match = _END_TIME.match(folded, after)
    if match and (match["sep"] != "." or match["minute"]):
        end = _minutes(match["hour"], match["minute"], match["half"])
        if end <= start and not match["half"] and end < 12 * 60 and end + 12 * 60 > start:
            end += 12 * 60  # "from 2:30pm to 4"
        if end > start:
            return end
    for pattern in (_FOR_DURATION, _ADJECTIVE_DURATION):
        found = pattern.search(folded)
        if found:
            count = _COUNT_WORDS.get(found["n"]) or float(found["n"].replace(",", "."))
            length = count * (60 if found["unit"].startswith("h") else 1)
            return min(DAY_MINUTES, start + max(1, round(length)))
    return None


def item_span(memory: Memory) -> tuple[int, int] | None:
    """When an item happens, in minutes since midnight: from a clock time written after "at" (the last one
    when the text says it was moved: "moved from 10:30 to 9:00") to its end time or duration when the text
    gives one ("from 9:00 to 12:30", "at 10:00 until 17:00", "9:00-12:30", "for 2 hours"); ``(start, start +
    1)`` when it gives none. Else the part of the day its text names ("every Sunday morning"), else None."""
    folded = fold(memory.text)[0]
    times = _clock_times(folded)
    if len(times) > 1 and _RESCHEDULED.search(folded[:times[-1][1]]):  # "moved from 10:30 to 9:00"
        start = times[-1][0]
        return start, start + 1
    if times:
        start, after = times[0]
        return start, _end_minute(folded, start, after) or start + 1
    bare = _BARE_RANGE.search(folded)
    if bare:
        start, end = _minutes(bare["h1"], bare["m1"], None), _minutes(bare["h2"], bare["m2"], None)
        return start, end if end > start else start + 1
    for pattern, _, first, end in _PARTS_OF_DAY:
        if pattern.search(folded):
            return first, end
    return None


def end_noted(span: tuple[int, int]) -> bool:
    """Whether an :func:`item_span` result carries an end (a lone start is ``(start, start + 1)``)."""
    return span[1] - span[0] > 1


def event_days(memory: Memory, today: date) -> tuple[date, date] | None:
    """The first and last day of a memory, resolved relative to ``today``: one day for a single date (the
    last one written when the text says it was moved: "rescheduled from 3 Nov to 4 Nov"), every day of a
    range ("from Monday 9 Nov to Wednesday 11 Nov", "9-11 Nov", "du 9 au 11 novembre"); None if undated."""
    found = find_dates(memory.text)
    if not found:
        return None
    # a reschedule word before the last date moves the item there ("moved from 3 Nov to 4 Nov"); one after the
    # dates is about something else ("from 9 Nov to 11 Nov; the venue is now Annecy")
    moved = bool(_RESCHEDULED.search(fold(memory.text[:found[-1].start])[0]))
    if len(found) > 1 and moved:
        day = found[-1].resolve(today)
        return day, day
    first = found[0].resolve(today)
    if len(found) > 1 and _joins_a_range(memory.text, found[0], found[1]):
        last = found[1].resolve(today)
        if first < last <= first + timedelta(days=MAX_RANGE_DAYS):
            return first, last
    before = None if moved else _DAY_BEFORE_DATE.search(fold(memory.text[:found[0].start])[0])
    if before and int(before["day"]) < first.day:
        return first.replace(day=int(before["day"])), first
    return first, first


def _joins_a_range(text: str, first: DateMatch, second: DateMatch) -> bool:
    """Whether the words between two dates of ``text`` make them the ends of a range."""
    gap = fold(text[first.end:second.start])[0]
    if _RANGE_GAP.fullmatch(gap):
        return True
    return gap.strip() == "and" and bool(_BETWEEN.search(fold(text[:first.start])[0]))


def event_date(memory: Memory, today: date) -> date | None:
    """The (first) day of a memory, as :func:`event_days` reads it."""
    days = event_days(memory, today)
    return days[0] if days else None


def item_span_on(memory: Memory, day: date, today: date) -> tuple[int, int] | None:
    """When an item happens on ``day``. An item over several days runs from its first time written (or
    midnight) on its first day, all day in between, and until its last time written (or midnight) on its
    last day; any other item has its :func:`item_span`."""
    days = event_days(memory, today) if memory.kind == "event" else None
    if days is None or days[0] == days[1] or not days[0] <= day <= days[1]:
        return item_span(memory)
    times = [minute for minute, _ in _clock_times(fold(memory.text)[0])]
    if day == days[0]:
        return (times[0] if times else 0), DAY_MINUTES
    if day == days[1]:
        return 0, (times[-1] if len(times) > 1 else DAY_MINUTES)
    return 0, DAY_MINUTES


def is_open_item(memory: Memory) -> bool:
    """True for to-dos: text starting with "To do" or mentioning something pending."""
    return bool(_OPEN_ITEM.search(fold(memory.text)[0]))


def recurs_on(memory: Memory, day: date) -> bool:
    """True when a memory describes something that happens "every <weekday of day>"."""
    return f"every {WEEKDAYS[day.weekday()]}" in fold(memory.text)[0]
