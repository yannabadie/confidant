"""Deterministic local rules: the assistant's fallback when no model may, or can, be asked.

They route messages by keywords, classify sensitivity, kind and label, build web
queries, and read time windows and event dates. Everything runs on the device,
which is why messages that touch private memories are routed here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta

from .memory import SENSITIVITIES, Memory
from .scan import CALENDAR_WORDS, STOPWORDS, WEEKDAYS, find_dates, find_values, fold, words

INTENTS = ("answer", "remember", "correct", "search", "brief")


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
_POSSESSIVE = re.compile(r"\b(?:for me|my|our|me)\b\s*", re.I)

_HEALTH_TERMS = (
    r"doctor|gp|nurse|dermatolog\w*|dentist|physio\w*|therap\w*|psycholog\w*|psychiatr\w*|diagnos\w*|"
    r"medication|medicine|prescription|pills?|symptoms?|clinic|hospital|blood test|surgery|pregnan\w*|"
    r"colonoscop\w*|endoscop\w*|biops\w*|mammogra\w*|ultrasound|mri|x-?rays?|scan results?|lab results?|"
    r"test results?|chemo\w*|cancer|tumou?rs?|diabet\w*|asthma|epilep\w*|hiv|aids|sti|std|hepatitis|"
    r"antidepressants?|ssris?|anxiety|depress\w*|adhd|bipolar|rehab|addict\w*|sobriety|ivf|fertility|"
    r"miscarr\w*|abortion|contracepti\w*|gyn(?:a)?ecolog\w*|urolog\w*|oncolog\w*|cardiolog\w*|"
    r"neurolog\w*|illness|disease|sick leave|medical|counsell?ors?|counsell?ing|grief|bereave\w*|"
    r"medecin|ordonnance|maladie|enceinte|grossesse|hopital|clinique|prise de sang|arret maladie|mutuelle"
)
_MONEY_TERMS = (
    r"rent|salary|salaries|wage|bank|loan|debt|mortgage|savings|tax|taxes|income|invoice|payslip|"
    r"credit|inheritance|heritage|housing costs?|monthly payments?|(?:utility|phone|electricity|energy|water|gas) bills?|overdraft|paycheck|pension|"
    r"insurance|loyer|salaire|impots?|banque|"
    r"pret|dette|epargne"
)
_JOB_TERMS = (
    r"interview\w*|recruiter|job offer|resign\w*|notice period|severance|redundan\w*|laid off|layoffs?|fired|"
    r"unemploy\w*|burn-?out|salary negotiation|entretien d'embauche|recruteur|demission|licenciement|chomage"
)
_LEGAL_TERMS = (
    r"confidential|lawyer|court|lawsuit|divorce|police|passport|visa|residence permit|id card|social security|"
    r"avocat|tribunal|plainte|titre de sejour|carte d'identite"
)
_INTIMATE_TERMS = r"sexual\w*|sexuality|religio\w*|political party|union member\w*"
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
    """Route a message by keywords: brief, correct, remember, search, else answer."""
    if _BRIEF.search(message):
        return Route("brief")
    if _CORRECTION.search(message):
        return Route("correct", text=memory_text_from(message))
    if _REMEMBER.search(message):
        return Route("remember", text=memory_text_from(message))
    if _SEARCH.search(message):
        return Route("search", query=web_query_from(message))
    return Route("answer")


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
    """Sensitivity by keywords: private for health, money, legal or job-search topics,
    public for work tooling without personal context, personal otherwise."""
    folded = fold(text)[0]
    if _PRIVATE_TERMS.search(folded) or _MONEY.search(folded):
        return "private"
    if _PUBLIC_TERMS.search(folded) and not _PERSONAL_TERMS.search(folded):
        return "public"
    return "personal"


def classify_new_fact(text: str) -> str:
    """Sensitivity of a fact the user asks Confidant to keep. Keyword lists cannot name every
    illness, debt or dispute, so a fact is private unless the rules see something clearly
    everyday (family, friends, plans, tastes) or work-public. When in doubt, private."""
    level = classify_sensitivity(text)
    if level != "personal":
        return level
    folded = fold(text)[0]
    return "personal" if _PERSONAL_TERMS.search(folded) or _PREFERENCE_TERMS.search(folded) else "private"


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
    """The period a question asks about, if any: an explicit date, today, tomorrow,
    this or next week, the weekend, "next N days", or two weeks for agenda words."""
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


def event_date(memory: Memory, today: date) -> date | None:
    """The first date written in a memory, resolved relative to ``today``."""
    found = find_dates(memory.text)
    return found[0].resolve(today) if found else None


def is_open_item(memory: Memory) -> bool:
    """True for to-dos: text starting with "To do" or mentioning something pending."""
    return bool(_OPEN_ITEM.search(fold(memory.text)[0]))


def recurs_on(memory: Memory, day: date) -> bool:
    """True when a memory describes something that happens "every <weekday of day>"."""
    return f"every {WEEKDAYS[day.weekday()]}" in fold(memory.text)[0]
