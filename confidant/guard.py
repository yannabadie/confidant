"""The privacy guard: decides what may leave the device and fills placeholders back in locally.

A policy maps each provider to the sensitivity levels it may see. Memories above
that level are sent as placeholders such as ``⟦m7⟧`` with only their label; the
model copies the placeholder into its answer and :meth:`Guard.late_bind` puts
the real value back on the device. :meth:`Guard.check_outbound` inspects any
outbound text (web queries, final prompts) for values of protected memories and
for credentials, and redacts or blocks them. Every decision is returned as data
so the UI can show it.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .memory import SENSITIVITIES, Memory
from .scan import KIND_PHRASES, find_secrets, find_values, fold, words

NEBIUS = "nebius_token_factory"
TAVILY = "tavily"
LOCAL = "local"

DEFAULT_POLICY: dict[str, dict[str, list[str]]] = {
    "providers": {
        NEBIUS: ["public", "personal"],
        TAVILY: ["public"],
        LOCAL: ["public", "personal", "private"],
    }
}

REDACTED = "[withheld]"
PLACEHOLDER_RE = re.compile(r"⟦\s*m(\d+)(?:v\d+)?\s*⟧")
_MEANINGFUL = re.compile(r"[^\W\d_]{3,}")
_MIN_WHOLE_TEXT = 12
_log = logging.getLogger(__name__)


def placeholder(memory_id: int) -> str:
    """The placeholder token for a memory id, e.g. ``⟦m7⟧``."""
    return f"⟦m{memory_id}⟧"


@dataclass(frozen=True)
class PreparedContext:
    """What one provider will see.

    ``shared`` memories are sent verbatim as ``[m12v2] text``; ``withheld`` ones
    as ``⟦m7⟧ (private: <label>, value withheld)``. ``context_text`` is exactly
    that text and ``placeholders`` maps each token (``"⟦m7⟧"``) to its memory.
    """

    shared: list[Memory]
    withheld: list[Memory]
    context_text: str
    placeholders: dict[str, Memory]
    provider: str = ""

    @property
    def shared_ids(self) -> list[int]:
        """Ids of the memories sent verbatim."""
        return [memory.id for memory in self.shared]

    @property
    def withheld_ids(self) -> list[int]:
        """Ids of the memories replaced by placeholders."""
        return [memory.id for memory in self.withheld]


@dataclass(frozen=True)
class GuardDecision:
    """The verdict on one outbound text.

    ``allowed`` is False when nothing may be sent; ``redacted_text`` is what may
    be sent (the original text when nothing was found); ``reasons`` explain each
    removal; ``withheld_ids`` are the memories whose values were removed.
    """

    allowed: bool
    redacted_text: str
    reasons: list[str]
    withheld_ids: list[int]
    provider: str = ""
    purpose: str = ""


@dataclass(frozen=True)
class Segment:
    """A piece of model output: plain text, or a placeholder and the memory it stands for."""

    text: str
    memory: Memory | None = None
    placeholder: bool = False

    @property
    def rendered(self) -> str:
        """The text shown to the user: the memory's value for a known placeholder."""
        return self.memory.text if self.memory is not None else self.text


class Guard:
    """Applies a sensitivity policy to everything that leaves the device."""

    def __init__(self, policy: Mapping[str, Mapping[str, Iterable[str]]] | None = None) -> None:
        providers = (DEFAULT_POLICY if policy is None else policy).get("providers")
        if not providers:
            raise ValueError('A guard policy needs a "providers" mapping.')
        self.policy: dict[str, frozenset[str]] = {}
        for provider, levels in providers.items():
            unknown = set(levels) - set(SENSITIVITIES)
            if unknown:
                raise ValueError(f"Unknown sensitivity {sorted(unknown)} for provider {provider!r}.")
            self.policy[provider] = frozenset(levels)

    def allowed(self, provider: str) -> frozenset[str]:
        """Sensitivity levels ``provider`` may see. Unknown providers raise ValueError."""
        try:
            return self.policy[provider]
        except KeyError:
            known = ", ".join(sorted(self.policy))
            raise ValueError(f"Unknown provider {provider!r}; the policy knows {known}.") from None

    def may_send(self, memory: Memory, provider: str) -> bool:
        """True when ``memory`` may be sent verbatim to ``provider``."""
        return memory.sensitivity in self.allowed(provider)

    def prepare_context(self, memories: Iterable[Memory], provider: str) -> PreparedContext:
        """Split ``memories`` into shared and withheld for ``provider`` and build the context text.

        Order is preserved; duplicates and non-current versions are dropped.
        """
        allowed = self.allowed(provider)
        shared: list[Memory] = []
        withheld: list[Memory] = []
        lines: list[str] = []
        seen: set[int] = set()
        for memory in memories:
            if memory.id in seen or memory.status != "current":
                continue
            seen.add(memory.id)
            if memory.sensitivity in allowed:
                shared.append(memory)
                lines.append(f"[{memory.cite()}] {memory.text}")
            else:
                withheld.append(memory)
                lines.append(
                    f"{placeholder(memory.id)} ({memory.sensitivity}: {memory.label}, value withheld)"
                )
        placeholders = {placeholder(memory.id): memory for memory in withheld}
        return PreparedContext(shared, withheld, "\n".join(lines), placeholders, provider)

    def segments(self, text: str, prepared: PreparedContext) -> list[Segment]:
        """Split model output into plain text and placeholders, each with its memory.

        Placeholders are recognised leniently (``⟦ m7 ⟧``, ``⟦m7v2⟧``). A UI can use
        this to show filled-in values distinctly; :meth:`late_bind` joins it.
        """
        parts: list[Segment] = []
        cursor = 0
        for match in PLACEHOLDER_RE.finditer(text):
            if match.start() > cursor:
                parts.append(Segment(text[cursor:match.start()]))
            memory = prepared.placeholders.get(placeholder(int(match.group(1))))
            parts.append(Segment(match.group(0), memory, placeholder=True))
            cursor = match.end()
        if cursor < len(text):
            parts.append(Segment(text[cursor:]))
        return parts

    def late_bind(self, text: str, prepared: PreparedContext) -> str:
        """Replace every placeholder in ``text`` with its private value, locally.

        Unknown placeholders stay visible; :meth:`unresolved` lists them and a
        warning is logged (with the tokens only, never values).
        """
        parts = self.segments(text, prepared)
        unknown = [part.text for part in parts if part.placeholder and part.memory is None]
        if unknown:
            _log.warning("Left %d unknown placeholder(s) visible: %s", len(unknown), ", ".join(unknown))
        return "".join(part.rendered for part in parts)

    def unresolved(self, text: str, prepared: PreparedContext) -> list[str]:
        """Placeholders in ``text`` that ``prepared`` does not know (left visible by late_bind)."""
        return [part.text for part in self.segments(text, prepared) if part.placeholder and part.memory is None]

    def check_outbound(
        self, text: str, provider: str, memories: Iterable[Memory] = (), *, purpose: str = ""
    ) -> GuardDecision:
        """Redact or block ``text`` before it goes to ``provider``.

        Removes credential-like values, and values of the given ``memories`` that
        ``provider`` may not see: their whole text (case-insensitive) or any of
        their dates, times, numbers, e-mail addresses or names, compared in
        normalised form ("1 180" matches "1,180", "3 Nov" matches "November 3rd").
        A value the user also stated in a memory this provider may see is not
        treated as secret. The text is blocked (``allowed=False``) when nothing
        meaningful would be left after redaction.
        """
        allowed = self.allowed(provider)
        spans: list[tuple[int, int]] = []
        reasons: list[str] = []
        withheld: list[int] = []
        for secret in find_secrets(text):
            spans.append((secret.start, secret.end))
            reasons.append(f"removed {secret.reason}")
        current = [memory for memory in memories if memory.status == "current"]
        protected = [memory for memory in current if memory.sensitivity not in allowed]
        if protected:
            outbound = _Outbound(text)
            declared = [m for m in current if m.sensitivity in allowed and m.source_type != "web"]
            exempt_keys = {value.key for memory in declared for value in find_values(memory.text)}
            exempt_words = {word for memory in declared for word in words(memory.text)}
            for memory in protected:
                hits = outbound.leaks(memory, exempt_keys, exempt_words)
                if hits:
                    spans.extend(span for span, _ in hits)
                    kinds = " and ".join(sorted({kind for _, kind in hits}))
                    withheld.append(memory.id)
                    reasons.append(
                        f"removed {kinds} from {memory.cite()} ({memory.sensitivity}: {memory.label})"
                    )
        if not spans:
            return GuardDecision(True, text, [], [], provider, purpose)
        redacted = _redact(text, spans)
        meaningful = bool(_MEANINGFUL.search(redacted.replace(REDACTED, " ")))
        if not meaningful:
            reasons.append("blocked: nothing meaningful would be left to send")
        return GuardDecision(meaningful, redacted, reasons, withheld, provider, purpose)


class _Outbound:
    """An outbound text prepared once for leak checks against many memories."""

    def __init__(self, text: str) -> None:
        self.folded, self._origin = fold(text)
        self._by_key: dict[str, list[tuple[int, int]]] = {}
        for value in find_values(text):
            self._by_key.setdefault(value.key, []).append((value.start, value.end))

    def leaks(
        self, memory: Memory, exempt_keys: set[str], exempt_words: set[str]
    ) -> list[tuple[tuple[int, int], str]]:
        """Spans of this text that reveal ``memory``, each with what kind of value it is."""
        hits: list[tuple[tuple[int, int], str]] = []
        whole = fold(memory.text)[0].strip(" .")
        if len(whole) >= _MIN_WHOLE_TEXT:
            hits += [(span, "the full text") for span in self._find(re.escape(whole))]
        for value in find_values(memory.text):
            if value.kind == "name":
                if value.folded not in exempt_words:
                    pattern = rf"(?<![\w-]){re.escape(value.folded)}(?![\w-])"
                    hits += [(span, KIND_PHRASES["name"]) for span in self._find(pattern)]
            elif value.key not in exempt_keys:
                hits += [(span, KIND_PHRASES[value.kind]) for span in self._by_key.get(value.key, [])]
        return hits

    def _find(self, pattern: str) -> list[tuple[int, int]]:
        """Original-text spans of every match of ``pattern`` in the folded text."""
        return [
            (self._origin[match.start()], self._origin[match.end() - 1] + 1)
            for match in re.finditer(pattern, self.folded)
            if match.end() > match.start()
        ]


def _redact(text: str, spans: list[tuple[int, int]]) -> str:
    """Replace the (merged) spans of ``text`` with the REDACTED marker."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    pieces: list[str] = []
    cursor = 0
    for start, end in merged:
        pieces += [text[cursor:start], REDACTED]
        cursor = end
    pieces.append(text[cursor:])
    return re.sub(r"[ \t]{2,}", " ", "".join(pieces))
