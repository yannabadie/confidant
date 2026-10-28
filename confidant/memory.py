"""Versioned, provenance-tracked memories in SQLite, searchable with FTS5.

Every memory keeps its history: a correction adds a version and marks the
previous one ``superseded``; a retraction adds a final ``retracted`` version.
Text that looks like a credential is refused before it reaches the database.
"""
from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import threading
from collections.abc import Iterable
from dataclasses import asdict, dataclass, fields, replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from .scan import STOPWORDS, detect_secret, fold

KINDS = ("fact", "preference", "event", "skill", "source")
SENSITIVITIES = ("public", "personal", "private")
STATUSES = ("current", "superseded", "retracted")
SOURCE_TYPES = ("user", "web", "inference", "seed")


class SecretRefused(ValueError):
    """Raised instead of storing text that looks like a credential.

    The message says what the text looks like but never repeats the value.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(
            f"Not stored: this looks like {reason}. Confidant never keeps credentials such as "
            "passwords, PINs, card numbers or API keys; a password manager is the right place for them."
        )


class VersionConflict(RuntimeError):
    """Raised when a correction is based on a version that is no longer the latest."""

    def __init__(self, memory_id: int, expected: int, actual: int) -> None:
        self.memory_id = memory_id
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"m{memory_id} is at version {actual}, not {expected}; reload it and retry the correction."
        )


class MemoryNotFound(LookupError):
    """Raised when no memory has the requested id."""


@dataclass(frozen=True)
class Memory:
    """One version of a memory, with its provenance.

    ``supersedes`` is the version number this one replaced (None for version 1);
    ``note`` holds free-text provenance such as the reason for a retraction.
    """

    id: int
    version: int
    text: str
    kind: str
    sensitivity: str
    status: str
    source_type: str
    source_ref: str
    author: str
    created_at: str
    supersedes: int | None
    label: str
    note: str = ""

    def cite(self) -> str:
        """Citation tag for this version, e.g. ``m12v2``."""
        return f"m{self.id}v{self.version}"

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form (JSON-ready), including the citation tag."""
        return {**asdict(self), "cite": self.cite()}


_COLUMNS = tuple(field.name for field in fields(Memory))
_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id          INTEGER NOT NULL,
    version     INTEGER NOT NULL,
    text        TEXT    NOT NULL,
    kind        TEXT    NOT NULL,
    sensitivity TEXT    NOT NULL,
    status      TEXT    NOT NULL,
    source_type TEXT    NOT NULL,
    source_ref  TEXT    NOT NULL,
    author      TEXT    NOT NULL,
    created_at  TEXT    NOT NULL,
    supersedes  INTEGER,
    label       TEXT    NOT NULL,
    note        TEXT    NOT NULL DEFAULT '',
    PRIMARY KEY (id, version)
);
CREATE INDEX IF NOT EXISTS memories_by_status ON memories (status);
CREATE VIRTUAL TABLE IF NOT EXISTS memory_index USING fts5(
    text, label, tokenize = 'porter unicode61 remove_diacritics 2'
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def utc_timestamp() -> str:
    """Current time as ISO-8601 UTC with second precision, e.g. ``2026-10-30T09:12:03Z``."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# A word that names a group stands for its members: memories say "Sister Léa", not "family".
RELATED_WORDS: dict[str, tuple[str, ...]] = {
    "family": ("sister", "brother", "partner", "mother", "father", "mum", "mom", "dad", "parents", "son",
               "daughter", "wife", "husband", "aunt", "uncle", "cousin", "niece", "nephew", "grandmother",
               "grandfather"),
}
RELATED_WORDS["relatives"] = RELATED_WORDS["family"]


def match_terms(query: str) -> list[str]:
    """The distinctive words of free text, as searched: stop words and bare numbers dropped (they match too
    much), at most 16, and a group word ("family") followed by its members (RELATED_WORDS)."""
    tokens = [
        token
        for token in re.findall(r"[^\W_]+", fold(query)[0])
        if len(token) > 1 and not token.isdigit() and token not in STOPWORDS
    ]
    unique = list(dict.fromkeys(tokens))[:16]
    unique += [word for token in unique for word in RELATED_WORDS.get(token, ()) if word not in unique]
    return unique


def match_expression(query: str) -> str | None:
    """FTS5 query for free text: the words of :func:`match_terms`, each quoted (safe for the FTS5 grammar),
    OR-ed together. Returns None if nothing is left."""
    return " OR ".join(f'"{token}"' for token in match_terms(query)) or None


class MemoryStore:
    """SQLite-backed memory with versions, provenance and full-text search.

    ``path`` is a file path or ``":memory:"``. The store is safe to share
    between threads (one connection, serialised by a lock).
    """

    def __init__(self, path: str | os.PathLike[str] = ":memory:") -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._db.executescript(_SCHEMA)

    # --- writes ------------------------------------------------------------

    def record(
        self,
        text: str,
        *,
        kind: str,
        sensitivity: str,
        source_type: str,
        source_ref: str,
        author: str,
        label: str,
    ) -> Memory:
        """Store a new memory as version 1 and return it.

        Raises SecretRefused when the text or label looks like a credential, and
        ValueError when a field is empty or not one of the allowed values.
        """
        with self._lock, self._db:
            return self._create(
                text, kind=kind, sensitivity=sensitivity, source_type=source_type,
                source_ref=source_ref, author=author, label=label,
            )

    def correct(
        self, memory_id: int, new_text: str, *, author: str, source_ref: str, expected_version: int,
        sensitivity: str | None = None, label: str | None = None,
    ) -> Memory:
        """Add a corrected version of a memory and mark the previous one superseded.

        Kind carries over; so do sensitivity and label unless given (the caller decides, e.g. a correction
        that makes a memory private). A correction never makes a memory less private: a lower sensitivity
        given here keeps the current one, and its label. Raises VersionConflict when ``expected_version`` is not the latest
        version, SecretRefused for credential-like text, and ValueError if the memory was retracted.
        """
        text = _required(new_text, "new_text")
        _refuse_secrets(text, label or "")
        if sensitivity is not None:
            _choice("sensitivity", sensitivity, SENSITIVITIES)
        with self._lock, self._db:
            current = self._latest(memory_id)
            if current.status == "retracted":
                raise ValueError(f"m{memory_id} was retracted; record a new memory instead.")
            if current.version != expected_version:
                raise VersionConflict(memory_id, expected_version, current.version)
            # never lower, decided here under the lock: a caller computed its level from the version it read
            level = current.sensitivity if sensitivity is None else max(
                sensitivity, current.sensitivity, key=SENSITIVITIES.index)
            updated = replace(
                current, version=current.version + 1, text=text, status="current",
                source_type="user", source_ref=source_ref, author=author,
                created_at=utc_timestamp(), supersedes=current.version, note="", sensitivity=level,
                label=(label or current.label) if level == (sensitivity or current.sensitivity) else current.label,
            )
            self._supersede(current)
            self._insert(updated)
            self._index(updated)
        return updated

    def retract(self, memory_id: int, *, author: str, reason: str) -> Memory:
        """Retract a memory: append a final ``retracted`` version carrying the reason.

        The memory leaves search results and ``all_current``; its history stays.
        Raises ValueError if it is already retracted.
        """
        with self._lock, self._db:
            current = self._latest(memory_id)
            if current.status != "current":
                raise ValueError(f"m{memory_id} is already retracted.")
            retracted = replace(
                current, version=current.version + 1, status="retracted", source_type="user",
                source_ref="", author=author, created_at=utc_timestamp(),
                supersedes=current.version, note=_required(reason, "reason"),
            )
            self._supersede(current)
            self._insert(retracted)
            self._db.execute("DELETE FROM memory_index WHERE rowid = ?", (memory_id,))
        return retracted

    def load_seed(self, path: str | os.PathLike[str]) -> list[Memory]:
        """Load a persona file as ``seed`` memories, in one transaction.

        The file is ``{"persona": {...}, "as_of": "YYYY-MM-DD", "memories": [...]}``
        or a bare list of memories. Entries need ``text``, ``sensitivity`` and
        ``label``; ``id``, ``kind`` and ``created_at`` are optional. ``as_of`` is
        kept as the snapshot date (see :meth:`as_of`).
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        persona = data if isinstance(data, dict) else {"memories": data}
        author = str(persona.get("persona", {}).get("name") or "user")
        loaded: list[Memory] = []
        with self._lock, self._db:
            for position, entry in enumerate(persona.get("memories", []), start=1):
                missing = {"text", "sensitivity", "label"} - entry.keys()
                if missing:
                    raise ValueError(f"seed entry {position} is missing {', '.join(sorted(missing))}")
                loaded.append(self._create(
                    entry["text"], kind=entry.get("kind", "fact"), sensitivity=entry["sensitivity"],
                    source_type="seed", source_ref="seed", author=entry.get("author", author),
                    label=entry["label"], memory_id=entry.get("id"), created_at=entry.get("created_at"),
                ))
            if persona.get("as_of"):
                as_of = date.fromisoformat(persona["as_of"]).isoformat()
                self._db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('as_of', ?)", (as_of,))
        return loaded

    # --- reads -------------------------------------------------------------

    def get(self, memory_id: int) -> Memory:
        """Latest version of a memory (status ``current`` or ``retracted``).

        Raises MemoryNotFound for an unknown id.
        """
        with self._lock:
            return self._latest(memory_id)

    def history(self, memory_id: int) -> list[Memory]:
        """Every version of a memory, oldest first. Raises MemoryNotFound for an unknown id."""
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM memories WHERE id = ? ORDER BY version", (memory_id,)
            ).fetchall()
        if not rows:
            raise MemoryNotFound(f"There is no memory m{memory_id}.")
        return [_to_memory(row) for row in rows]

    def search(
        self, query: str, *, limit: int = 8, sensitivities: Iterable[str] | None = None
    ) -> list[Memory]:
        """Current memories matching ``query``, best first, the label weighted twice.

        Matching ignores case and accents and stems English words, so "Lea"
        finds "Léa" and "appointments" finds "appointment". ``sensitivities``
        restricts the result to those levels, and then the ranking is computed from
        those levels' memories only (see :meth:`_search_within`).
        """
        terms = match_terms(query)
        if not terms or limit <= 0:
            return []
        if sensitivities is not None:
            levels = [_choice("sensitivity", level, SENSITIVITIES) for level in sensitivities]
            return self._search_within(terms, levels, limit) if levels else []
        sql = (
            "SELECT m.* FROM memory_index JOIN memories AS m ON m.id = memory_index.rowid "
            "WHERE memory_index MATCH ? AND m.status = 'current' ORDER BY bm25(memory_index, 1.0, 2.0) LIMIT ?"
        )
        with self._lock:
            rows = self._db.execute(sql, [" OR ".join(f'"{term}"' for term in terms), limit]).fetchall()
        return [_to_memory(row) for row in rows]

    def _search_within(self, terms: list[str], levels: list[str], limit: int) -> list[Memory]:
        """Rank the current memories of ``levels`` that match ``terms`` with statistics of those memories alone.

        FTS5's bm25 weighs a word by how rare it is in the whole index, private memories included, so the
        order of shared results would change with the words of private ones ("tuesday friday" put a different
        memory first once a private appointment moved from Tuesday to Friday). Here FTS5 only says which
        memory holds which word (a yes or no), and the weight of a word comes from the allowed memories:
        an inverse document frequency over them, doubled for a word in the label; ties go to the lower id.
        """
        placeholders = ", ".join("?" * len(levels))
        with self._lock:
            rows = self._db.execute(
                f"SELECT * FROM memories WHERE status = 'current' AND sensitivity IN ({placeholders})", levels
            ).fetchall()
            allowed = {row["id"]: row for row in rows}
            hits: list[dict[int, float]] = []
            for term in terms:
                weights: dict[int, float] = {}
                for column, weight in (("text", 1.0), ("label", 2.0)):
                    found = self._db.execute(
                        "SELECT rowid FROM memory_index WHERE memory_index MATCH ?", (f'{column} : "{term}"',)
                    ).fetchall()
                    for (rowid,) in found:
                        if rowid in allowed:
                            weights[rowid] = weights.get(rowid, 0.0) + weight
                hits.append(weights)
        count = len(allowed)
        scores: dict[int, float] = {}
        for weights in hits:
            if not weights:
                continue
            idf = math.log(1 + (count - len(weights) + 0.5) / (len(weights) + 0.5))
            for memory_id, weight in weights.items():
                scores[memory_id] = scores.get(memory_id, 0.0) + idf * weight
        ranked = sorted(scores, key=lambda memory_id: (-scores[memory_id], memory_id))[:limit]
        return [_to_memory(allowed[memory_id]) for memory_id in ranked]

    def all_current(self) -> list[Memory]:
        """Every current memory, by id."""
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM memories WHERE status = 'current' ORDER BY id"
            ).fetchall()
        return [_to_memory(row) for row in rows]

    def all_versions(self) -> list[Memory]:
        """Every version of every memory (current, superseded and retracted), by id and version.

        The guard protects old versions too: correcting a private memory must not make
        its previous value shareable.
        """
        with self._lock:
            rows = self._db.execute("SELECT * FROM memories ORDER BY id, version").fetchall()
        return [_to_memory(row) for row in rows]

    def as_of(self) -> date | None:
        """Snapshot date of the loaded seed persona, or None if no seed set one."""
        with self._lock:
            row = self._db.execute("SELECT value FROM meta WHERE key = 'as_of'").fetchone()
        return date.fromisoformat(row["value"]) if row else None

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self._db.close()

    # --- internals (callers hold the lock and the transaction) ---------------

    def _create(
        self,
        text: str,
        *,
        kind: str,
        sensitivity: str,
        source_type: str,
        source_ref: str,
        author: str,
        label: str,
        memory_id: int | None = None,
        created_at: str | None = None,
    ) -> Memory:
        """Validate and insert version 1 of a memory; ``memory_id`` is assigned if None."""
        text = _required(text, "text")
        label = _required(label, "label")
        _refuse_secrets(text, label)
        if memory_id is None:
            row = self._db.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM memories").fetchone()
            memory_id = int(row[0])
        elif int(memory_id) < 1:
            raise ValueError(f"memory ids start at 1; got {memory_id}")
        elif self._db.execute("SELECT 1 FROM memories WHERE id = ?", (int(memory_id),)).fetchone():
            raise ValueError(f"memory id {memory_id} already exists")
        if created_at is not None:
            datetime.fromisoformat(created_at.replace("Z", "+00:00"))  # ValueError if malformed
        memory = Memory(
            id=int(memory_id), version=1, text=text,
            kind=_choice("kind", kind, KINDS),
            sensitivity=_choice("sensitivity", sensitivity, SENSITIVITIES),
            status="current",
            source_type=_choice("source_type", source_type, SOURCE_TYPES),
            source_ref=str(source_ref), author=_required(author, "author"),
            created_at=created_at or utc_timestamp(), supersedes=None, label=label,
        )
        self._insert(memory)
        self._index(memory)
        return memory

    def _latest(self, memory_id: int) -> Memory:
        """Latest version of a memory, or MemoryNotFound."""
        row = self._db.execute(
            "SELECT * FROM memories WHERE id = ? ORDER BY version DESC LIMIT 1", (memory_id,)
        ).fetchone()
        if row is None:
            raise MemoryNotFound(f"There is no memory m{memory_id}.")
        return _to_memory(row)

    def _insert(self, memory: Memory) -> None:
        """Insert one memory version row."""
        values = asdict(memory)
        self._db.execute(
            f"INSERT INTO memories ({', '.join(_COLUMNS)}) VALUES ({', '.join('?' * len(_COLUMNS))})",
            [values[column] for column in _COLUMNS],
        )

    def _supersede(self, memory: Memory) -> None:
        """Mark one version as superseded."""
        self._db.execute(
            "UPDATE memories SET status = 'superseded' WHERE id = ? AND version = ?",
            (memory.id, memory.version),
        )

    def _index(self, memory: Memory) -> None:
        """Make ``memory`` the searchable text for its id."""
        self._db.execute("DELETE FROM memory_index WHERE rowid = ?", (memory.id,))
        self._db.execute(
            "INSERT INTO memory_index (rowid, text, label) VALUES (?, ?, ?)",
            (memory.id, memory.text, memory.label),
        )


def _to_memory(row: sqlite3.Row) -> Memory:
    """Build a Memory from a database row."""
    return Memory(**{column: row[column] for column in _COLUMNS})


def _required(value: object, name: str) -> str:
    """``value`` as a single-spaced string; ValueError if it is empty."""
    cleaned = " ".join(str(value).split())
    if not cleaned:
        raise ValueError(f"{name} must not be empty")
    return cleaned


def _choice(name: str, value: str, allowed: tuple[str, ...]) -> str:
    """``value`` if it is one of ``allowed``, else ValueError naming the options."""
    if value not in allowed:
        raise ValueError(f"{name} must be one of {', '.join(allowed)}; got {value!r}")
    return value


def _refuse_secrets(*texts: str) -> None:
    """Raise SecretRefused if any of ``texts`` looks like a credential."""
    for text in texts:
        reason = detect_secret(text)
        if reason:
            raise SecretRefused(reason)
