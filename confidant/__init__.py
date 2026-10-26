"""Confidant: a personal AI that keeps your secrets and shows its receipts.

Memories carry provenance and versions (:mod:`confidant.memory`); a guard decides
what may leave the device (:mod:`confidant.guard`); Nemotron on Nebius Token
Factory writes answers with placeholders that are filled in locally
(:mod:`confidant.assistant`); every outbound call leaves a receipt that holds a
hash of the payload, never the payload (:mod:`confidant.receipts`).
"""
from .assistant import BRIEF_QUESTION, Assistant, GuardReport, Turn
from .budget import Budget
from .config import VERSION as __version__
from .guard import DEFAULT_POLICY, Guard, GuardDecision, PreparedContext, Segment
from .llm import ChatResult, ModelSelectionError, TokenFactory, discover_models, pick_models
from .memory import Memory, MemoryNotFound, MemoryStore, SecretRefused, VersionConflict
from .net import ServiceError
from .receipts import Receipt, ReceiptLog, ReceiptSummary
from .replay import RecordedTurn, ReplayBook
from .search import Tavily, WebResult, record_results

__all__ = [
    "BRIEF_QUESTION",
    "DEFAULT_POLICY",
    "Assistant",
    "Budget",
    "ChatResult",
    "Guard",
    "GuardDecision",
    "GuardReport",
    "Memory",
    "MemoryNotFound",
    "MemoryStore",
    "ModelSelectionError",
    "PreparedContext",
    "Receipt",
    "ReceiptLog",
    "ReceiptSummary",
    "RecordedTurn",
    "ReplayBook",
    "SecretRefused",
    "Segment",
    "ServiceError",
    "Tavily",
    "TokenFactory",
    "Turn",
    "VersionConflict",
    "WebResult",
    "__version__",
    "discover_models",
    "pick_models",
    "record_results",
]
