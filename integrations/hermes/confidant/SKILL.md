---
name: confidant
description: Answer from the user's personal memory through Confidant's privacy guard, keep that memory up to date, and send the daily brief, without ever handling private values.
---

# Confidant: guarded personal memory

Confidant is an MCP server (tools `memory_search`, `memory_record`, `memory_correct`, `memory_history`,
`guarded_answer`). It holds the user's memories, each marked public, personal or private. Private
memories never reach you in clear: they appear as a placeholder with a label, such as
`⟦m7⟧ (private: dermatologist appointment)`. The value is filled in only on the user's device.

## When the user asks something about their own life
Call `guarded_answer` with the user's question, word for word. Return its answer as it is, citations
such as `[m6v1]` included. If it contains placeholders, keep them exactly and add one line: "Private items
are kept on your device; open Confidant to see them." Never ask the user to type a private value.

## When the user states a fact to remember, or corrects one
- New fact: `memory_record` with the text, a kind and a 2–4 word label that names only the category
  ("medical appointment", "job search"), never a diagnosis, a name, a number or a date: the label is
  what leaves the device in place of a private value.
  Confidant refuses credentials (passwords, PINs, card numbers, API keys); say so if it does.
- Correction: read the memory with `memory_history`, then `memory_correct` with the id, the new text and
  the current version as `expected_version`. A version conflict means read again, never overwrite.

## Daily brief
The scheduled brief needs no model on your side: a Hermes cron job in no-agent mode runs
`python -m confidant brief --delivery` and delivers its output, where each private item is only its label.
If the user asks for the brief in a conversation, call `guarded_answer` with "Run my daily brief." and
return the answer unchanged: placeholders stay placeholders.

## Never
Store or repeat a secret, guess a withheld value, or start Confidant with `--reveal-private`.
