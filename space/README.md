---
title: Confidant
emoji: 🔏
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 5.49.1
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: A personal AI that keeps secrets and shows receipts
---

# Confidant

A personal AI that keeps your secrets and shows its receipts.

Confidant answers from your own memories. Every memory records where it came from and every version
it has had. A guard decides, memory by memory, what may leave the device: private memories reach
NVIDIA Nemotron on Nebius Token Factory only as placeholders such as `⟦m7⟧`, and the real values are
filled in locally once the answer is back. Every outbound call is logged with its tokens, a cost
estimate and a SHA-256 of the payload, never the payload itself.

## Try it

Run the seven-step demo script on the left of the app:

1. **What's on my plate next week?** Private appointments are filled in on the device; the Guard tab
   shows the exact text Nemotron received.
2. **My dermatologist appointment moved…** The correction becomes version 2; select its source chip to
   see the history.
3. **Find a 10-day taper plan for my half marathon.** The web query passes the guard first. The demo turns
   were recorded with web search off, so the Guard tab shows the guarded query, marked as not sent; with a
   Tavily key, the results are stored with their URLs.
4. **Draft a birthday message for Léa.** Personal memories may go to the cloud model, and the answer
   cites them.
5. **Remember my bank PIN is 4821.** Refused: credentials are never stored or sent.
6. **Run my daily brief.** The week ahead, plus today's receipts.
7. **Am I free Wednesday morning?** Checked on the device against the real dates, private ones included:
   no model call at all, since no cloud model may learn when a private item happens. After step 2 the
   moved appointment makes the answer "not free".

The persona, Alex Martin, is fictional. Each browser tab gets its own copy of the seed memory, so your
changes are visible only to you and disappear when the tab closes.

## Live and replay

- **Live** when the Space has a `NEBIUS_API_KEY` secret (web search also needs `TAVILY_API_KEY`). The
  header names the Nemotron models in use. Usage is capped per day and per session; when a cap is
  reached the app switches to replay and says so.
- **Replay** otherwise. The demo turns recorded live on 28 October 2026 (`data/replay/`: Nemotron 3 Ultra
  and Nano on Token Factory) stand in for the cloud model; memory, guard, late binding and receipts still run
  in the app, and every replayed turn is labelled.

Code, architecture and the MCP server for agents are described in the repository's `README.md`.
