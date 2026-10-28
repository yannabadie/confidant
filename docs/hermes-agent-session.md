# A Hermes agent on Nemotron, using Confidant over MCP (28 October 2026)

The morning brief ([`hermes-deliveries.md`](hermes-deliveries.md)) needs no model on the Hermes side. This
page is the other half: a Hermes Agent conversation where the agent itself is a Nemotron model on Token
Factory and reaches the user's memory only through Confidant's MCP tools.

## Setup

- Hermes Agent profile `confidant`, created for this demo on 28 October 2026 (config:
  [`integrations/hermes/profile-config.yaml`](../integrations/hermes/profile-config.yaml)). Model:
  `nvidia/nemotron-3-super-120b-a12b`, provider `nebius-token-factory`. The key is read from the macOS
  keychain by Hermes's `secrets.command` helper and by `scripts/with-keys.sh` for the MCP server; it is in no
  file.
- The agent gets one toolset, Confidant's MCP server (`-t mcp-confidant`), and the skill
  [`integrations/hermes/confidant/SKILL.md`](../integrations/hermes/confidant/SKILL.md) preloaded. Confidant
  runs live on a fresh copy of the synthetic persona (`--home ~/.confidant-hermes`).
- Each question is one run of
  `hermes -p confidant chat -Q --format stream-json -t mcp-confidant -s confidant --max-turns 6 -q "<question>"`,
  between 18:44:34 and 18:44:46 Paris time. Below, the tool calls and results are copied from the JSON event
  stream as Hermes wrote it; the raw streams are in
  [`evidence/2026-10-28-hermes-agent/`](evidence/2026-10-28-hermes-agent/) (`s1` to `s3`; `q1` is the earlier
  run described at the end).

## The three exchanges

### "What's on my plate next week?"

The agent called `guarded_answer` with `{"question": "What's on my plate next week?"}`. Confidant answered
with Nemotron 3 Ultra (live) and returned:

```text
- Quarterly product review on Wednesday 4 Nov 2026 at 15:00, where the design team presents the onboarding redesign. [m6v1]
- Dinner with friends at the Lebanese place near Bellecour on Friday 6 Nov 2026 at 20:00. [m9v1]
⟦m7⟧ (private: health appointment, value withheld)
⟦m11⟧ (private: private appointment, value withheld)
```

with `"guard": "Private memories stay in Confidant: an agent sees at most a placeholder and its category, and
only in answers about a period of time."` The agent passed it on unchanged and added "Private items are kept on
your device; open Confidant to see them." The two private appointments (Tuesday 3 Nov at 10:30, Thursday
5 Nov at 11:00) never reached the agent or its model, as values or as dates.

### "Am I free on Wednesday morning?"

`guarded_answer` again. Confidant checked it on the Mac, against the real dates, and made no model call:

```text
Wednesday 4 November 2026, morning: no shared item in memory.
Private items are left out of this message: whether one falls in this period shows on your device only.
```

The agent replied with that text. It cannot tell the user "you are free", because Confidant never tells it
whether a private item falls in that morning.

### "Remember that I'm seeing a cardiologist about palpitations on Monday 9 Nov at 8:30."

The agent called `memory_record` with its own choice of sensitivity and label:

```json
{"kind": "event", "label": "cardiologist visit", "text": "Seeing a cardiologist about palpitations on Monday 9 Nov at 8:30.", "sensitivity": "personal"}
```

Confidant decided the sensitivity on the device and overrode both: the memory is stored as **private**, with
the category label `health appointment`, and the agent got back only its placeholder:

```json
{"id": 26, "version": 1, "text": "⟦m26⟧ (private: health appointment, value withheld)", "sensitivity": "private", "label": "health appointment", "cite": "m26v1", "withheld": true}
```

The agent's own model had read that sentence, since it wrote the tool call, and its request had reached
Token Factory: Confidant cannot unsend it. From then on, no request Confidant answers carries the word
"cardiologist", the date or the time, and a label written by the agent would have travelled with every
placeholder.

## What it cost

| run | Hermes agent (Nemotron 3 Super) | Confidant's own calls (Nemotron 3 Ultra) |
|---|---|---|
| plate next week | 7,884 in, 221 out | 1 answer call, 417 + 704 tokens, withheld m7 and m11 |
| free Wednesday morning | 7,740 in, 128 out | none (checked on the device) |
| remember the cardiologist | 12,271 in, 287 out | none (classified on the device) |

At the list prices ($0.30 and $0.90 per million tokens for Super, $1 and $3 for Ultra), the three runs cost
under one cent for the agent and about a quarter of a cent for Confidant.

One earlier run, a minute before, is not in the table: with the first version of the skill, the agent answered
"What's on my plate next week?" with `daily_brief` instead (Confidant: 516 + 1,794 tokens, withheld m7 and
m11), so it listed the coming seven days rather than next week. The skill now says to use `guarded_answer` for
such questions and `daily_brief` only when the user asks for the brief; the runs above use that version.
