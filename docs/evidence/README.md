# Evidence of the live runs

Raw records of runs on Nebius Token Factory, copied unmodified from the Mac they ran on, so that the figures in
the README and the Devpost text can be checked against something other than prose. Each file's SHA-256 is
listed below. They hold no key: keys came from the macOS keychain through `scripts/with-keys.sh` or a Hermes
`secrets.command`, and only reached the child processes' environment.

| file | what it is | what it shows |
|---|---|---|
| `2026-10-28-1748utc-replayed-run-receipts.txt` | `python -m confidant receipts` after `scripts/record_demo_turns.sh`, 28 Oct 2026, 17:47:58 to 17:48:10 UTC | the run the hosted demo replays (`data/replay/`): 5 calls, 5,824 tokens, about $0.0126 at list prices; per call: purpose, tokens in and out, latency, payload SHA-256, memories sent and withheld, model id |
| `2026-10-28-1748utc-replayed-run-transcript.txt` | the same run's chat transcript, as the recorder printed it | the six demo turns, with the device's late-bound values (the persona is synthetic) |
| `2026-10-28-0634utc-first-live-run-receipts.txt` | receipts of the first live run of the demo script, 28 Oct 2026, 06:34 UTC | the brief hit its 1,024-token limit (504+1024), the reply whose unfinished reasoning trace came back as the answer; the router then ran with reasoning on (256 of 256 tokens). Both are fixed since: see `docs/token-factory-probe.md` |
| `2026-10-28-hermes-agent/q1.jsonl`, `s1.jsonl`, `s2.jsonl`, `s3.jsonl` | Hermes Agent's own event stream (`--format stream-json`) for the three exchanges of `docs/hermes-agent-session.md` (`s1` to `s3`) and the earlier run described at its end (`q1`), profile `confidant`, model `nvidia/nemotron-3-super-120b-a12b` (set in [`integrations/hermes/profile-config.yaml`](../../integrations/hermes/profile-config.yaml); the streams' own `model` field is empty) | every tool call the agent made to Confidant's MCP server and every reply it got, word for word |
| `2026-10-28-hermes-agent/session-digest.json` | a digest of the three exchanges (`s1` to `s3`) written when they were recorded | tool calls, replies and token counts per session |
| `2026-10-28-2124utc-filmed-run/receipts-tab.jpg`, `receipts-zoomed.jpg`, `prove-it.jpg` | frames of the screen recording of the run filmed for the video, 28 Oct 2026 from 21:24:39 UTC, live on Token Factory at commit `ffefbbc` (no receipts file: the web app keeps a session's receipts in memory) | its Receipts tab: 5 outbound calls, 2,206 tokens sent and 2,872 received, about $0.0104, 2 withheld memories that never left; the daily brief (577 tokens in, 1,485 out, 3,926 ms, $0.0050, sent m6 m8 m9 m14 m15, withheld m7 m11) and the birthday message (419 in, 421 out, 1,368 ms, $0.0017); the Prove it check, both SHA-256 `553904932b5ae926…`, the hash of the recorded request in `data/replay/` |
| `2026-10-28-2124utc-filmed-run/run.json` | the recorder's record of that run, paths left out | start time, app header (live, models), the clips and their length |

```
52c075608a4e77145db0d5f3a738df450842e77e69cd7f45ed96429aa36e25dc  2026-10-28-1748utc-replayed-run-receipts.txt
e230f8c18edf900556282eecae4b5386fb4a9c594043a63bf4d610d53f2a2055  2026-10-28-1748utc-replayed-run-transcript.txt
eb5ffbce3f87dd3018efc5b78126451f170d7bf7c4d12c3d2269215069d39ae8  2026-10-28-0634utc-first-live-run-receipts.txt
265a1a54642a1dc5ea682fd3fe9720259778aa61c47e8edbd10d7306205422b5  2026-10-28-hermes-agent/q1.jsonl
9a031e97dc5da69aac07213ea68ff404c56a690d9b5d4788eda6ec3936c30dc3  2026-10-28-hermes-agent/s1.jsonl
a14b419472a596b5f6c8df35e2e8f6a621634c383d7e8e8225e17666dbae02ef  2026-10-28-hermes-agent/s2.jsonl
e642aa4eb5c1e0cbea701e7478f8d082ca050912bc87eb478c725835c250542d  2026-10-28-hermes-agent/s3.jsonl
9b69f4f4802dce02e274275484e6be4764c9751436ffe20bb59074021019d9d0  2026-10-28-hermes-agent/session-digest.json
2df394142138305fb8f187b9805192085fbe660ea421e0eec05eeb7d7ba0a94e  2026-10-28-2124utc-filmed-run/receipts-tab.jpg
5f430472d9d2d2cd4fd32e4598370a7012e3bf628e8fc9aaccec82d68d255a1e  2026-10-28-2124utc-filmed-run/receipts-zoomed.jpg
02f886b6e950fb14a8c824c4c1da3043b6db54383d28e2d58162ecae5524ee9e  2026-10-28-2124utc-filmed-run/prove-it.jpg
495dcfbb9bb412a0738ddd10d6a0696dd958f15e4c67922a8c5c8d763a1749df  2026-10-28-2124utc-filmed-run/run.json
```

What these files cannot show: that Token Factory received exactly these payloads. The receipts are Confidant's
own record (the SHA-256 is of the messages it sent); the Hermes streams are Hermes's own record.
