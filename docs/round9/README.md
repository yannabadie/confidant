# Round 9: the call that wrote it

`scripts/nemotron_attack_round.py` made one call to NVIDIA Nemotron 3 Ultra on Nebius Token Factory on
28 October 2026 at 19:07:24 UTC, from the Mac the project is built on (key from the macOS keychain).

- `prompt.txt`: the whole request, as sent (one user message). SHA-256
  `d2f8b10d6125fd59fd19a0def30312a259e06ee9d2d7ca044be3f3c030cdbd78`.
- `reply.txt`: the reply's content, unmodified (the reasoning trace came back apart and was not kept). SHA-256
  `5675a443239eec7ad242dd459f1f5ac30dd7da46febdd1939ea36c0d46961212`.
- `run.json`: model id as returned, both hashes, tokens (1,805 in, 3,835 out), latency, finish reason,
  temperature and the parse. The first parse kept no case, because Nemotron wrote targets as `"m7"` rather
  than `7`; the script was changed to read both, and `--from-reply` parsed the same reply again without a new
  call, rewriting only `cases_kept` and `lines_rejected` (29 kept, 1 cut-off line rejected).

The fixture `tests/fixtures/leak_prompts_round9.jsonl` is what that parse kept: Nemotron's messages and forms
as written, with `expect` set to `no_leak` and its categories prefixed `r9-`.
