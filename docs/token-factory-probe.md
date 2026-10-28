# How the Nemotron models spend their tokens on Token Factory

Two runs of [`scripts/probe_reasoning.py`](../scripts/probe_reasoning.py) on 28 October 2026. Each run plays the six
demo turns live on a throwaway memory seeded with the synthetic persona, keeps the exact messages of every model
call, then sends the router's and the brief's calls again with variants. Columns: how the call ended
(`finish_reason`), completion tokens, latency, the length of the trace returned apart in `reasoning_content`,
whether `<think>` tags appeared in the text, the route parsed from the reply, and how the text begins.

What the runs showed, and what Confidant does about it:

- **A trace that does not close arrives as the answer.** With 1,024 tokens, Nemotron 3 Ultra spent the whole
  budget reasoning about the daily brief. The reply came back with `finish_reason: "length"`, nothing in
  `reasoning_content`, and the unfinished trace as `content`, without tags: Confidant showed it as the brief.
  With a closed trace, the same model returns it apart in `reasoning_content`. Now a reply that stops at the
  token limit is never used: the turn falls back to a local answer and says why, and it is never recorded for
  replay. The answer budget is now 8,192 tokens: with its trace, the brief took 1,236 to 1,874 tokens in the
  runs below, 2,146 in the first recording of the demo turns (07:30 UTC), 1,214 in the session filmed for the
  video (07:37 UTC) and 1,352 in the recording the hosted demo replays (17:48 UTC).
- **The router reasoned away its budget.** Nemotron 3 Nano spent its 256 tokens thinking in both routed turns and
  returned an empty answer, or an unfinished JSON object; `/no_think` at the end of the system prompt did not
  stop it. With `chat_template_kwargs: {"enable_thinking": false}` it answered in 13 to 43 tokens. The router now
  runs with reasoning off, and the answers with it on.
- **The same message, routed two ways.** With the final router prompt and reasoning off, at temperature 0, Nano
  routed "What's on my plate next week?" to a web search in five calls out of seven: twice in the second run
  below, once in a single router check, twice in the three-call check of [`docs/router-check.md`](router-check.md);
  the two other calls answered from memory. A first-person question that names a period of the user's time is
  now routed on the device. (The second run predates that rule, which is why its first turn still calls the
  router.) Rewriting the router prompt, to make answering from memory the default, did not measurably help:
  13 of 16 cases right in one run before, 14 and then 13 in single runs after, 40 of 48 calls over three runs.

## Before the fix (06:42 UTC): 1,024 tokens for answers, reasoning on everywhere

Models: reason `nvidia/Nemotron-3-Ultra-550b-a55b`, fast `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B`.

### Calls made by the six demo turns

| purpose | model | max_tokens | finish | completion tokens |
| --- | --- | --- | --- | --- |
| route | NVIDIA-Nemotron-3-Nano-30B-A3B | 256 | length | 256 |
| answer | Nemotron-3-Ultra-550b-a55b | 1024 | stop | 460 |
| answer | Nemotron-3-Ultra-550b-a55b | 1024 | length | 1024 |
| route | NVIDIA-Nemotron-3-Nano-30B-A3B | 256 | length | 256 |
| answer | Nemotron-3-Ultra-550b-a55b | 1024 | stop | 212 |
| brief | Nemotron-3-Ultra-550b-a55b | 1024 | length | 1024 |

### Variants

| call | variant | finish | completion | ms | reasoning_content chars | tags in text | route JSON | text begins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| route #0 | as sent (256) | length | 256 | 5946 | 1104 | no | invalid |  |
| route #0 | budget x4 (1024) | stop | 482 | 8188 | 2134 | no | availability |  / {"intent":"availability"} |
| route #0 | thinking off (256) | stop | 13 | 3321 | 0 | no | search | {"intent":"search","query":"user schedule next week"} |
| route #3 | as sent (256) | length | 256 | 7227 | 1025 | no | invalid |  / { /   "intent": "answer", /   "text": "Happy Birthday, [name]! Wishing you a day |
| route #3 | budget x4 (1024) | stop | 214 | 5530 | 766 | no | answer |  / { /   "intent": "answer", /   "text": "Happy birthday, [name]! Wishing you a wonderful day filled with joy and cherished moments." / } |
| route #3 | thinking off (256) | stop | 39 | 3651 | 0 | no | answer | {"intent":"answer","text":"Happy Birthday, [name]! Wishing you a wonderful day filled with joy and cherished moments.","target":"[name]","qu |
| brief #5 | as sent (1024) | length | 1024 | 2908 | 0 | no |  | The user wants a daily brief for the next 7 days from today (Friday 30 October 2026). They want events of the next 7 days, soonest first, th |
| brief #5 | budget x4 (4096) | stop | 1874 | 4793 | 4797 | no |  | **Events (next 7 days, soonest first)**   / - Friday 30 Oct 2026, 17:00 – Weekly review [m8v1]   / - Wednesday 4 Nov 2026, 15:00 – Quarterly pro |
| brief #5 | thinking off (1024) | stop | 259 | 1255 | 0 | no |  | **Daily Brief — Friday 30 October 2026** /  / **Next 7 days (soonest first)**   / - Weekly review — today at 17:00 [m8v1]   / - Quarterly product re |

## After the fix (06:50 UTC): 4,096 tokens for answers, the router with reasoning off

Models: reason `nvidia/Nemotron-3-Ultra-550b-a55b`, fast `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B`.

### Calls made by the six demo turns

| purpose | model | reasoning | max_tokens | finish | completion tokens |
| --- | --- | --- | --- | --- | --- |
| route | NVIDIA-Nemotron-3-Nano-30B-A3B | off | 256 | stop | 28 |
| answer | Nemotron-3-Ultra-550b-a55b | on | 4096 | stop | 657 |
| answer | Nemotron-3-Ultra-550b-a55b | on | 4096 | stop | 905 |
| route | NVIDIA-Nemotron-3-Nano-30B-A3B | off | 256 | stop | 43 |
| answer | Nemotron-3-Ultra-550b-a55b | on | 4096 | stop | 250 |
| brief | Nemotron-3-Ultra-550b-a55b | on | 4096 | stop | 1462 |

### Variants

| call | variant | finish | completion | ms | reasoning_content chars | tags in text | route JSON | text begins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| route #0 | as sent (256) | stop | 24 | 1349 | 0 | no | search | {"intent":"search","text":"What's on my plate next week?","query":"my schedule next week"} |
| route #0 | budget x4 (1024) | stop | 29 | 1351 | 0 | no | search | {"intent": "search", "text": "What's on my plate next week?", "query": "upcoming events next week"} |
| route #3 | as sent (256) | stop | 43 | 3334 | 0 | no | answer | { /   "intent": "answer", /   "text": "Happy Birthday, [name]! Wishing you a wonderful day filled with joy and cherished moments.", /   "target":  |
| route #3 | budget x4 (1024) | stop | 43 | 3867 | 0 | no | answer | { /   "intent": "answer", /   "text": "Happy Birthday, [name]! Wishing you a wonderful day filled with joy and cherished moments.", /   "target":  |
| brief #5 | as sent (4096) | stop | 1236 | 5930 | 3155 | no |  | **Events (next 7 days, soonest first)**   / - Today, Friday 30 Oct 2026, 17:00 – Weekly review: clear inbox, check calendar for next two weeks |
| brief #5 | budget x4 (16384) | stop | 1684 | 7445 | 4312 | no |  | **Events (next 7 days, soonest first)**   / - Friday 30 Oct 2026, 17:00 – Weekly review (clear inbox, check calendar for next two weeks, pick  |
| brief #5 | thinking off (4096) | stop | 283 | 1512 | 0 | no |  | **Daily Brief — Friday 30 Oct 2026** /  / **Next 7 days (soonest first)** / - Weekly review — every Friday at 17:00 (clear inbox, check calendar f |
