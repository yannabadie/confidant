# Router check, 2026-10-28 07:07 UTC

Model `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B`, reasoning off, max 256 tokens, temperature 0.0; each case sent 3 times. Produced by `scripts/check_router.py`; the cases were written before any run.

**40 of 48** answers as expected; 13 of 16 cases right every time; 385 ms per call on average. The same message can get a different route from one call to the next, even at temperature 0.

In the app, the router only sees messages that the local rules cannot route, that carry no date, time or amount, and that are not questions about a period of the user's time ("What's on my plate next week?" is routed on the device). When it picks a web search for a question about the user's own life, the answer still draws on memory, and the query, with names and values removed, goes to the web only if web search is on.

| message | expected | router, each call | right |
| --- | --- | --- | --- |
| What's on my plate next week? | answer | search, answer, search | **1 of 3** |
| Draft a birthday message for Léa. | answer | answer, answer, answer | yes |
| What did my coach say about race week? | answer | answer, answer, answer | yes |
| What should I get Léa for her birthday? | answer | search, search, search | **0 of 3** |
| Which restaurant did we pick for dinner with friends? | answer | search, search, search | **0 of 3** |
| How is my training going? | answer | answer, answer, answer | yes |
| Look up how to treat a blister before a race. | search | search, search, search | yes |
| What's the world record for the half marathon? | search | search, search, search | yes |
| Search the web for carb-loading tips. | search | search, search, search | yes |
| Find a recipe for a pre-race pasta dinner. | search | search, search, search | yes |
| Am I free on Tuesday afternoon? | availability | availability, availability, availability | yes |
| Do I have anything on Saturday morning? | availability | availability, availability, availability | yes |
| Run my daily brief. | brief | brief, brief, brief | yes |
| I prefer aisle seats on long train trips. | remember | remember, remember, remember | yes |
| Léa's favourite flowers are actually tulips, not peonies. | correct | correct, correct, correct | yes |
| Our weekly review moved to Thursdays. | correct | correct, correct, correct | yes |

## History (28 October 2026, single runs, written by hand)

- 06:48 UTC, the first prompt ("answer" described last, as "anything else"): 13 of 16. Wrong: "What's on my
  plate next week?", "What did my coach say about race week?" and "Which restaurant did we pick for dinner with
  friends?", all three sent to search.
- 06:49 UTC, the current prompt: 14 of 16. Wrong: the gift and restaurant questions, sent to search.
- 07:06 UTC, the current prompt: 13 of 16. Wrong: the plate, gift and restaurant questions, sent to search.
- 06:49 UTC, Nemotron 3.5 Lightning with the current prompt: 11 of 16, one reply not JSON; 361 ms per call.

Rewriting the prompt did not measurably change the results. Since then, a first-person question that names a
period of the user's time ("What's on my plate next week?") is routed on the device and never reaches the router.
