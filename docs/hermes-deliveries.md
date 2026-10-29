# Hermes deliveries of the daily brief

Each entry is a brief that Hermes Agent's cron job (set up as described in
[`integrations/hermes/README.md`](../integrations/hermes/README.md)) ran and sent to Telegram, with the text as
Hermes recorded it in the job's output file. Status and time come from `hermes cron list` and `hermes cron runs`,
and the delivery from the line Hermes's gateway log writes for it.

## 27 October 2026, 07:30:22 (Paris): job `f4cb9c44c57b`, replay

Status `ok`, dispatched on time, no error, failure streak 0. **Replay mode:** the Nebius key was not yet in the
keychain, so the brief came from the sample turns written by hand that stood in for recordings until the demo
turns were first recorded live (28 October, 07:30 UTC), not from a Nemotron call; the last line says "replay".
Private items left only as their category.

```text
Confidant · daily brief (demo persona Alex Martin, synthetic data)

Good morning. Here is Friday 30 October.

Coming up:
- **Wednesday 4 Nov, 15:00**: quarterly product review with the onboarding redesign [m6v1]
- **Friday 6 Nov, 20:00**: dinner with friends near Bellecour [m9v1]

Private, filled in on your device:
- [private: health appointment]
- [private: private appointment]

Open items:
- Send the onboarding redesign slides before the review [m15v1]
- Book a table for Léa's birthday dinner by Tuesday 10 November [m14v1]

For today: your weekly review is at 17:00 [m8v1], a good moment to finish those slides.

Receipts today: 0 outbound calls, 2 replayed · 0 tokens · ≈$0.0000 · 2 withheld memories never left Confidant

Private items stay on your device; open Confidant to see them. (2 private memories never left Confidant; replay)
```

## 28 October 2026, 07:30:17 (Paris): job `f4cb9c44c57b`, replay

Status `completed`, on time. **Replay mode again:** the Nebius key went into the keychain later that morning
(the demo turns were recorded live at 08:30 Paris), so the brief came from the same hand-written sample. The text
is the same as on 27 October except for the receipts line, which counted 1 replayed call instead of 2.

```text
Receipts today: 0 outbound calls, 1 replayed · 0 tokens · ≈$0.0000 · 2 withheld memories never left Confidant
```

The key has been in the keychain since that morning (checked for presence at 18:37 Paris, never read), and
the job runs through `with-keys.sh`, which passes it to Confidant.

## 29 October 2026, 07:30:28 (Paris): job `f4cb9c44c57b`, live

Status `ok`, dispatched on time (scheduled 07:30:00), execution completed. Hermes's gateway log: "Job
'f4cb9c44c57b': delivered to telegram:[the owner's chat] via live adapter thread=- message_id=217" (07:30:28,963;
the chat id is left out here). **The first live delivery:** the brief was written by Nemotron 3 Ultra on Token
Factory, through Confidant, in one outbound call. Its receipt
([`evidence/2026-10-29-0630utc-hermes-brief/`](evidence/README.md)): 577 tokens in and 2,860 out, 7,876 ms,
about $0.0092 at list prices; memories m6, m8, m9, m14 and m15 sent, m7 and m11 withheld; payload SHA-256
`49752f32718aac24db6fb1423cac75837239fc560200e7c4317fc8b0ee335ae7`, the same as the request of the brief the
hosted demo replays (`data/replay/06_daily_brief.json`): same memories, same clock (the demo is set on Friday 30
October 2026), so the same bytes went out. Private items left only as their category.

```text
Confidant · daily brief (demo persona Alex Martin, synthetic data)

**Events (next 7 days, soonest first)**  
- Today (Fri 30 Oct) 17:00 – Weekly review [m8v1]  
- Wed 4 Nov 15:00 – Quarterly product review [m6v1]  
- Fri 6 Nov 17:00 – Weekly review [m8v1]  
- Fri 6 Nov 20:00 – Dinner with friends at the Lebanese place near Bellecour [m9v1]  

**Withheld**  
[private: health appointment]  
[private: private appointment]  

**Open to‑dos**  
- Book a table for Léa’s birthday dinner by Tue 10 Nov [m14v1]  
- Send the onboarding redesign slides to the product team before the quarterly review [m15v1]  

**Practical suggestion for today**  
Do your weekly review at 17:00 (clear inbox, check calendar for the next two weeks, pick three priorities, log running km) [m8v1] and send the onboarding redesign slides to the product team now so they’re ready before Wednesday’s quarterly review [m15v1, m6v1].

Receipts today: 1 outbound call · 3,437 tokens · ≈$0.0092 · 2 withheld memories never left Confidant

Private items stay on your device; open Confidant to see them. (2 private memories never left Confidant; live)
```

