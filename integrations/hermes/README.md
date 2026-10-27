# Confidant in Hermes Agent

Two pieces, both running on the user's Mac:

1. **The MCP server**, for conversations: an entry under `mcp_servers` in the Hermes profile's
   `config.yaml` whose command runs, from the repository folder,
   `/bin/sh scripts/with-keys.sh .venv/bin/python -m confidant --home ~/.confidant serve-mcp`
   (`with-keys.sh` reads the API keys from the macOS keychain, so none is written in the config).
   `hermes mcp test confidant` then lists the six tools. The skill in [`confidant/SKILL.md`](confidant/SKILL.md)
   (copied to the profile's `skills/` folder) tells the agent how to use them: answers through
   `guarded_answer`, facts through `memory_record`, the brief through `daily_brief`.

2. **The morning brief**, without any model on the Hermes side: a cron job in no-agent mode runs
   [`confidant-brief.sh`](confidant-brief.sh) every day at 07:30 and delivers its standard output to
   Telegram (on failure, the error is delivered locally instead). The script prints
   `python -m confidant brief --delivery`: the brief built from the draft, where each private item is only
   its label, for example `[private: health appointment]`.

On our Mac the job was created on 26 October 2026 (job id `f4cb9c44c57b`). Each delivery is listed in
[`docs/hermes-deliveries.md`](../../docs/hermes-deliveries.md) as soon as it has happened, with the text
Telegram received.
