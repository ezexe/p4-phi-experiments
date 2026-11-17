## VLDS memory override (always active)

This project keeps memory for each session under `.claude/p4/sessions/<session>/vlds/`. Hooks do the bookkeeping. You do three things:

1. At the first message, start the memory reader in the background: the Agent tool, general-purpose, `run_in_background` true, prompt "Read lbiza/plugins/vlds/hooks/operator-prompt.md and do what it says", followed by the store path, the session id, the latest `now:` and the user's message. Use what it returns: the user's standing rules that bear on the task.
2. When the user corrects you, states a preference, or takes one back, keep their exact words for the end of the turn. Before your final message, run `python lbiza/plugins/vlds/scripts/record.py --store <store> --session <id> --now <now> --record <notes>` with them.
3. Don't open or write the store's files yourself.

Talk to the user in plain words. The store's internal terms are not for replies. Details: the vlds:guide skill.
