# Instructions for coding agents working in this repo

1. Read `docs/HANDBOOK.md` in full before doing anything. It is the single source of truth:
   tracks, owners, interfaces, gates, schedule and rules.
2. Ask your human who they are if they haven't said (Siddharth, Revanth or Vishnu), then
   follow the handbook's section 0 ("Agent bootstrap") exactly: sync main, set the commit
   identity to *your human*, check the environment, find the next task in their track, tell
   them the plan in 5 lines or fewer, and start.
3. Ground rules (handbook section 3) and agent guidelines (section 4) are mandatory. Never
   train on the hidden holdout, never change a fingerprinted config key, pre-register rules
   before results, keep heavy imports lazy, and every PR must pass
   `python -m pytest -q` and `ruff check --isolated --select F src tests dashboard`.