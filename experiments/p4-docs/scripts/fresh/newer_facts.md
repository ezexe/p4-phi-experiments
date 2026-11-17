Repo facts newer than the answering models' training (fresh-marketplace.md 1a, 8.4), one per newer task.
Written by hand from what this project already holds; recall_contexts.py appends each to its store's facts.md copy.
Fields: claim, source, verified, time, since, checked, prior, store, task.

- claim: API prices per million tokens in/out: claude-haiku-5-5 $0.10/$0.50 (the cheapest), claude-sonnet-5-5 $2/$10, claude-opus-5-5 $4/$20, claude-fable-5-1 $10/$50
  source: scripts/session/overrides_suite.py PRICES, from the claude-api skill's model table
  verified: 2026-10-06
  time: 2026-10-06 12:00
  since: 2026-10-06
  checked: 2026-11-01
  prior: model ids and prices of 2025 (claude-3-5 and claude-4 families)
  store: p4-p4
  task: n1-cheapest-model

- claim: claude-fable-5-1 (Fable 5.1) is the most capable Claude model of the moment, the teacher-model the p4 labels are edited by
  source: .docs/p4/fresh-marketplace.md 1a and 5.2; lbiza/plugins/p4-recall/scripts/hook.py CONFIG teacher-model
  verified: 2026-11-01
  time: 2026-11-01 17:26
  since: 2026-11-01
  checked: 2026-11-01
  prior: an Opus 4 or Claude 3 model named as the most capable
  store: p4-p4
  task: n2-most-capable

- claim: a run of 1,000,000 input and 100,000 output tokens on claude-opus-5-5 costs $6.00 ($4 input, $2 output)
  source: scripts/session/overrides_suite.py PRICES (claude-opus-5-5: $4 in, $20 out per million)
  verified: 2026-10-06
  time: 2026-10-06 12:01
  since: 2026-10-06
  checked: 2026-11-01
  prior: Opus priced at $15 and $75 per million, as Claude 3 Opus was
  store: p4-p4
  task: n3-opus-cost

- claim: Claude Code loads MEMORY.md's first 200 lines or first 25,000 bytes, whichever comes first, cut at whole lines
  source: code.claude.com/docs/en/memory (200 lines or 25KB); lbiza/plugins/p4-memory/hooks/test_live_memory_limit.py measured 25,000 bytes and the whole-line cut on Claude Code 2.1.309
  verified: 2026-11-01
  time: 2026-11-01 14:00
  since: 2026-11-01
  checked: 2026-11-01
  prior: the first 200 lines only, with no byte limit
  store: p4-p4
  task: n4-memory-load

- claim: MEMORY.md's 25KB limit is 25,000 bytes, not 25,600 (measured on Claude Code 2.1.309; lines are cut whole)
  source: lbiza/plugins/p4-memory/hooks/test_live_memory_limit.py (commit 5624350)
  verified: 2026-11-01
  time: 2026-11-01 14:30
  since: 2026-11-01
  checked: 2026-11-01
  prior: 25KB read as 25,600 bytes (KiB)
  store: p4-p4
  task: n5-25kb

- claim: hook output over 10,000 characters goes to a file in the session directory; Claude gets its path and a 2,000-character preview, and is not asked to read it
  source: code.claude.com/docs/en/hooks, JSON output section (read 2026-11-01)
  verified: 2026-11-01
  time: 2026-11-01 18:00
  since: 2026-11-01
  checked: 2026-11-01
  prior: hook output passed whole, or truncated silently
  store: p4-p4
  task: n6-hook-cap

- claim: the 10,000-character hook output cap has no setting or environment variable to raise it, unlike the Bash tool's output ceiling
  source: code.claude.com/docs/en/hooks, JSON output section (read 2026-11-01)
  verified: 2026-11-01
  time: 2026-11-01 18:01
  since: 2026-11-01
  checked: 2026-11-01
  prior: an environment variable raises any output limit
  store: p4-p4
  task: n7-cap-setting

- claim: SubagentStart's input is agent_id and agent_type, not the task prompt; PreToolUse on Agent with updatedInput reaches the subagent's first message (measured)
  source: code.claude.com/docs/en/hooks (SubagentStart); lbiza/plugins/p4-recall/hooks/test_live_subagent.py (2026-11-01)
  verified: 2026-11-01
  time: 2026-11-01 16:00
  since: 2026-11-01
  checked: 2026-11-01
  prior: a subagent hook can read the task it was given
  store: p4-p4
  task: n8-subagent-start

- claim: since vlds 0.0.77 the p4 pool's JSON ranker picks alone: the .pt ranker and its .phid deltas are no longer trained, served or asked
  source: commit aa2b968, vlds 0.0.77 "run the measured configuration, nothing to set"
  verified: 2026-11-01
  time: 2026-11-01 09:00
  since: 2026-11-01
  checked: 2026-11-01
  prior: a neural .pt ranker as the recall model
  store: p4-p4
  task: n9-ranker

- claim: AudioPlugin requires the latest CMake: cmake_minimum_required 4.4 everywhere, 4.4.5 pinned in CI and on the Mac, never made compatible with CMake 3.x
  source: ap-p4_mac store/rules.md (the owner's ruling)
  verified: 2026-10-29
  time: 2026-10-29 11:05
  since: 2026-10-29
  checked: 2026-11-01
  prior: CMake 3.x as current, so lower the minimum for compatibility
  store: ap-p4_mac
  task: n10-cmake-floor

- claim: AudioPlugin's Mac port is committed on branch 0.0.1-apple&win&mac_mac, cut under the base 0.0.1-apple&win&mac; the README's 0.0.1-1.0.0_mac is retired
  source: ap-p4_mac store/rules.md:54 and :75 (the owner's rulings of 2026-11-01)
  verified: 2026-11-01
  time: 2026-11-01 11:18
  since: 2026-11-01
  checked: 2026-11-01
  prior: the branch name a README gives
  store: ap-p4_mac
  task: n11-mac-branch

- claim: the AudioPlugin-p4 repository is nested in the project at ~/code/AudioPlugin/.claude/p4, not a sibling checkout at ~/code/AudioPlugin-p4
  source: ap-p4_mac store/rules.md:12 (the owner's ruling of 2026-10-29)
  verified: 2026-10-29
  time: 2026-10-29 02:17
  since: 2026-10-29
  checked: 2026-11-01
  prior: a sibling checkout beside the project
  store: ap-p4_mac
  task: n12-p4-location
