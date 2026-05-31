# Verified Status — 2026-05-31

This doc supersedes the 2026-05-30 session summaries, which inflated progress.
Everything below was checked against running code, not docs.

## Corrections to prior claims

| Prior claim | Reality |
|---|---|
| "Ray PyPI blocker, system-level, unresolvable" | **FALSE.** Ray 2.55.1 installed + working under py-3.11. `pip index versions ray` lists all 128 versions. Prior session likely tested under py-3.12 (the gate CLAUDE.md sidesteps). |
| "Phase 3 survival validation PASSED" | **Vacuous smoke test.** Carpet fake players are damage/hunger-immune → health pinned at 20.0 → the only fail condition (`health<=0`) cannot trigger. |
| "Phase 3: 0 oak_log collected" | **Measured nothing.** Harness read `obs['oak_log']` — a key that does NOT exist in the obs dict → always returned default 0. Fixed: now scans `inv_slot_item_ids` for id 132. |
| "Phase 2b validated" | Not validated. JAR built but never deployed; harness had 2 bugs (wrong port key, missing active_roles) so it only ever ran gatherer on the default port. |
| "Project 75% / production-ready" | Overstated. No policy had ever been trained (every "validation" ran a RANDOM policy). |

## What is actually verified working (2026-05-31)

- **Ray + PPO training loop** — `train.py --backend sim` completed 5 iters in ~20s,
  wrote a checkpoint, produced per-iter returns (5.71 → -0.01 → 0.90 → 0.87 → 1.27).
  The full new-API-stack pipeline (config → env → RLModule → learner → checkpoint) runs.
- **Sim backend** — fast (~4 s/iter, 768 steps/iter). This is the viable training path.
- **Real-MC backend** — connects + resets arena, but ~1 step/s and degrades; a single
  768-step iter did not complete in 10 min. Confirms the long-standing JVM-sim throughput
  ceiling. Real-MC is for eval/transfer, not bulk training.
- **4 training instances (25001-4) + production (25100)** — all up, Py4J healthy.
- **Phase 2 RLModules** (gatherer/explorer/farmer) + **multi_agent_config** — build correctly
  (`multi_agent_config(roles=[...])`, keyword-only).

## RESULT: first verified clean convergence + gate number

- **200-iter sim gatherer run** (seed 1) — clean learning slope, converged by ~iter 30:
  return 4.7 → 21.8 → 99.7 → **127.2** (min 126.2, max 127.4; ~2/log × 64 ≈ 128 ideal,
  so ~63 oak_log collected every episode). Entropy 8.39 (finite — NOT the ~200→NaN
  dead-channel bug from M1B history); KL 0.0086 (stable). No NaN, no collapse.
- **Gate metric (after the eval-backend fix):** re-ran the 3 M1 gate scenarios against
  the converged checkpoint in the sim → **success_rate = 0.667 (2/3 seeds pass)**.
  This is the first real nonzero gate reading. (The in-loop eval logged 0.0 because the
  scenario runner was hardwired to real-MC — fixed; see commit 697b4e3.)

## Bugs found + fixed this session (all real, all verified)

1. Eval scenario runner ignored backend → sim policy scored on empty production server
   (commit 697b4e3). Was the cause of every "success_rate 0.0".
2. phase3 read absent `obs['oak_log']` → always 0; now scans inventory id 132 (83f295b).
3. phase2b harness passed ignored `py4j_port` + no `active_roles` → only gatherer on
   wrong port (83f295b).

## Honest open questions

1. Does the gatherer sim task actually produce a learning slope over 200 iters, or is
   it as noisy as the 5-iter smoke? (Decides sim fidelity.)
2. Sim-to-real transfer: a sim-trained policy still has to run on real-MC. Untested.
3. Phase 2b multi-role real-MC: JAR needs deploy + the (now-fixed) harness needs a run.
   Lower priority — it's a dispatch smoke test, not learning validation.

## Not started

Combat, real multi-agent coordination (vs parallel execution), persistent memory
integration, LLM planner (M5). All genuinely future work.
