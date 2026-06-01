# Real-MC HARVEST non-determinism — source investigation + measurement protocol (2026-06-01)

The #1 P0 fidelity item (HANDOFF.md S3): back-to-back HARVEST dispatches / repeated gate
runs on the SAME seed collect different oak counts (46–64 swings; memory `ai-utopia-fast-sim-plan`
recorded 1/6/0 across identical dispatches), making real-MC eval unmeasurable at n=1.

This is a **source-only** analysis (no JDK/live server in the cloud container, so nothing
here is runtime-verified). Its value is narrowing the search and giving a localization
protocol — NOT a confirmed fix. Files: `bridge/skill/HarvestSkill.java`,
`bridge/MotorBridge.java`.

## What is RULED OUT (deterministic from source — do not chase these)

A prior automated scan flagged float precision and HashMap order; on a **single machine,
same JVM build**, those are NOT run-to-run variance sources. Concretely:

- **`scanShell` block selection is deterministic** (`HarvestSkill.java:259-281`). The scan
  is fixed-order nested `dx,dy,dz` loops; the winner uses strict `d < bestDist`, so ties
  break by *scan order* (first-found wins), not by chance. `Math.sqrt` is IEEE-754
  correctly-rounded and bit-identical across runs on one platform. Given a fixed agent
  origin + fixed world, `findNearest` returns the SAME block every time.
- **Break timing is a plain counter** (`BREAK_TICKS_PER_LOG`, `breakProgress`,
  `HarvestSkill.java:215`) — no randomness.
- **Drops bypass the item-entity pickup race** (`getDroppedStacks` → `offerOrDrop` directly
  into inventory, `HarvestSkill.java:222-233`). oak_log's loot table is deterministic
  (1 log/break); even though `getDroppedStacks` consumes `world.getRandom()`, the oak COUNT
  doesn't depend on RNG state.
- **`HashMap<String,…>` iteration** in MotorBridge (`active`, `dispatched`) is NOT randomized
  across runs by default in HotSpot (String.hashCode + fixed insertion order ⇒ stable order),
  and the `active` map holds ≤1 entry for a single gatherer anyway. Not the cause.
- **Log placement is seeded** (`WorldOps.resetEpisode` seeds `epRand`) ⇒ arena is identical
  per seed.

So the variance does **not** originate in the skill's decision logic.

## Where it LIKELY enters (ranked, hedged — needs runtime confirmation)

The skill ticks exactly once per **server** tick (`MotorBridge.onServerTick`, registered on
`END_SERVER_TICK`, line 72/169-203). Python waits via `advanceTickAwaitEvents(timeoutMs)`
(line 227), which blocks up to a **wall-clock** `timeoutMs` for a completion. The coupling
between *wall-clock* and *server ticks executed* is the prime suspect:

1. **Server-tick count per dispatch varies under tick-warp + GC jitter (MOST LIKELY).**
   Training/eval runs the server above 20 TPS ("tick warp"); CLAUDE.md notes G1GC on
   training instances and that >~60 TPS crashes the server. If `timeoutMs` (or the env-step
   tick budget) is wall-clock-bounded while real TPS fluctuates (GC pauses, scheduler
   jitter), a dispatch that needs K server ticks to chain 64 logs (each log = 15 break
   ticks + walk ticks ⇒ hundreds of ticks) can be cut off at a *different* tick on each run.
   With the `STALL_TICK_BUDGET=20` watchdog and the documented tail-reach stall (seed-3
   55/64), a few logs at the end are exactly where a cutoff lands ⇒ 58 vs 64 vs 46. This
   matches the *shape* of the observed swings (high counts that fall short by a handful).

2. **Carpet fake-player respawn / physics float-state under tick-warp.** `agent.move`
   (`HarvestSkill.java:191`) runs vanilla physics; deterministic GIVEN identical start pos +
   tick cadence. If the fake player's post-`resetEpisode` spawn pos carries sub-block
   variance, or entity tick-ordering vs the END_SERVER_TICK callback differs under warp, the
   walk trajectory (and thus which logs are in REACH when the cutoff lands) drifts.

3. **World-state interference between episodes.** `world.getRandom()` and random block ticks
   are not reset per episode; if the arena isn't fully isolated/forceloaded, ambient
   simulation could perturb timing. Lower likelihood on a peaceful flat arena.

## Localization protocol (run on the live server — this is the actual next step)

Instrument one dispatch and repeat the SAME gate seed N≥10× on a FRESH instance, logging:
- agent **start position** at `HarvestSkill.start` (x,y,z to full float),
- **total server ticks consumed** by the dispatch (count `tick()` calls until terminal),
- **per-log break tick index** and final `brokenCount`,
- the terminal `SkillResult` + `failureReason`.

Then read off the cause:
- start pos **varies** across repeats → respawn/reset non-determinism (cause #2).
- start pos fixed but **ticks-consumed varies** → wall-clock/tick-warp sync (cause #1):
  fix by making the step boundary **tick-count-bounded, not wall-clock-bounded**, or run eval
  at native 20 TPS (no warp) so cadence is stable.
- both fixed but **count varies** → world-state/RNG interference (cause #3).

## Interim mitigation (already shipped this session)

`scripts/transfer_eval_bc.py` now runs each scenario N times and gates on a **success rate**
(`summarize`, `--repeats`/`--pass-threshold`/`--warmup`), so the gate is usable for pass/fail
decisions *despite* this non-determinism while the root cause is localized and fixed. Fixing
the root cause (cause #1 most likely) is what ultimately restores clean, low-N real eval.
