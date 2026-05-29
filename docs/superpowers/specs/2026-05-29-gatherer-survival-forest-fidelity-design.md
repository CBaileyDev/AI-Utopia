# Gatherer Survival-Forest Fidelity — Design Spec

**Date:** 2026-05-29
**Status:** approved (design); spike completed; ready for implementation plan
**Author:** Claude (autonomous, advisor-reviewed)

## Goal

Raise the Lumberjack (gatherer) from the M1B *toy* arena — flat 8×8 oak-log grid,
instant ("creative") breaking — to **genuine survival-forest fidelity**: survival
block-break *timing* (multi-tick mining with an axe) and **real vertical oak
trees** scattered in a clearing. Validate every step with the proven **sim→real
transfer loop** (the same loop that just passed the M1B gate 3/3).

## Why this milestone (vs more roles / MARL)

M1B proved the **methodology** but on a task so simplified it never exercised
survival mechanics. Every downstream goal — each new role, MARL, the M6 deploy —
rests on the sim→real loop holding under **real survival conditions**. The highest-
leverage de-risking step is to prove that on *one* role before building more on top.
The user (the stakeholder) independently flagged both gaps ("he breaks them right
away — this will be survival!" and "wood should be placed like trees, not all on the
same level"). This milestone closes them.

## Honest scope — what the gate proves (read this first)

In the current architecture the **skill does all the spatial work**: `HarvestSkill`
internally chains (re-scan → walk to next-nearest log → break), so the policy emits
only `skill_type=HARVEST` + `target_class=oak_log` and makes **no spatial / sequencing
decision**. Therefore:

- A passing transfer gate here validates **skill + sim parity** (the sim models the
  survival skill faithfully enough to transfer), **not** policy *learning* of
  navigation/harvest strategy.
- That is the correct and sufficient bar for a **fidelity** milestone — we are making
  the *world* realistic, not yet making the *policy* smarter.
- **M2+ architectural note (out of scope, but flagged):** for agents to make
  *meaningful* decisions, skills must become more primitive (or the policy must own
  navigation/target-selection). When that happens, a richer 3D observation becomes
  load-bearing. **Until something consumes it, we do NOT enrich the obs** (the obs
  grid + `g_nearest_resources` stay as-is; trees simply project trunk logs into the
  existing representation).

## De-risking spike (DONE — `scripts/n21_breaktiming_determinism.py`)

The scariest unknown was whether *timed, multi-tick* breaking is deterministic enough
on real MC to match in the sim (project memory flagged real HARVEST as historically
non-deterministic). Measured empirically with a minimal timed-break (20 ticks/log) on
instance-1, repeating the same `reset(seed)` + HARVEST sequence:

- seed 1 (×3): `[0, 17, 32, 47, 61, 64]` — **byte-identical every run**
- seeds 2/3 (×2 each): identical within each seed; all clear 64/64
- ~28 s wall per full clear (vs ~8 s instant — realistic survival slowdown)

**Verdict: survival break-timing is DETERMINISTIC on real MC.** Root cause it's safe:
the N16c fix already inserts drops **directly into inventory** (no item-drop/pickup
race), and a tick counter is deterministic. **No mitigation needed.** The spike also
quantified the budget: **~17 logs per 400-tick dispatch** at 20 ticks/log, so clearing
a field is a deterministic ~5-dispatch sequence.

## Architecture & increments

One coherent spec, built as **two independently-testable, transfer-validated
increments**. Order is **break-timing first** (smallest experiment that retires the
biggest risk; directly fixes the "instant break" complaint; sidesteps the reach trap).

### Increment 1 — Survival break-timing + axe (flat grid retained)

**What changes**

- **`HarvestSkill.java`**: break each log over `breakTicks` computed from
  **block hardness × tool speed**, instead of 1 tick. Equip the fake player with a
  **stone axe** (pre-given — crafting is explicitly deferred, see below). Use vanilla
  break-time semantics (vanilla oak-log break times): hardness `2.0` → bare-hand
  3.0 s ≈ **60 ticks**, wooden axe 1.5 s ≈ 30, **stone axe 0.75 s ≈ 15**, iron 0.5 s
  ≈ 10, diamond 0.4 s ≈ 8. Equip a **stone axe** ⇒ `BREAK_TICKS_PER_LOG ≈ 15`
  (compute from the actual equipped tool via the vanilla formula, rounded, min 1).
  (The spike used a placeholder 20; the real value is tool-derived.) The N16c
  direct-inventory-insert is
  **unchanged** (deterministic); we only gate it behind a `breakProgress` counter that
  resets on each new target. (The spike's fixed-20 is replaced by the tool-based value.)
- **`sim/skills.py`**: model the same — each log consumes `BREAK_TICKS_PER_LOG` (the
  tool-based value, single source of truth shared conceptually with Java) of the
  per-dispatch tick budget, in addition to walk ticks. `_apply_harvest` already loops
  per-log; add the break-tick cost so the sim reproduces ~17 logs / 400-tick dispatch.
- **Tick budget (advisor flag):** keep `timeout_ticks=400` and accept **multi-dispatch
  clearing** (deterministic, ~5 dispatches → 64, validated by the spike) rather than
  inflating the budget. Make `BREAK_TICKS_PER_LOG` a named constant on both sides so
  the budget math is explicit and tunable.
- **Obs:** no change. **Reward:** `oak_log` stays the unit (1.0); no new shaping.

**Validation:** retrain in sim (fast); confirm the sim rollout clears via multi-dispatch
HARVEST; rebuild + redeploy the Java jar; run `scripts/transfer_eval.py` → expect 3/3
(now at survival speed, ~28 s/scenario). The spike already shows real-MC determinism.

### Increment 2 — Real trees (capped height, non-decaying leaves)

**What changes**

- **`WorldOps.java` + `sim/world.py`**: replace the flat Y=66 grid with **scattered
  vertical oak trunks**. Each trunk is a vertical stack of oak_log of height
  `h ∈ [3, 4]`, placed at jittered (x,z) ground positions across the arena, with the
  **byte-faithful `java.util.Random` layout parity preserved** (`_JavaRandom` port).
  **Cap trunk height ≤ 4** so every log is within the 4.5 reach from the ground — the
  sim can **never out-reach** the ground-standing fake player (this is the seed-3
  flat-y optimism bug re-dressed; capping height eliminates the variable). Keep total
  oak_log count comparable to today (≈64, so ≈18 trees × 3–4 logs) so the gate target
  is unchanged. **Leaves:** place **static, non-decaying leaves** (the `persistent=true`
  leaf state, which never decays) for visual fidelity — leaves are not oak_log so they
  don't affect harvest; we **never enable random leaf decay** (it would inject
  non-determinism + item drops into a fidelity-sensitive sim). The sim's `world.py`
  does not model leaves at all (inert in obs/skill); only the Java arena places them,
  so sim↔real parity is unaffected.
- **`HarvestSkill` + `sim/skills.py`**: `findNearest`'s existing two-pass dy band
  (`dy ∈ [-2,+1]` then full) + chaining already handle a vertical trunk within reach;
  with height ≤ 4 and reach 4.5, **no climbing state machine** is needed. Verify the
  sim's `_nearest_alive_log` selects trunk logs identically (it already uses 3D
  Euclidean distance).
- **`GathererOverlayBuilder.java` + `sim/obs_adapter.py`**: trunk logs project to their
  `(x,z)` grid cell (a trunk shows as one occupied cell; `g_nearest_resources` reports
  the nearest individual logs with their true `dy`, which the format already carries).
  **No new obs channels.** **Regenerate the golden-trace fixture** for the tree arena.

**Deferred to later milestones (explicit non-goals):**

- **Crafting the axe** (wood→planks→sticks→axe): the agent is *given* a stone axe.
  Crafting is a shared multi-role capability deserving its own milestone.
- **Tall trees / climbing** (height > reach, bottom-up-chop-and-step-up): height is
  capped ≤ 4 this milestone. Climbing is a follow-on fidelity refinement.
- **Mobs / hunger / combat:** server stays `difficulty=peaceful`.
- **Leaf decay, multiple species, terrain/biomes.**

## Components touched

| Component | Increment | Change |
|---|---|---|
| `fabric_mod/.../skill/HarvestSkill.java` | 1, 2 | break-timing + axe (1); verify trunk reach (2) |
| `src/aiutopia/sim/skills.py` | 1, 2 | break-tick cost per log (1); verify trunk selection (2) |
| `fabric_mod/.../bridge/WorldOps.java` | 2 | scattered capped-height oak trees |
| `src/aiutopia/sim/world.py` | 2 | tree arena, `_JavaRandom`-parity |
| `fabric_mod/.../obs/GathererOverlayBuilder.java` | 2 | (verify trunk projection; no new channels) |
| `src/aiutopia/sim/obs_adapter.py` | 2 | (verify; regenerate golden trace) |
| equip-axe path (WorldOps bootstrap or skill) | 1 | give the fake player a stone axe |
| `scripts/transfer_eval.py` + eval scenarios | 1, 2 | survival-speed + tree-arena gate |

## Testing

- **Unit (pytest, sim):** break-tick cost (a cap=64 dispatch under a 400-tick budget
  collects ~17, not 64; the per-log tick math is exact); tree-arena layout parity
  (sim `world.py` placement matches `WorldOps` seed-for-seed); `_nearest_alive_log`
  selects the correct trunk log. Keep the existing sim suite green.
- **Golden-trace fidelity gate:** regenerate `tests/fixtures/` for the tree arena;
  sim obs must match real MC byte-for-byte on the deterministic prefix.
- **Sim→real transfer gate (per increment):** clear the arena within budget on 3 fixed
  seeds → 3/3, with determinism re-checked (`n21_breaktiming_determinism.py` pattern).
- **Determinism:** every change preserves the demonstrated real-MC determinism; re-run
  the probe after each increment.

## Success criteria

1. Increment 1: sim-trained policy clears the **flat grid at survival speed** → real-MC
   gate 3/3, deterministic.
2. Increment 2: sim-trained policy clears a **real-tree forest** (capped trunks) → real-MC
   gate 3/3, deterministic, golden-trace parity intact.
3. The sim→real fidelity loop is shown to survive survival mechanics — de-risking the
   multi-role roadmap.
