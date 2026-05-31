# seed_1 Gate Hole — Verified Root Cause (parity-checked)

## Symptom
Converged sim gatherer (return 127, iter 200) scores gate 2/3:
seed_2=64, seed_3=64, **seed_1=0**. Persistent across checkpoints, not noise.

## Trace (greedy policy, per-step skill + oak)
- seed_2: **1 step**, skill=HARVEST(1) → oak=64, terminates. (Sim HARVEST collects all
  reachable logs in one env-step.)
- seed_1: **120+ steps**, skill=SEARCH(3) every step, never HARVEST/NAVIGATE, oak=0.

## Root cause — VERIFIED, and it is NOT a sim bug

HARVEST is mask-gated on `nearest_res_dist <= REACH_RADIUS_BLOCKS (4.5)`
(obs_adapter.py:235). `nearest_res_dist` is the distance to the **topmost log per
(dx,dz) column** within a ±3 dy window (gatherer_nearest_columns, obs_adapter.py:97-142).

seed_1 spawn at (64.5, 65, -47.5). Nearest trunk is the column at block-delta (dx=3, dz=2)
— a vertical stack of logs at dy=0,1,2,3 (y=65→68):
- BOTTOM log dy=0: dist = √(9+0+4) = **3.61b** → physically within 4.5b reach.
- TOPMOST log dy=3: dist = √(9+9+4) = **4.69b** → the column distance the mask uses.

So the mask sees 4.69 > 4.5 → **HARVEST masked**, even though the trunk's base is reachable
at 3.61b. seed_2/3 differ only by geometry: their nearest trunk's *topmost* log still lands
≤4.5b (3.16b), so HARVEST stays valid and the one-shot collect fires.

**Parity check (important):** the Java `GathererOverlayBuilder` does the SAME thing —
`for (dy=3; dy>=-3; dy--) … break  // only the topmost LOG per (dx,dz)` then
`nearestResDist = sqrt(nearby.get(0).distSq())` (GathererOverlayBuilder.java:65,79,146).
The sim faithfully reproduces real. **This is the real system's actual behavior, not a
sim artifact.** "Fixing" gatherer_nearest_columns to use the nearest (bottom) log instead
of the topmost would BREAK sim↔real parity unless the Java side is changed in lockstep.

## What the policy actually learned (the real limitation)
A degenerate **HARVEST-when-topmost-in-reach** strategy. When a trunk's top is ≤4.5b it
one-shots 64; when the trunk is 1 block farther (top >4.5b but base reachable) it should
NAVIGATE one step closer to unmask HARVEST — but it never learned that, it picks SEARCH
forever. The trunk sits at 3.6b, well inside the 16b perception window, so navigation IS
learnable on the current obs (this is NOT the "blind beyond 16b" limitation cited in
config.py:327). The policy simply took the shortcut because randomized training layouts
usually gift a topmost-in-reach trunk at spawn.

## Why this matters
- The "127 convergence" overstates competence: it is HARVEST-spam on a too-generous sim
  HARVEST (64-in-one-step) plus usually-in-reach spawns. Real gather competence (approach
  a trunk, then harvest) was never learned.
- seed_1=0 is a LEGITIMATE gate failure, correctly measured. Gate 0.667 stands.

## Fix direction (a training experiment, not a code fix)
Remove the in-reach shortcut during training so the policy must learn NAVIGATE→HARVEST:
either a curriculum/arena that never spawns a topmost-in-reach trunk, or reduce the
HARVEST one-shot generosity so a single HARVEST ≠ instant 64. Do NOT use `decision_core`
as the fix — it changes the obs (bearing cue), the arena (clusters, half=34), adds
distance shaping, and per config.py:144 demotes HARVEST to mine cobblestone, which is
incoherent with the oak_log gate predicate. Keep the task fixed; remove only the shortcut.

## DECISIVE evidence (skill logits, final checkpoint)
Both seeds produce IDENTICAL skill logits:
`HARVEST=8.98, SEARCH=-2.51, NOOP=-3.14, NAVIGATE<-3.14`.
- seed_2: HARVEST unmasked → argmax=HARVEST → the skill `_walk_into_reach`'s each log
  internally and collects all 64 in one dispatch. ✓
- seed_1: HARVEST masked → argmax over unmasked skills = **SEARCH** (next-highest), which
  does nothing → repeats 120 steps → oak=0. ✗

NAVIGATE's logit is BELOW SEARCH, so when HARVEST is masked the policy never chooses to
move — it has **near-zero learned NAVIGATE preference**. It is a pure HARVEST button-presser.

## Corrected mechanism (the skill walks; the mask is fine)
`_apply_harvest` (skills.py:194-211) loops `_nearest_alive_log` → `_walk_into_reach` →
break, up to `cap≈64` logs per dispatch. So HARVEST itself does the walking; the policy
needs only to (a) get close enough to unmask HARVEST, then (b) press it. On seed_1 the
trunk base is 3.61b away — ONE NAVIGATE unmasks HARVEST and the skill finishes the job.
The mask (topmost-column ≤4.5) is doing its job: "too far, move first." The defect is
purely that the policy learned to press HARVEST and SEARCH, never NAVIGATE — because
randomized training layouts almost always gift a topmost-in-reach trunk at spawn, so the
"masked → must navigate" state was effectively never in the training distribution.

## Status — DECISIVELY diagnosed (not a code bug; a training-distribution gap)
Env, mask, and skill are all correct and sim↔real faithful. The fix is a training change,
cleanly scoped and NON-confounded: ensure a meaningful fraction of training episodes spawn
with HARVEST masked (e.g. spawn the agent ≥~5b from the nearest trunk top, or bias layouts
so no trunk is topmost-in-reach at t=0). Then the policy must learn NAVIGATE→HARVEST.
Re-eval expecting NAVIGATE present in skill usage and gate 3/3. This does NOT require
touching obs/mask/skill code or Java (parity preserved). Estimated: one sim training run
(~15 min) + re-eval, next session.
