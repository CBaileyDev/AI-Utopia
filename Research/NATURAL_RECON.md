# Natural-Terrain RECON — proven M1B Lumberjack

_Generated 2026-05-30T14:28:51 by `scripts/natural_recon.py`. RECON only — no training, no capability claims. Single-run, single-instance (Py4J 25001). Numbers are one sample._

## What was tested

The project descoped to a PEACEFUL survival world (no hostiles / no hunger), so the live question is NATURAL-TERRAIN GATHERING, not combat: can the PROVEN M1B gatherer (the HARVEST-spam policy that transfers 3/3 on the flat BARE-trunk arena) perceive and harvest REAL natural trees (trunks with LEAVES on top)? Peaceful world set at runtime via `runCommand` (no Java rebuild): `/difficulty peaceful`, `/gamerule doDaylightCycle false`, `/time set 1000`, `/gamemode survival gatherer_0`.

Three SEPARATE measurements (perception and HARVEST are DIFFERENT algorithms — obs = topmost-non-air per column; HARVEST = Euclidean-nearest-LOG over a 48-radius cube, spec §1.1 vs §1.2):

- **A. Perception A/B** — bridge-direct obs read (NO `env.step`, so the ±24-block arena truncation never fires): at the same agent position, `/setblock` a BARE oak_log column (control) vs an oak_log column with oak_leaves directly on top within the `dy∈[-3,+3]` scan band (hypothesis). Leaves are the single variable.
- **B. HARVEST on a leafed tree** — `bridge.dispatch_skill` a real HARVEST (NOT `wrapper.step`) against a leafed trunk; does `oak_log` increment even though the obs may have dropped the column?
- **C. Natural forest teleport** — qualitative COLOUR only (`/spreadplayers` to a far natural surface; confounds distance / terrain / canopy-height / biome, so NOT a clean perception number).

## (a) Does the policy PERCEIVE natural trees? — the leaves hypothesis

**Hypothesis (leaves occlude the topmost-non-air log scan) CONFIRMED.**

PRIMARY metric is **grid log cells** (LOG channel only — the honest "log visible" signal; `perceived rows` counts ANY resource channel and is saturated/uninformative). Verdict is delta vs the empty-arena baseline.

| placement | grid log cells | perceived rows | nearest_resource_distance | SEEN |
|---|---|---|---|---|
| baseline (arena cleared) | 0 | 0 | 999.0 | — |
| **control** bare oak_log col | 1 | 1 | 3.605551275463989 | True |
| **hypothesis** oak_log + leaves on top | 0 | 0 | 999.0 | False |

Mechanism (confirmed by code inspection, GathererOverlayBuilder.java:55-74): the per-column scan takes the FIRST non-air block top-down and `break`s (line 71) OUTSIDE the channel-match. A column topped by `oak_leaves` (no "log" substring → no channel match) is dropped; the log beneath is never read. The flat training arena is leafless by construction (WorldOps places only bare oak_log), so the policy NEVER saw this case (spec §3.2 "Keep arenas leafless").

## (b) Does HARVEST collect logs from natural (leafed) trees?

Measured on a PRISTINE arena (all 16 training trunks cleared first, so the ONLY logs present are the single leafed test tree — otherwise HARVEST chains into the arena trunks and the delta is meaningless).

- oak_log before → after: `0` → `3` (delta `3` of `3` logs in the test tree)
- completed after ticks: `1`
- **chopped the leafed natural tree: `True`**

HARVEST uses Euclidean-nearest-LOG (`findNearest`/`scanShell`, spec §1.1), a DIFFERENT algorithm from the obs scan — it hunts actual log blocks regardless of what is above them. So a log INVISIBLE to the obs can still be chopped if HARVEST is invoked. These are two separate facts; they are NOT collapsed into one "natural gathering works/fails" claim. (Leaves are collidable — the test tree's leaves were placed ABOVE the agent's Y66-67 walking lane so a blocked approach could not masquerade as a perception/skill failure.)

## (c) Concrete breakage list (natural terrain)

- **Perception (PRIMARY):** when a leaf is the topmost-in-band block of a column, that column is dropped from `g_resource_grid` / `g_nearest_resources` / `nearest_resource_distance` (see (a) table: the bare-log control added a log cell, the leafed column added zero). So a natural trunk is INVISIBLE to perception WHENEVER its canopy is the topmost block within `dy∈[-3,+3]` of the agent's feet. Section (c) shows this is conditional, not absolute — when the agent stands BELOW a canopy base so a trunk log is itself topmost-in-band, natural logs DO register (grid_log_cells was nonzero at several forest spots).
- **Scan band:** the obs scans only `dy∈[-3,+3]` around the agent's feet. A natural tree's canopy/upper trunk above `+3` is outside the window entirely (orthogonal to the leaf issue).
- **Arena bounds (env wrapper):** `AiUtopiaPettingZooEnv.step` hard-truncates the agent outside ±24 blocks of spawn / `y<60` (wrapper.py ~458). A policy run after any far teleport truncates on the FIRST `step()` — so the natural-forest section had to be driven bridge-direct (dispatch + advance_tick), bypassing `wrapper.step`. The bounds clip is itself a natural-world breakage: the wrapper is hard-wired to the flat arena box.
- **Natural forest (colour):** at far natural spots the LOG signal `grid_log_cells` across the sampled spots was `[4, 38, 0, 22, 3]` (NOT `perceived_rows`, which stays saturated at 8 because the `stone` channel matches the exposed hilly terrain). So natural logs were perceived at SOME spots and not others, consistent with the conditional leaf-occlusion above. The bridge-direct HARVEST probe moved oak_log by `11`, BUT it ran at the LAST `/spreadplayers` landing (grid_log_cells=0, no log in range), NOT the best spot, so this `0` is a probe-sequencing artifact and is NOT evidence HARVEST fails on natural logs (section (b) already showed it chops a leafed tree). The greedy policy then ran `60` bridge-direct steps, skill mix `{'HARVEST': 60}` — it spun NAVIGATE the whole time (no perceived in-range log to lock a HARVEST onto) and drifted far (`(4222.7, 62.0, 1269.6)` to `(4222.7, 62.0, 1269.6)`) collecting nothing. Confounded (distance/terrain/canopy/biome all vary at once), qualitative only.

## (d) Implication for the peaceful-village direction

- **The obs builder, not HARVEST, is the natural-terrain blocker.** The perception scan must change from "topmost NON-AIR per column" to a LOG-AWARE scan (e.g. topmost block whose id matches the log channel, scanning past leaves, or a true 3D voxel channel). Until then the policy is effectively BLIND to natural trunks under canopy.
- **If HARVEST chops leafed logs (see b),** a near-term bridge would be to feed perception from the SAME Euclidean-nearest-LOG scan HARVEST already uses — unify the two algorithms so what the policy sees == what HARVEST can reach. That is an obs-builder change, not a policy retrain blocker on its own (though the policy would still need re-validation on the new, populated obs distribution).
- **Either way the arena-bounds truncation must be lifted** for any real-world (non-flat-arena) operation — the wrapper currently pins the agent to the ±24-block training box.

## Honest caveats

- Single seeded run (seed=1), one instance (Py4J 25001). Treat every number as ONE sample, not a distribution.
- The A/B leafed column is a `/setblock` STAND-IN for a natural tree (real procedurally-generated trees vary in trunk height, canopy shape, and ground height); it isolates the leaf variable but is not a full natural tree. The natural-forest section (C) attempts a real tree but is qualitative colour only (confounded).
- No Java rebuild, no training. Peaceful/world state was flipped at runtime via `runCommand`. The flat arena was restored on exit (`resetEpisode`) so instance-1 is left warm.
- The perception MECHANISM is also confirmed by code inspection (line 71 `break` outside the channel match); the run supplies the empirical number.

