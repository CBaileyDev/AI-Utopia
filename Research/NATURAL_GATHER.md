# Natural-Forest Gathering RECON — proven M1B Lumberjack (post perception-fix)

_Generated 2026-05-30T17:27:00 by `scripts/natural_gather.py`. RECON only — no training, NO capability claim. Single seeded run (seed=1), one instance (Py4J 25001). Every number is ONE sample, not a distribution._

## What was tested

The leaves-occlusion perception bug is FIXED in the deployed jar (GathererOverlayBuilder now scans top-down for the topmost LOG per column within `dy∈[-3,+3]`, skipping leaves/non-logs instead of break-on-first-non-air). This run is the REAL capability test of that fix: does the PROVEN M1B greedy policy (3/3 on the flat bare-trunk arena, sim-control 64/64) PERCEIVE and HARVEST oak_log from FULL PROCEDURAL trees (trunk + leaf canopy, uneven terrain) when teleported into a natural forest?

Run via `wrapper.step` (the production step path), made possible by the flag-gated arena-bounds box (`arena_bounds_check=False`) — the prior recon had to drive bridge-direct because a forest teleport truncated on the first `step()`. Peaceful world set at runtime via `runCommand` (`/difficulty peaceful`, `doDaylightCycle false`, `/time set 1000`, `/gamemode survival`). Forest located by using the post-fix obs itself as the detector (`/spreadplayers` to candidate far surfaces, keep the spot where procedural logs register).

## Perception-fix confirmation (pre-run)

Before the natural run, the prior recon's A/B probe was re-run against the live jar: a BARE oak_log column (control) AND an oak_log column with `oak_leaves` on top (the case the OLD builder dropped) BOTH now register a log cell (control Δ+1, leafed Δ+1 vs cleared baseline 0). The deployed jar fixes the occlusion bug; perception on the leafless training distribution is unchanged (control still +1).

## (a) Does the policy PERCEIVE natural (procedural) trees?

Forest located by obs-as-detector. Per-spot natural-log perception (`grid_log_cells` = LOG-channel (x,z) cells flagged; `n_perceived` = populated `g_nearest_resources` rows):

| probe center | landed pos | grid_log_cells | n_perceived | nearest_dist |
|---|---|---|---|---|
| (1500, 1500) | (1476.5, 67.0, 1527.5) | 23 | 8 | 4.36 |
| (2500, -1800) | (2542.5, 63.0, -1766.5) | 0 | 0 | 999.00 |
| (-2200, 2200) | (-2245.5, 88.0, 2214.5) | 3 | 3 | 2.45 |
| (3000, 3000) | (2945.5, 64.0, 2990.5) | 7 | 4 | 11.58 |
| (-1500, -2500) | (2945.5, 64.0, 2990.5) | 7 | 4 | 11.58 |
| (4000, -500) | (2945.5, 64.0, 2990.5) | 7 | 4 | 11.58 |

**Best spot:** center `(1500, 1500)`, pos `(1476.5, 67.0, 1527.5)`, `grid_log_cells=23`, `n_perceived=8`, `nearest_dist=4.358898943540674`.

The policy run starts from a DETERMINISTIC `/tp` back to that exact measured landing (not a fresh `/spreadplayers` roll). Verify-before-measure at the start spot: pos `(1476.5, 67.0, 1527.5)`, health `20.0`, grid_log_cells `23` (probe measured `23`). A pos=None or health<=0 here would mean the tp/chunk-reload failed (artifact), not a policy result.

The policy DID perceive procedural logs at the best spot (`23` log cells in band) — the post-fix obs surfaces natural trunks under canopy, which the old builder could not.

## (b) Does HARVEST collect oak_log from procedural trees? — accumulation

- oak_log start → final: `0` → `14` (delta `+14`, max seen `14`)
- steps run: `400` (NAT_STEPS=400)
- oak_log curve sample `(step, count)`: `[(1, 0), (101, 14), (201, 14), (301, 14), (400, 14)]`
- skill histogram: `{'HARVEST': 396, 'NAVIGATE': 4}`
- position first → last: `(1476.5, 67.0, 1527.5)` → `(1475.7, 65.0, 1523.5)`
- terminated/truncated count: `0`/`0`
- wall time: `229.2`s

The proven policy DID accumulate `14` oak_log from procedural trees in this single run. This is the unambiguous metric: natural logs entered the bag via the production step path. It is NOT a capability claim (one seed, one spot) — it is what the proven policy DID on this natural terrain.

## (c) Failure modes (natural terrain)

- **dy∈[-3,+3] window vs tall oaks:** a 6-log bare trunk on the cleared arena registers only `4/6` logs in the band (probe section). Natural oaks are 4-7 logs tall; the upper trunk + canopy above feet+3 is OUTSIDE the obs window entirely — orthogonal to the (now-fixed) leaf occlusion. So even with leaves transparent, a tall trunk is only partially visible, and the topmost reachable target is the in-band log.
- **Terrain / pathing:** the agent spawns on uneven natural ground (vs the flat arena's single y-plane). The skill histogram + position drift (`(1476.5, 67.0, 1527.5)` → `(1475.7, 65.0, 1523.5)`) show whether the policy walked toward perceived logs or wandered. The proven policy was trained to NAVIGATE-then-HARVEST a fixed ring at a known y; natural terrain is OOD on ground height, trunk height, and tree density.
- **Trees out of reach / scan radius:** `nearest_resource_distance` at the best spot was `4.358898943540674`. If no log falls within HARVEST's reach after navigation, HARVEST degrades to NAVIGATE-spin (the prior recon's observed natural-terrain behavior).
- **OOD distribution:** this is sim→real→natural — the checkpoint trained entirely in the headless sim on a flat 8-log ring. Natural forest varies ground height, trunk height (>dy+3), canopy, and density all at once. Read every number as OOD behavior of the proven policy, not a measure of the natural-gather ceiling.

## Honest caveats

- Single seeded run (seed=1), one instance (Py4J 25001), one set of probe coords. Treat every number as ONE sample. A different `/spreadplayers` landing would give a different spot and likely a different result.
- `/spreadplayers` surface-snaps but does not target forests; the obs detector picks the best of the probed spots, not the best forest in the world. Procedural-tree placement is a confound (density / proximity / trunk height vary per spot).
- No Java rebuild, no training. World state flipped at runtime via `runCommand`. The flat arena was restored on exit (`resetEpisode`) so instance-1 is left warm. The arena-bounds box was disabled via `env_config` (default unchanged; flag-gated).

