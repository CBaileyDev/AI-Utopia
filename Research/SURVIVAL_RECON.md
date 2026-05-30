# Survival-Pressure RECON — proven M1B Lumberjack

_Generated 2026-05-30T12:36:20 by `scripts/survival_recon.py`. RECON only — no training, no capability claims. Observations are quantified and reported as-is._

## What was tested

The PROVEN M1B gatherer_policy checkpoint (HARVEST-spam policy that transfers 3/3 on the peaceful flat arena) was loaded greedily and faced real Minecraft survival pressure flipped on at RUNTIME via `Py4JEntryPoint.runCommand` (no Java rebuild):

- `/difficulty hard`
- `/gamemode survival gatherer_0`
- `/gamerule doMobSpawning true`
- `/gamerule doDaylightCycle false`
- `/time set 18000`

Plus a deterministic pre-flight that `/summon`s zombies adjacent to the fake player to confirm the threat actually lands before committing to a long run (natural night spawns are stochastic; on `/difficulty normal` starvation caps at 1 HP and never kills, so a kill requires mob damage).

The policy is OUT-OF-DISTRIBUTION on every survival signal: it was trained at constant `health=20`, `hunger=20`, empty `g_hostiles_nearby`.

## Headline result

- **survived**: `False`
- **cause**: `MOB_ATTACK`
- **steps to death**: `1`
- **steps run**: `1`
- **wall time**: `30.0s`  (capped: `None`)
- **oak_log at end**: `0`
- **hostiles ever populated `g_hostiles_nearby`**: `True`
- **reacted to hostiles (dominant-skill shift)**: `None`

## Pre-flight (threat-lands check)

```
baseline_h = 20.0
health_is_binary_20_or_0 = True
armed_threat = 1 zombie summoned ~4 blocks away
death_atomic_under_tickwarp = True
```

## Behavior under pressure

- skill histogram: `{'NAVIGATE': 1}`
- oak_log first->last: `0` -> `0`
- min health seen: `0.0`   min hunger seen: `0.0`

- reaction detail: hostiles never populated g_hostiles_nearby during the run, so reaction-to-hostiles cannot be measured

## Death / death-oracle methodology

A Carpet fake player that dies is REMOVED from the server, so its `player_name` disappears from `observationsAll()`. The robust death oracle used here is **raw-key presence in `bridge.observations_all()`** (absent key == dead), corroborated by the decoded `term` flag and the agent dropping out of `env.agents`. Decoded `health` is logged but NOT trusted as the primary death signal (it zero-fills to 0.0 on the missing-key step, which is itself only a derived signal).

## First 30 steps (trace)

| step | skill | health | hunger | hostiles | oak_log | pos | raw_alive |
|---|---|---|---|---|---|---|---|
| 1 | NAVIGATE | 0.0 | 0.0 | 0 | 0 | (0.0, 0.0, 0.0) | False |

## Honest caveats

- This is a single seeded run (seed=1) on one instance. Minecraft mob spawning/pathing has stochastic elements; treat counts as one sample, not a distribution.
- Pressure was flipped at runtime mid-world (no Java rebuild); the arena remains the flat M1B ring.
- `cause = MOB_ATTACK` is inferred from 4 hostiles being present in the PRE-step obs of the terminal step, not from a death-message parse.
- The `reacted to hostiles` measurement returned `None` because there was no alive-with-hostiles decision step to measure on (death was atomic — see addendum).

---

## Analyst addendum — corrected findings (the recon value)

> An earlier version of this report concluded the fake player was "invulnerable,
> survival untestable without a rebuild." **That was wrong** and is retracted. The
> player was simply spawned in creative/invulnerable gamemode; `/gamemode survival`
> is a one-command runtime flip (no rebuild) and makes survival fully testable.
> Findings below were each confirmed by direct probes against the live server
> (port 25001, MC 1.21.1).

### Finding 1 — Survival pressure is a RUNTIME flip; the proven policy dies on step 1 to the mob (headline)

With `/gamemode survival gatherer_0` (plus `/difficulty hard`, midnight, doMobSpawning)
applied after reset and one zombie summoned ~4 blocks away, the proven HARVEST-spam
policy was run greedily. It **died on the very first env-step**: pre-step obs carried
4 hostiles in `g_hostiles_nearby`, the policy chose `NAVIGATE` (its normal first move
toward the oak ring), and within that single tick-warped step (~400 ticks ≈ 20 game-
seconds) the mobs killed it (`raw_alive=False`, `term=True`, `steps_to_death=1`,
`oak_log=0`). The `term AND trunc` both-true on the terminal step is a side effect of
removal (the zero-filled position reads as out-of-bounds); `raw_alive=False` is the
authoritative death signal.

**The mob is genuinely the cause** — isolated by control (see Finding 2b): the SAME seed
and SAME first NAVIGATE in survival mode with NO mob survived 60 steps at full health.
The only thing that changed in the lethal run was the threat.

The Carpet fake player spawns via `/player <name> spawn` (`WorldOps.carpetSpawn`),
inheriting the server's default (creative-style, invulnerable) context — which is why
mobs and `/damage` are no-ops until `/gamemode survival` is issued. `reset_episode`
re-spawns the player, so the survival flip must be re-applied every episode (the script
does this).

### Finding 2 — Damageability is gamemode-gated, established by probe matrix

| stimulus | gamemode | decoded health | player removed? |
|---|---|---|---|
| `/damage 10`/`19`/`100`, 3–5 zombies on player | default (creative/invuln) | 20.0 → 20.0 | no |
| `/damage 7` then step | **survival** | 20.0 → 0.0 | **yes** |
| `/damage 1` then step | **survival** | 20.0 → 0.0 | **yes** |
| `/kill` | any | 20.0 → 0.0 | **yes** |

Note the `/damage <n>` rows: even `/damage 1` instantly removed the player, which is
anomalous for a 20-HP entity. This appears to be a `/damage`-COMMAND artifact on Carpet
fake players (the command over-applies / instant-kills), **not** evidence of low HP —
`getHealth()` reports the full 20.0 (see Finding 3). The `/damage` quirk was flagged,
not root-caused; it is irrelevant to mob-driven death, which works normally (Findings 1,
2b).

### Finding 2b — POSITIVE control: survival + NO threat → policy runs normally, 60 steps, full health

`/gamemode survival` + `/difficulty hard` + `doMobSpawning false` + daytime, NO mob, then
the proven greedy policy for 60 steps: it **survived all 60 steps at health=20.0**
(4 NAVIGATE + 56 HARVEST, no death, no truncation). This is the load-bearing control: it
(a) proves survival mode by itself does NOT kill the player (so the armed run's death is
the mob, not survival/fragility), and (b) shows baseline survival works. It still
collected 0 oak_log — consistent with the harness artifact in Finding 6.

### Finding 3 — Health was only ever SAMPLED as 20.0 or 0.0 (a sampling artifact, not low HP)

Across every probe, the only decoded health values observed were `20.0` and `0.0` —
never an intermediate. This is a **sampling artifact**, not a property of the player: the
obs reports `agent.getHealth()` with max 20 (`CoreObsBuilder.vec1(agent.getHealth())`),
and the no-threat control held a steady 20.0 for 60 steps, so the player genuinely has
20 HP. The reasons no intermediate value was ever sampled are (a) the `/damage`-command
quirk (one-shots regardless of `<n>`) and (b) mob death resolving inside a single
~400-tick warped env-step. We do NOT claim health is inherently binary or that the player
has ~1 HP.

### Finding 4 — In the ARMED config, mob death is atomic from the policy's view; reaction was not measurable here

Under tick-warp, one env-step spans many ticks; combat resolves *between* obs samples.
In the armed run the policy received obs=alive → picked `NAVIGATE` → the ~400-tick step
ran → the next obs was dead/zero-filled. So for THIS config (4 mobs, adjacent, hard,
tick-warp) the policy never received a mid-combat observation and `reacted_to_hostiles`
is **not measurable** — we cannot distinguish "ignored the signal" (OOD) from "never saw
a mid-combat signal." We did NOT establish that a gentler config (distant mob, lower warp,
fewer mobs) could not yield alive-with-hostiles decision steps; that was not found here
but is not ruled out. So this is a per-config observation, not a structural impossibility
claim.

### Finding 5 — `g_hostiles_nearby` works with real mobs (positive)

Summoned and naturally-spawned zombies populated `g_hostiles_nearby` (up to 4 nonzero
rows tracking real `HostileEntity` instances in the 16-block scan box). The obs channel
that was only ever zero in training carries real mob data correctly.

### Finding 6 — UNPREDICTED: on this greedy harness the proven policy collects ZERO oak_log even with NO pressure

A clean control (NO summon, NO survival flip, proper arena spawn at `(64.5, 65, -47.5)`,
fresh `reset_episode`) reproduced a degenerate trajectory: 1 `NAVIGATE` to
`(65.3, 65, -62.5)` (~15 blocks off the oak ring) then 199 `HARVEST` collecting **0**
oak_log. This is **independent of survival pressure** — it happens identically in the
peaceful control — so it is a harness artifact of THIS rollout path
(load checkpoint → `reset()` → greedy single-step decode with a freshly initialised
LSTM), NOT something caused by mobs/OOD obs. It does not contradict the documented
"transfers 3/3" result (produced by `scripts/transfer_eval.py`'s `run_instrumented`
loop with the same checkpoint and same `_greedy_decode`); it shows that result is
**harness-sensitive**. The discrepancy between the two greedy loops was not root-caused
here and is flagged for follow-up.

### Death oracle is sound

`/kill` drove decoded health to 0.0, removed the player_name from `observationsAll()`,
and dropped the agent from `env.agents` — all three signals agreed — so a real death is
reliably detected via the raw-key-absence oracle.

### Finding 7 — n=1 datapoint: the policy's first action was IDENTICAL with and without the threat

At step 1 the greedy policy chose `NAVIGATE` in BOTH the armed run (obs carried 4
hostiles + night) and the no-mob control (no hostiles, daytime) — the same immediate
action despite the survival signal being present in one and absent in the other. This is
a cleaner "did not alter its immediate action under the survival signal" datapoint than
the dominant-skill histogram (which returned None). Heavy caveats: n=1, greedy decode,
and the two obs differed in more than just hostiles (time-of-day too), so this is
suggestive, not a measured reaction result.

### What this recon did NOT establish (no over-claims)

- It did **not** observe behavior-under-sustained-attack, pathing-among-mobs, or
  HARVEST-under-attack in the ARMED config — death was atomic (step 1) there, so none of
  these were reachable. A gentler config might reach them; not tested.
- It did **not** root-cause the `/damage`-command one-shot quirk (irrelevant to mob death).
- It did **not** observe starvation (hunger never decremented off 20.0; in the 60-step
  no-threat control hunger stayed 20.0, and the armed run died too fast to starve).
- It did **not** resolve the oak_log=0 harness discrepancy with `transfer_eval` (Finding 6).
- The oak_log=0 harness discrepancy with `transfer_eval` is reported as an open
  anomaly, not resolved.

