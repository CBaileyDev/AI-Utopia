# Survival-Pressure RECON — proven M1B Lumberjack

_Generated 2026-05-30T12:24:10 by `scripts/survival_recon.py`. RECON only — no training, no capability claims. Observations are quantified and reported as-is._

## What was tested

The PROVEN M1B gatherer_policy checkpoint (HARVEST-spam policy that transfers 3/3 on the peaceful flat arena) was loaded greedily and faced real Minecraft survival pressure flipped on at RUNTIME via `Py4JEntryPoint.runCommand` (no Java rebuild):

- `/difficulty normal`
- `/gamerule doMobSpawning true`
- `/gamerule doDaylightCycle false`
- `/time set 18000`

Plus a deterministic pre-flight that `/summon`s zombies adjacent to the fake player to confirm the threat actually lands before committing to a long run (natural night spawns are stochastic; on `/difficulty normal` starvation caps at 1 HP and never kills, so a kill requires mob damage).

The policy is OUT-OF-DISTRIBUTION on every survival signal: it was trained at constant `health=20`, `hunger=20`, empty `g_hostiles_nearby`.

## Headline result

- **survived**: `None`
- **cause**: `FAKE_PLAYER_INVULNERABLE_NO_DAMAGE_LANDS`
- **steps to death**: `None`
- **steps run**: `0`
- **oak_log at end**: `0`
- **hostiles ever populated `g_hostiles_nearby`**: `True`
- **reacted to hostiles (dominant-skill shift)**: `None`

## Pre-flight (threat-lands check)

```
pre_h = 20.0
post_summon_h = 20.0
post_damage_h = 20.0
pre_hostiles = 0
post_summon_hostiles = 1
raw_alive = True
death_oracle_validated_separately = True
```

## Behavior under pressure

- no trace recorded. Summoned zombies populated g_hostiles_nearby (the obs channel works with real mobs) but did zero damage; a direct /damage burst also did nothing. The death oracle (raw-key absence) was validated in a separate probe via /kill, which DID remove the player (health->0, key gone), so the oracle is sound — the player is simply invulnerable to non-/kill damage in this config.

## Death / death-oracle methodology

A Carpet fake player that dies is REMOVED from the server, so its `player_name` disappears from `observationsAll()`. The robust death oracle used here is **raw-key presence in `bridge.observations_all()`** (absent key == dead), corroborated by the decoded `term` flag and the agent dropping out of `env.agents`. Decoded `health` is logged but NOT trusted as the primary death signal (it zero-fills to 0.0 on the missing-key step, which is itself only a derived signal).

## First 30 steps (trace)

| step | skill | health | hunger | hostiles | oak_log | pos | raw_alive |
|---|---|---|---|---|---|---|---|

## Honest caveats

- This is a single seeded run (seed=1) on one instance. Minecraft mob spawning/pathing has stochastic elements; treat counts as one sample, not a distribution.
- 'Reacted to hostiles' is a weak observational signal (dominant-skill shift between hostile/non-hostile steps), NOT proof of intent. The policy never saw a non-empty `g_hostiles_nearby` in training, so any apparent reaction is most likely incidental.
- Pressure was flipped at runtime mid-world, not baked into the arena generation; the arena remains the flat M1B ring.
- Some sections above (`MOB_ATTACK` cause-inference, the trace table, the reaction caveat) are templated for the death-path that never executed; on this run the recon stopped at the invulnerability gate before any policy steps. They are retained as documentation of the intended methodology.

---

## Analyst addendum — validated supplementary probes (the recon value)

The headline above is machine-emitted. The findings below were each confirmed by a
direct, separate probe against the live server (port 25001, MC 1.21.1). They are the
substantive recon result.

### Finding 1 — The Carpet fake player is INVULNERABLE to mob/`/damage` harm (headline)

Quantified, reproduced across two independent probes:

| stimulus | decoded health before | after | player removed? |
|---|---|---|---|
| 3 zombies summoned on top of player, ~6 WAIT steps under tick-warp | 20.0 | 20.0 | no |
| `/damage gatherer_0 10` + step | 20.0 | 20.0 | no |
| `/damage gatherer_0 19` + step (separate probe) | 20.0 | 20.0 | no |
| `/damage gatherer_0 100` (lethal magnitude) + step | 20.0 | 20.0 | no |
| `/gamemode survival` + `/difficulty hard`, then step | 20.0 | 20.0 | no |
| **`/kill gatherer_0`** + step | 20.0 | **0.0** | **yes (key gone)** |

Interpretation: under the current launch config, the Carpet fake player takes **no
net damage** from mobs, the `/damage` command, or a forced survival gamemode flip.
Only the unconditional `/kill` removes it. The most likely mechanism is that the fake
player spawns invulnerable / in a non-damageable mode (consistent with Carpet
fake-player defaults), but this recon did NOT isolate whether it is gamemode,
an invulnerability flag, or regen-masking — it only establishes the **observable
outcome**: incidental/mob damage never lands.

**Consequence for the stated goal:** survival pressure (night + mobs + hunger) as
flipped at runtime on this path **cannot put the policy's survival at stake.** A
600-step "it survived" run would be an artifact of invulnerability, not evidence of
any survival capability — so the recon correctly stops rather than report a bogus
"SURVIVED." Testing real survival would require changing how the fake player is
spawned (vulnerable survival-mode player) — a Java/launch-config change, out of scope
for a no-rebuild runtime recon.

### Finding 2 — The death oracle is sound

The robust death detector chosen for this recon (player_name absent from
`observationsAll()` == dead) was validated: `/kill` drove decoded health to 0.0, the
key vanished from the raw obs dict, and the agent was dropped from `env.agents` — all
three signals agreed. So a real death WOULD have been detected had one occurred.

### Finding 3 — `g_hostiles_nearby` works with real mobs (positive)

Summoned zombies populated `g_hostiles_nearby` (1–3 nonzero rows tracking real
`HostileEntity` instances within the 16-block scan box). The obs channel that was only
ever zero in training carries real mob data correctly. This is a genuine positive: the
observation plumbing for hostiles is functional, even though the policy was never
trained on a non-empty value.

### Finding 4 — UNPREDICTED: the policy collects ZERO oak_log on this harness path — independent of survival pressure

During the (now-superseded) first full run the greedy policy chose `HARVEST` on
599/600 steps and collected **0 oak_log**, ending stuck at `(65.3, 65, -62.5)`.
A clean **control** run — NO summon, NO survival-pressure flip, proper arena spawn at
`(64.5, 65, -47.5)`, fresh `reset_episode` — reproduced the **identical** pathology:
1 `NAVIGATE` to `(65.3, 65, -62.5)`, then 199 `HARVEST` collecting **0 oak_log**, same
final position.

This means the zero-collection is **NOT** caused by survival pressure or by the OOD
`g_hostiles_nearby` signal — it is a pre-existing artifact of THIS harness path
(load checkpoint → `reset()` → greedy single-step decode loop with a freshly
initialized LSTM, continuing without the scenario-runner's loop structure). The agent
NAVIGATEs ~15 blocks off the oak-log ring on step 1 and then HARVEST-spams in place
where nothing is in reach.

Honest framing: this does NOT contradict the documented "transfers 3/3" result, which
was produced by `scripts/transfer_eval.py`'s `run_instrumented` loop. It DOES show the
proven-policy result is **harness-sensitive** — a different but superficially-equivalent
greedy-rollout harness reproduces a degenerate trajectory. The discrepancy between this
loop and `run_instrumented` (both use the same `_greedy_decode` and the same checkpoint)
was not root-caused here and is flagged for follow-up; candidate differences include
LSTM-state seeding and the `max_episode_ticks`/scenario wiring. Until reconciled, treat
"3/3 transfer" as harness-specific, not unconditional.

### What this recon did NOT establish (no over-claims)

- It did **not** measure how the policy behaves under genuine survival threat — no
  threat ever landed, so behavior-under-attack, pathing-among-mobs, and
  HARVEST-under-attack are all **untested**.
- It did **not** determine the exact reason the fake player is invulnerable (gamemode
  vs flag vs regen) — only the observable no-damage outcome.
- It did **not** observe starvation: on `/difficulty normal` starvation caps at 1 HP
  and hunger never decremented off 20.0 in any probe.
- The single oak_log=0 discrepancy with `transfer_eval` is reported as an open
  anomaly, not resolved.

