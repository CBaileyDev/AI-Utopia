# N7.5 — Ray 2.55 Tune stop-key path investigation

**Status:** RESEARCH COMPLETE. Original key path `custom_metrics/M1/gate_passed`
was CORRECT. The warning that caused the team to revert it was a benign
`log_once` artifact emitted on iteration 1, before the gate callback had
written anything into `custom_metrics`. Empirically reproduced.

**Scope:** read-only investigation. `scripts/train.py` and
`src/aiutopia/train/callbacks.py` were not modified (M1B run in flight).

**Ray version under test:** 2.55.1 (verified `py -3.11 -c "import ray; print(ray.__version__)"`).

---

## A. The actual key path

`custom_metrics/M1/gate_passed` IS a valid Tune stop-criterion key in Ray 2.55,
provided the callback has populated it at least once before Tune evaluates the
stop dict.

### How the result flows from callback to `should_stop`

1. `Algorithm._compile_iteration_results` (algorithm.py:3909) assembles
   `train_results` into the result dict. **On the new API stack it does NOT
   touch `custom_metrics`** — that key only gets attention in the old-API
   path (`_compile_iteration_results_old_api_stack` at algorithm.py:4380,
   which calls `iteration_results.pop("custom_metrics", {})`).
   So on the new API stack, whatever the callback writes into
   `result["custom_metrics"]` survives verbatim.

2. `Algorithm.log_result(result)` (algorithm.py:3287) fires the
   `on_train_result` callback BEFORE forwarding to `Trainable.log_result`.
   Our `EvalGateStopCallback.on_train_result` mutates `result` in place
   (`result["custom_metrics"]["M1/gate_passed"] = 1.0`). The mutation
   therefore persists into the dict that Tune ultimately sees.

3. `TuneController._process_trial_result` (tune_controller.py:1564) flattens:
   ```python
   flat_result = flatten_dict(result)         # delimiter='/'
   if self._stopper(...) or trial.should_stop(flat_result):
       decision = TrialScheduler.STOP
   ```
   `ray.tune.utils.flatten_dict` uses `/` as its delimiter, so
   `{"custom_metrics": {"M1/gate_passed": 1.0}}` flattens to
   `{"custom_metrics/M1/gate_passed": 1.0}`.

4. `Trial.should_stop` (trial.py:805–826) iterates the stop dict, checks
   `criterion in result` (the flat dict), and stops when value ≥ threshold.

So the path `custom_metrics/M1/gate_passed` matches the flattened key
exactly. **No code change to the gate write or the stop dict was ever
required.**

### Why the warning fired

Two interacting facts:

- `flatten_dict` **silently drops empty nested dicts** (verified
  empirically — see Experiment 1 below). If `result["custom_metrics"] == {}`
  on a given iteration, the flat result contains no `custom_metrics/*`
  keys at all.
- `Trial.should_stop` wraps its missing-key warning in
  `log_once("tune_trial_stop_criterion_not_found")` (trial.py:817–823).
  It fires **exactly once per process**, then is silenced forever.

On iteration 1, the gate callback's `on_train_result` typically returns
early — `M1EvalScenarioCallback` only emits `eval_m1_oak_log_success_rate`
every `eval_interval=10` iters, so on iter 1 `rate is None` and the gate
returns without writing anything. `AiUtopiaMetricsCallback` does call
`result.setdefault("custom_metrics", {})`, but if `result["info"]["learner"]`
is missing (or empty on iter 1), it then returns without populating any
key, leaving `custom_metrics` as `{}`. After `flatten_dict` strips the
empty dict, the available-keys list shown in the warning contains no
`custom_metrics/*` entries — exactly the symptom that was reported.

By iteration 10 (or whenever the first evaluation completes), the gate
callback writes `custom_metrics["M1/gate_history"]` and
`custom_metrics["M1/gate_passed"] = 0.0`, the flat result contains
`custom_metrics/M1/gate_passed`, and `should_stop` will trigger correctly
the moment that value crosses the threshold. But because `log_once`
already fired, the warning never reappears to confirm "actually, the key
is here now". The warning is misleading by construction.

### The available-keys list in the original report (decoded)

The truncated list (`'num_training_step_calls_per_iteration', ...,
'timers/env_runner...'`) is the OUTPUT of `flatten_dict` — note the `/`
delimiters in `timers/training_iteration` etc. It is consistent with
new-API-stack PPO at iter 1 with no callback contributions yet, which
matches our diagnosis.

---

## B. The minimal change for M2

**Recommended: restore the original stop dict and seed the metric at zero
from iter 1.**

In `scripts/train.py` `RunConfig`:

```python
stop={
    "custom_metrics/M1/gate_passed": 0.5,   # gate threshold
    "training_iteration":            args.max_iters,  # safety ceiling
},
```

In `EvalGateStopCallback.on_train_result`, hoist the `setdefault` writes
ABOVE the early-return on `rate is None`:

```python
def on_train_result(self, *, algorithm, metrics_logger=None,
                       result, **kwargs):
    # Seed unconditionally so flatten_dict doesn't drop custom_metrics
    # on iterations before the first evaluation completes.
    result.setdefault("custom_metrics", {})
    result["custom_metrics"].setdefault(
        f"{self.milestone}/gate_passed", 0.0)

    sampler = result.get("env_runners", result.get("sampler_results", {}))
    rate    = sampler.get("episode_extra_stats", {}).get(self.success_metric)
    if rate is None:
        return
    # ... rest unchanged ...
```

That single change eliminates the `log_once` warning, makes the available
keys reflect reality from iter 1, and avoids future operator confusion. The
0.5 threshold is well below the 1.0 written on pass and above the 0.0
written when not yet passing.

### Alternative: custom `Stopper`

If the team prefers to keep the metric write conditional, a small
`tune.Stopper` subclass that reads the flat dict directly works too:

```python
from ray.tune.stopper import Stopper

class GatePassedStopper(Stopper):
    def __init__(self, milestone="M1", threshold=0.5):
        self._key       = f"custom_metrics/{milestone}/gate_passed"
        self._threshold = threshold
    def __call__(self, trial_id, result):
        # `result` here is already the flat dict (tune_controller.py:1567
        # passes flat_result through self._stopper).
        return result.get(self._key, 0.0) >= self._threshold
    def stop_all(self):
        return False
```

Used as `RunConfig(stop=GatePassedStopper(...))`. No `log_once` warning
because `Stopper.__call__` does its own lookup and does not log on miss.

The Stopper route is the cleaner long-term answer because it makes the
gate criterion explicit and self-documenting; the stop-dict route is the
minimal-diff answer.

### NOT recommended: `metrics_logger.log_value`

`RLlibCallback` in 2.40+ passes a `metrics_logger` kwarg backed by
`ray.rllib.utils.metrics.metrics_logger.MetricsLogger`. Values logged
through it land in `result["env_runners"]["..."]` or
`result["learners"]["..."]` namespaces (depending on which logger you grab),
which is a different and less predictable key path. Sticking with direct
`result["custom_metrics"]` mutation is what the existing tests, dashboards,
and `aiutopia_metrics.json` consumer in `train.py:129` already expect; do
not migrate write path without updating all three consumers.

---

## C. The unit test that would have caught this

`tests/unit/test_evaluation_gate_callback.py` exercises
`EvalGateStopCallback.on_train_result` in isolation and verifies its
**output dict**. It does NOT verify Ray's propagation chain, so it cannot
catch:

- `flatten_dict` dropping empty `custom_metrics` on early iterations.
- Whether the stop-dict key path actually matches the flattened key.
- Whether `Trial.should_stop` accepts the key.

### Suggested integration test

Add a `tests/integration/test_tune_stop_propagation.py` that runs a stub
Trainable through `tune.Tuner`. The pattern (verified working in this
investigation) is:

```python
import tempfile, ray
from ray import tune
from ray.tune import Trainable
from aiutopia.train.callbacks import EvalGateStopCallback

class _FakePPO(Trainable):
    """Stand-in that mimics the result-dict shape PPO emits and routes it
    through the gate callback, but skips actual training."""
    def setup(self, cfg):
        self._i  = 0
        self._cb = EvalGateStopCallback(threshold=0.8, consecutive_required=3)
    def step(self):
        self._i += 1
        # Simulate eval scenario producing 0.9 success every 5 iters.
        result = {"env_runners": {"episode_extra_stats": {}}}
        if self._i >= 3:
            result["env_runners"]["episode_extra_stats"][
                "eval_m1_oak_log_success_rate"] = 0.95
        self._cb.on_train_result(algorithm=None, result=result)
        return result
    def save_checkpoint(self, d): return d
    def load_checkpoint(self, d): pass

def test_gate_stop_key_path_triggers_tune_stop():
    ray.init(num_cpus=2, include_dashboard=False, configure_logging=False)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tuner = tune.Tuner(
                _FakePPO,
                run_config=tune.RunConfig(
                    name="gate_stop_test",
                    storage_path=tmp,
                    stop={"custom_metrics/M1/gate_passed": 0.5,
                          "training_iteration": 50},
                    verbose=0,
                ),
            )
            results = tuner.fit()
            stopped_at = results[0].metrics["training_iteration"]
            # 3 consecutive >=0.8 needed; eval starts firing at iter 3.
            # Gate passes on iter 5 (3rd consecutive observation).
            assert 5 <= stopped_at < 50, (
                f"trial should stop on gate, got iter={stopped_at}")
    finally:
        ray.shutdown()
```

This catches: stop-key path regressions, `flatten_dict` behavior changes
in future Ray versions, and accidental key renames in the callback. It
runs in well under a minute and does not need MuJoCo, Fabric, or
Minecraft envs. Note: under recent Ray (>=2.42-ish) `local_mode=True` is
removed; use the normal `ray.init` and accept a 1–2s startup tax.

### Verification I performed inline for this memo

1. **Empty-dict drop**: `flatten_dict({'custom_metrics': {}, 'foo': 1})`
   returns `{'foo': 1}`. Confirmed.
2. **End-to-end key path with non-empty value**: stub Trainable emitting
   `{"custom_metrics": {"M1/gate_passed": 1.0 if i>=3 else 0.0}}` with
   `stop={"custom_metrics/M1/gate_passed": 0.5}` stops at iter 3 as
   expected. Output: `STOPPED_AT_ITER: 3`.
3. **`log_once` reproduction**: stub Trainable emitting empty
   `custom_metrics` on iters 1–2, then real data on iter 3, with the
   same stop dict, stops at iter 3 AND emits exactly one
   `Stopping criterion 'custom_metrics/M1/gate_passed' not found`
   warning on the first iteration. This is the M1B symptom verbatim.

---

## D. M2 implications: other callbacks under the same trap

The same key-path mechanics apply to every callback that writes into
`result["custom_metrics"]`. Concretely in this repo:

### `AiUtopiaMetricsCallback` — `custom_metrics/<policy_id>/{entropy,vf_loss,kl}`

Writes only when `result["info"]["learner"]` exists and the policy has the
source keys. On iters before the first learner update completes, this is
silently a no-op. Same `flatten_dict` drop applies — the keys are absent
for the first iter or two and then appear. TensorBoard's
`flatten_dict(result, delimiter='/')` (tensorboardx.py:126) writes them
under `custom_metrics/gatherer_policy/entropy` etc. once present, with
gaps for the missing iters. That is correct behavior; consumers
(`train.py:129`) should `.get(...)` with a default, which they do.

### `ExploitHuntCallback` — `custom_metrics/exploit_hunt/<key>`

Writes only every `every_n_iters=200` AND only when
`episode_extra_stats` has keys starting with `exploit_`. Even at iter 200,
if no exploits have fired, nothing is written. For 199 iters out of every
200 the key is absent. TensorBoard will log a sparse scalar series — that
is fine for visualization (TB treats missing steps as gaps, not as zero).
Anyone in M2 attempting `stop={"custom_metrics/exploit_hunt/exploit_X_total": ...}`
would hit the same `log_once` warning at iter 1 and then have it correctly
stop later, but should not be deterred by the warning.

**Recommendation for M2 callbacks:** adopt the convention that any
callback contributing keys consumed downstream (stop dict, scheduler,
search alg, evaluation gate) must seed its top-level scalar keys to a
sentinel default (0.0, or `float('nan')` for ones that are explicitly
"no observation yet") on every `on_train_result`, BEFORE any conditional
return. Sketch:

```python
class _SeedingCallback(RLlibCallback):
    SEED_KEYS: ClassVar[dict[str, float]] = {}   # overridden per subclass
    def on_train_result(self, *, algorithm, metrics_logger=None,
                           result, **kwargs):
        result.setdefault("custom_metrics", {})
        for k, v in self.SEED_KEYS.items():
            result["custom_metrics"].setdefault(k, v)
```

Then `EvalGateStopCallback` declares
`SEED_KEYS = {"M1/gate_passed": 0.0}`, `ExploitHuntCallback` declares
`SEED_KEYS = {"exploit_hunt/exploit_total_per_episode": 0.0}`, etc.
This eliminates the entire class of "Tune doesn't see my custom metric"
confusion across M2/M3.

### Side note: `M1/gate_history` is a list

`Trial.update_last_result` (trial.py:892) only stores values where
`isinstance(value, Number)` into the tracked metrics. The
`M1/gate_history` list value the gate callback writes is therefore
**dropped from `run_metadata.update_metric` tracking** (no TB scalar, no
sortable Tune metric). It still appears in `result.json` and in
`result_grid.metrics`, which is fine — the list is for human inspection,
not Tune consumption. Not a bug, but worth knowing if M2 tries to
condition on `gate_history`: don't, condition on `gate_passed`.

---

## Summary

- **Root cause:** `log_once` warning on iter 1 from an empty
  `custom_metrics` nested dict that `flatten_dict` silently strips.
  Not a key-path bug.
- **Correct key path:** `custom_metrics/M1/gate_passed` — works in Ray
  2.55, verified empirically.
- **Minimal fix for M2:** seed `custom_metrics["M1/gate_passed"] = 0.0`
  unconditionally at the top of `EvalGateStopCallback.on_train_result`,
  and restore `stop={"custom_metrics/M1/gate_passed": 0.5,
  "training_iteration": max_iters}` in `train.py`.
- **Cleaner long-term fix:** a small `tune.Stopper` subclass that reads
  the flat key with a `.get(..., 0.0)` default — no `log_once` warning,
  self-documenting criterion.
- **Test gap:** existing unit tests only check callback output. Add a
  Tune-integration test with a stub Trainable to lock in the propagation
  contract.

## Appendix: source locations referenced

- `ray/tune/experiment/trial.py:805–826` — `Trial.should_stop`
- `ray/tune/experiment/trial.py:817–823` — `log_once` wrapping the warning
- `ray/tune/experiment/trial.py:880–896` — `update_last_result`
  (drops non-`Number` values)
- `ray/tune/execution/tune_controller.py:1564–1567` — `flatten_dict`
  + `trial.should_stop(flat_result)`
- `ray/_private/dict.py` — `flatten_dict`, deprecated tag, drops empty
  nested dicts
- `ray/rllib/algorithms/algorithm.py:3287–3307` — `Algorithm.log_result`
  fires `on_train_result` callback before `Trainable.log_result`
- `ray/rllib/algorithms/algorithm.py:3909–3953` — new-API
  `_compile_iteration_results` (does not touch `custom_metrics`)
- `ray/rllib/algorithms/algorithm.py:4367–4412` — old-API
  `_compile_iteration_results_old_api_stack` (does pop+merge
  `custom_metrics`)
- `src/aiutopia/train/callbacks.py:61–100` —
  `EvalGateStopCallback.on_train_result`
- `scripts/train.py:106` — current (post-revert) `stop={...}` line
