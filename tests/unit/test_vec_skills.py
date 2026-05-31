"""Unit parity: vec_skills.vec_apply_skills must match looping scalar apply_skill.

The vectorized batched skill dynamics (vec_skills) replace VecGathererSim's per-env
scalar HARVEST/NAVIGATE walk loops. This test pins the batched path directly against
the scalar ``aiutopia.sim.skills.apply_skill`` looped per world: agent_pos
(allclose atol=1e-5), log_alive (exact), oak count (exact), and the n_clipped
popcount (exact) must all match for a battery of skills -- including the cap>1
multi-log HARVEST chain and large-move NAVIGATE the closed-form walk rewrites.
"""

from __future__ import annotations

import numpy as np

from aiutopia.sim.skills import apply_skill
from aiutopia.sim.vec_skills import vec_apply_skills
from aiutopia.sim.world import SimWorld


def _fresh_worlds(seeds):
    ws = []
    for s in seeds:
        w = SimWorld()
        w.reset(int(s), arena_mode="trees")
        ws.append(w)
    return ws


def _popcount(bitset: int) -> int:
    return bin(int(bitset)).count("1")


def _scalar_step(world, action):
    _, comp = apply_skill(world, action)
    return _popcount(comp.get("clippedAxesBitset", 0))


# (skill_type, spatial, scalar, target_class)
_ACTIONS = [
    (1, [0.0, 0.0, 0.0], 1.0 / 64.0, 0),  # HARVEST cap-1
    (1, [0.0, 0.0, 0.0], 3.0 / 64.0, 0),  # HARVEST cap-3 chain
    (1, [0.0, 0.0, 0.0], 1.0, 0),  # HARVEST cap-64 (clears arena)
    (1, [0.0, 0.0, 0.0], -0.5, 0),  # HARVEST scalar OOB -> bit3 set
    (1, [0.0, 0.0, 0.0], 0.5, 5),  # HARVEST non-oak class -> no collect
    (0, [0.3, 0.0, -0.2], 0.0, 0),  # NAVIGATE small
    (0, [-1.0, 0.1, 1.0], 0.0, 0),  # NAVIGATE large + spatial OOB (x,z)
    (0, [2.0, -2.0, 0.0], 0.0, 0),  # NAVIGATE spatial OOB on 2 axes
    (3, [0.0, 0.0, 0.0], 0.0, 0),  # SEARCH noop
    (4, [0.0, 0.0, 0.0], 0.0, 0),  # WAIT noop
    (5, [0.0, 0.0, 0.0], 0.0, 0),  # NOOP_BROADCAST
]


def test_vec_apply_skills_matches_scalar_per_world() -> None:
    """Each action applied to a fresh batch must equal the scalar per-world loop."""
    seeds = np.array([1, 2, 3, 7, 11], dtype=np.int64)
    B = len(seeds)

    for st, sp, sc, tc in _ACTIONS:
        vec_worlds = _fresh_worlds(seeds)
        scalar_worlds = _fresh_worlds(seeds)

        skill_type = np.full(B, st, dtype=np.int64)
        target_class = np.full(B, tc, dtype=np.int64)
        spatial = np.tile(np.asarray(sp, dtype=np.float64), (B, 1))
        scalar = np.full((B, 1), sc, dtype=np.float64)

        n_clipped = vec_apply_skills(vec_worlds, skill_type, target_class, spatial, scalar)

        for i in range(B):
            action = {
                "skill_type": int(st),
                "target_class": int(tc),
                "spatial_param": np.asarray(sp, dtype=np.float64),
                "scalar_param": np.asarray([sc], dtype=np.float64),
            }
            s_clip = _scalar_step(scalar_worlds[i], action)
            vw, sw = vec_worlds[i], scalar_worlds[i]
            dpos = float(np.max(np.abs(vw.agent_pos - sw.agent_pos)))
            assert np.allclose(vw.agent_pos, sw.agent_pos, atol=1e-5), (
                f"pos st={st} sc={sc} tc={tc} env {i}: max|d|={dpos}"
            )
            assert np.array_equal(vw.log_alive, sw.log_alive), f"alive st={st} env {i}"
            assert int(vw.inventory.get("oak_log", 0)) == int(sw.inventory.get("oak_log", 0)), (
                f"oak st={st} sc={sc} env {i}"
            )
            assert int(n_clipped[i]) == int(s_clip), f"n_clipped st={st} sc={sc} env {i}"


def test_vec_apply_skills_chained_sequence() -> None:
    """A multi-step interleaved sequence (chains compound) must stay in lockstep."""
    seeds = np.array([1, 2, 3, 7], dtype=np.int64)
    B = len(seeds)
    seq = [
        (1, [0.0, 0.0, 0.0], 3.0 / 64.0, 0),  # cap-3 chain
        (0, [0.3, 0.0, 0.3], 0.0, 0),  # NAVIGATE
        (1, [0.0, 0.0, 0.0], 5.0 / 64.0, 0),  # cap-5 chain from new pos
        (0, [-0.3, 0.0, -0.3], 0.0, 0),  # NAVIGATE back
        (1, [0.0, 0.0, 0.0], 2.0 / 64.0, 0),  # cap-2 chain
    ]
    vec_worlds = _fresh_worlds(seeds)
    scalar_worlds = _fresh_worlds(seeds)

    for st, sp, sc, tc in seq:
        skill_type = np.full(B, st, dtype=np.int64)
        target_class = np.full(B, tc, dtype=np.int64)
        spatial = np.tile(np.asarray(sp, dtype=np.float64), (B, 1))
        scalar = np.full((B, 1), sc, dtype=np.float64)
        vec_apply_skills(vec_worlds, skill_type, target_class, spatial, scalar)
        for i in range(B):
            action = {
                "skill_type": int(st),
                "target_class": int(tc),
                "spatial_param": np.asarray(sp, dtype=np.float64),
                "scalar_param": np.asarray([sc], dtype=np.float64),
            }
            apply_skill(scalar_worlds[i], action)

    for i in range(B):
        vw, sw = vec_worlds[i], scalar_worlds[i]
        assert np.allclose(vw.agent_pos, sw.agent_pos, atol=1e-5), f"pos env {i}"
        assert np.array_equal(vw.log_alive, sw.log_alive), f"alive env {i}"
        v_oak = int(vw.inventory.get("oak_log", 0))
        s_oak = int(sw.inventory.get("oak_log", 0))
        assert v_oak == s_oak, f"oak env {i}"


def test_vec_apply_skills_mixed_skill_types() -> None:
    """Different skill_type per env in one batched call (the real trainer case)."""
    seeds = np.array([1, 2, 3, 7, 11, 13], dtype=np.int64)
    B = len(seeds)
    sts = [1, 0, 1, 3, 0, 1]
    sps = [[0, 0, 0], [0.5, 0, 0.5], [0, 0, 0], [0, 0, 0], [-0.6, 0, 0.2], [0, 0, 0]]
    scs = [3.0 / 64.0, 0.0, 1.0 / 64.0, 0.0, 0.0, 5.0 / 64.0]
    tcs = [0, 0, 0, 0, 0, 0]

    vec_worlds = _fresh_worlds(seeds)
    scalar_worlds = _fresh_worlds(seeds)

    skill_type = np.asarray(sts, dtype=np.int64)
    target_class = np.asarray(tcs, dtype=np.int64)
    spatial = np.asarray(sps, dtype=np.float64)
    scalar = np.asarray(scs, dtype=np.float64).reshape(B, 1)
    n_clipped = vec_apply_skills(vec_worlds, skill_type, target_class, spatial, scalar)

    for i in range(B):
        action = {
            "skill_type": int(sts[i]),
            "target_class": int(tcs[i]),
            "spatial_param": np.asarray(sps[i], dtype=np.float64),
            "scalar_param": np.asarray([scs[i]], dtype=np.float64),
        }
        s_clip = _scalar_step(scalar_worlds[i], action)
        vw, sw = vec_worlds[i], scalar_worlds[i]
        assert np.allclose(vw.agent_pos, sw.agent_pos, atol=1e-5), f"pos env {i}"
        assert np.array_equal(vw.log_alive, sw.log_alive), f"alive env {i}"
        v_oak = int(vw.inventory.get("oak_log", 0))
        s_oak = int(sw.inventory.get("oak_log", 0))
        assert v_oak == s_oak, f"oak env {i}"
        assert int(n_clipped[i]) == int(s_clip), f"n_clipped env {i}"
