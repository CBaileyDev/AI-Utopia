# Multi-Agent Minecraft AI Village — Architectural Research Report

> **Status:** Pre-implementation research synthesis
> **Project:** Multi-Agent Realm Orchestrator — a swarm of role-specialized AI agents that collaboratively grow a Minecraft village under LLM-driven high-level planning and multi-agent reinforcement learning.
> **Target hardware:** Ryzen 9800X3D, 64GB DDR5, RTX 4080 16GB (solo developer workstation)
> **Source:** Synthesis of Gemini Deep Research output with architectural corrections and additions.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Top Architectural Decisions](#top-architectural-decisions)
3. [Prior Art in Minecraft AI](#1-prior-art-in-minecraft-ai)
4. [Mod Ecosystem and Integration Layer](#2-mod-ecosystem-and-integration-layer)
5. [Observation and Action Space Design](#3-observation-and-action-space-design)
6. [Multi-Agent RL Frameworks](#4-multi-agent-rl-frameworks)
7. [LLM-as-Planner Architecture](#5-llm-as-planner-architecture)
8. [Reward Shaping](#6-reward-shaping)
9. [Multi-Agent Failure Modes](#7-multi-agent-failure-modes)
10. [Server Performance and Accelerated Training](#8-server-performance-and-accelerated-training)
11. [Episode Reset and Determinism](#9-episode-reset-and-determinism-gap-filled)
12. [Debugging and Observability](#10-debugging-and-observability-gap-filled)
13. [Bedrock Compatibility via Geyser](#11-bedrock-compatibility-via-geyser)
14. [Reference Repositories](#12-reference-repositories)
15. [Revised Milestone Plan](#revised-milestone-plan)
16. [Risks and Showstoppers](#risks-and-showstoppers)
17. [Reading List](#reading-list-priority-ranked)

---

## Executive Summary

The state of the art in Minecraft AI converged on a clear pattern in 2024–2025: **LLMs as slow, event-driven high-level planners; reinforcement learning as fast, decentralized low-level controllers; a library of pre-built skills bridging the two.** Every successful long-horizon Minecraft agent (Voyager, GITM, JARVIS-1, Plan4MC, Odyssey) uses some variant of this hierarchy. Every project that tried end-to-end RL from pixels failed to scale beyond narrow tasks.

For a multi-agent village builder, this means four hard architectural commitments before writing code:

1. **Hierarchical hybrid architecture.** The LLM emits parameterized JSON task graphs. RL policies execute discrete semantic skills. Never put the LLM in the inner control loop.
2. **Server-side Fabric mod with WebSocket + JSON-RPC bridge.** Not Mineflayer (latency, sync issues). Not gRPC (overkill, poor tooling fit). WebSockets give bidirectional streaming, browser-debuggable traffic, and alignment with the Model Context Protocol standard.
3. **Symbolic observations and parameterized actions.** No pixel-based RL. Egocentric voxel grids encoded as tensors. Action space is `(skill_type, target, spatial_offset)` — not raw keypresses.
4. **CTDE multi-agent algorithms (MAPPO or QMIX) via MARLlib on Ray RLlib.** Centralized training, decentralized execution. The only proven path through the non-stationarity and credit-assignment problems.

The original project plan was directionally correct but underspecified the bridge protocol, the action space, and the LLM/RL synchronization model. This report locks those down.

---

## Top Architectural Decisions

These five decisions cascade into everything else. Lock them before writing code.

### 1. Hierarchical, not end-to-end

The LLM is a slow planner. RL policies are fast controllers. Skills are the interface between them. No raw pixels, no JavaScript code synthesis at runtime, no LLM in the per-tick loop.

### 2. WebSocket + JSON-RPC, not gRPC, not Mineflayer

- **Mineflayer** spawns one Node.js process per agent. Client-side chunk loading caps observation fidelity. Cannot stay synchronized with accelerated server tick rates.
- **gRPC** adds proto-buffer serialization overhead and tooling friction for marginal type-safety gains over JSON Schema validation.
- **WebSockets + JSON-RPC** are natively bidirectional, debuggable with off-the-shelf tools, align with the Model Context Protocol, and are already proven in production Fabric mods like [Not-Enough-Management](https://github.com/Dan-Mizu/Not-Enough-Management/).

### 3. Symbolic observation, parameterized discrete actions

Observation: structured JSON dictionary (global state + egocentric voxel grid + inventory + peer agent states) flattened into a tensor.

Action: a `Dict` space with `skill_type ∈ {Idle, Navigate, Mine, Place, Attack, Interact}`, a `target_block_class`, and a 3D `spatial_offset`.

Pathfinding is delegated to embedded Baritone — the RL policy never learns block-by-block navigation.

### 4. MAPPO via MARLlib on Ray RLlib

PettingZoo defines the environment interface. MARLlib provides battle-tested MAPPO implementations. Ray RLlib handles distributed rollouts. Start with independent PPO per role for the first milestone, migrate to MAPPO/CTDE when introducing multi-agent rewards.

### 5. Deferred Bedrock support

Build for Java only in Phases 1–4. Design custom packets to avoid Geyser incompatibilities, but do not ship Geyser until the core RL system is stable. The doc reviewing Bedrock specifically says drop it; the right framing is *defer* it.

---

## 1. Prior Art in Minecraft AI

The field split cleanly into two paradigms around 2022, and neither survived alone. Modern systems are hybrids.

### The two paradigms

**Low-level controllers (video pretraining era).** OpenAI's VPT used behavioral cloning on 70k hours of YouTube footage to learn keyboard-and-mouse control directly from pixels, eventually crafting a diamond pickaxe. STEVE-1 extended VPT with MineCLIP to make it text-conditionable. Both work for short-horizon instruction following but suffer catastrophic forgetting over long tasks — if the agent is interrupted mid-task, it loses the context of what it was doing and fails.

**High-level planners (LLM era).** Voyager bypassed RL entirely, using GPT-4 to write JavaScript that calls the Mineflayer API, storing successful programs in a growing skill library. GITM discarded pixels in favor of a text-based voxel representation and achieved 55% on the ObtainDiamond benchmark. DEPS added a closed-loop selector that picks viable sub-tasks based on current state.

**The synthesis.** Plan4MC, JARVIS-1, MP5, and Odyssey all converged on the same recipe: LLM for offline skill graph generation and online task decomposition; RL or hand-coded policies for skill execution; explicit memory and reflection layers for failure recovery.

### Reference projects compared

| Project | Architecture | Observation | Action | Notable result | Limitation |
|---------|-------------|-------------|--------|----------------|------------|
| **VPT** | Behavioral cloning from video | Pixels | Keypresses | Diamond pickaxe from scratch | Catastrophic forgetting; compute-heavy |
| **STEVE-1** | Text-conditioned VPT | Pixels | Keypresses | Strong short-horizon performance | No episodic memory |
| **MineDojo** | Internet-scale priors + RL | Pixels + text | Keypresses | Massive task diversity benchmark | RL bottleneck on sequential crafting |
| **Voyager** | GPT-4 writes Mineflayer code | Text via API | Generated JavaScript | Continual skill discovery | High API cost; variable execution time |
| **GITM** | LLM planner + symbolic state | Text voxel grid | Symbolic actions | 55% ObtainDiamond | Hand-coded low-level policies |
| **DEPS** | LLM with selector module | Text | High-level skills | Dynamic replanning | Depends on controller success |
| **Plan4MC** | LLM-built skill graph + RL | Symbolic | Compressed discrete | 24 tasks with >10 sequential skills | Terrain generalization gap |
| **JARVIS-1** | Multimodal LLM + memory | Pixels + text | High-level skills | Lifelong self-improvement | Multimodal latency |
| **MP5** | Objective-conditioned perception | Pixels on demand | Symbolic | Token-efficient observation | Tuning the perception trigger |
| **DeepMind XLand** | Open-ended RL + CTDE | Egocentric 3D | Primitive actions | General capability from procedural tasks | Not built on Minecraft |

### What this means for the village

- **Do not use end-to-end visual RL.** Sample inefficiency and inference cost will kill the project on a single workstation.
- **Do not let the LLM write code at runtime.** Voyager's variable execution time will not synchronize with a 20 Hz RL tick loop. The LLM emits structured JSON; pre-built skills execute it.
- **Study Odyssey and MINDCraft most closely.** They are the closest existing implementations of the role-based, action-by-action collaborative paradigm the village needs.
- **Plan4MC's skill graph approach is the right pattern.** Use the LLM offline to generate a comprehensive skill graph, then have RL execute within it.

---

## 2. Mod Ecosystem and Integration Layer

The choice of integration layer caps the upper bound on training performance. Get this wrong and nothing else matters.

### Options compared

| Interface | Architecture | Throughput | Sync with tick-warp | Verdict |
|-----------|-------------|------------|---------------------|---------|
| **Mineflayer** | External Node.js client per agent | Limited by TCP + chunk loading | Poor — client cannot keep up | Skip |
| **Baritone (client mod)** | Client-side pathfinding | High for nav only | N/A | Embed, do not use standalone |
| **MCP-Reborn / fabric-mcp** | Server-side WebSocket JSON-RPC | Good for async LLM tools | Possible with hooks | Reference architecture |
| **Not-Enough-Management** | Server-side WebSocket mod | Excellent | Excellent with Carpet | Best blueprint to study |
| **Custom Fabric mod** | Server-side, in-process | Best possible | Native | **Build this** |

### Why a custom server-side Fabric mod wins

A custom server-side mod has direct in-process access to the full server state. There is no client render distance limit, no TCP round-trip, no separate process per agent. The mod can:

- Spawn fake players using Carpet Mod's fake player API and control them directly via server-side AI.
- Read the full chunked world state, all entity positions, all inventories, all redstone state.
- Hook the server tick loop to implement synchronous stepping — halt the tick, await Python's action payload, execute, advance, stream observation back.
- Embed Baritone for pathfinding so the RL action space stays small and semantic.

### Tooling stack inside the mod

- **Fabric Loader + Fabric API:** the base modding framework.
- **Mixins:** for surgical hooks into the server tick loop and entity AI internals.
- **Carpet Mod (Fabric Carpet):** provides `/tick warp` for accelerated training, fake player API, and headless server tooling. Non-negotiable.
- **Baritone:** embedded as a library, exposed via the mod's API rather than as a standalone client mod. RL policies emit `Navigate(target_xyz)` and Baritone handles the execution.
- **Java WebSocket library (e.g., Java-WebSocket or Netty):** for the JSON-RPC bridge to Python.

### JSON-RPC over WebSocket: protocol sketch

The mod exposes two channels per agent:

**Observation stream (server → Python):** push-based JSON messages, one per tick (or per N ticks during accelerated training):

```json
{
  "jsonrpc": "2.0",
  "method": "observation",
  "params": {
    "agent_id": "gatherer_1",
    "tick": 14820,
    "global_state": { "time_of_day": 12000, "weather": 0 },
    "egocentric_voxel_grid": { "shape": [9, 9, 9], "data": "<base64 int8>" },
    "inventory": { "oak_log": 12, "wooden_axe": 1 },
    "peer_agents": [
      { "id": "builder_1", "rel_pos": [5, -1, 2], "current_skill": "place" }
    ]
  }
}
```

**Action channel (Python → server):** request-response. Python sends an action; mod responds when the action terminates (success, failure, or timeout):

```json
{
  "jsonrpc": "2.0",
  "id": 4821,
  "method": "step",
  "params": {
    "agent_id": "gatherer_1",
    "action": {
      "skill_type": 2,
      "target_block_class": 17,
      "spatial_offset": [3, 0, -2]
    }
  }
}
```

This decouples the Python tick rate from the Minecraft tick rate cleanly. When the server runs at 500 TPS via `/tick warp`, the WebSocket simply streams faster.

### Open question to resolve in Milestone 1

Baritone's pathfinding has variable execution time. A `Navigate` action might complete in 50ms or 5 seconds depending on terrain. The synchronous `env.step()` API in PettingZoo expects predictable returns. Two options:

1. **Block until completion or timeout** (simpler, but wastes server time on slow paths during accelerated training).
2. **Multi-tick skill abstraction** — the action is queued and the agent reports `current_skill=navigating` in subsequent observations until done.

Recommendation: start with option 1 with a generous timeout, migrate to option 2 if it becomes a bottleneck.

---

## 3. Observation and Action Space Design

The shape of these two spaces determines whether the policies converge. Get them wrong and you will train for weeks and learn nothing.

### Observation space

Symbolic, structured, ego-centric. No pixels.

**Top-level schema (per agent, per tick):**

```python
from gymnasium import spaces
import numpy as np

observation_space = spaces.Dict({
    # Scalar global context
    "global_state": spaces.Dict({
        "time_of_day": spaces.Box(low=0, high=24000, shape=(1,), dtype=np.int32),
        "weather": spaces.Discrete(3),  # clear, rain, thunder
        "village_center_dist": spaces.Box(low=0, high=1000, shape=(1,), dtype=np.float32),
    }),

    # Egocentric voxel grid (9x9x9 around agent)
    # Channels: [block_id, walkable, interactable, light_level]
    "voxel_grid": spaces.Box(
        low=0, high=255, shape=(4, 9, 9, 9), dtype=np.uint8
    ),

    # Inventory: 50 most relevant item slots
    "inventory": spaces.Box(low=0, high=64, shape=(50,), dtype=np.int32),

    # What the LLM has asked this agent to do
    "current_goal": spaces.Dict({
        "target_item": spaces.Discrete(50),
        "target_quantity": spaces.Box(low=0, high=128, shape=(1,), dtype=np.int32),
    }),

    # Up to N peer agents, including their current goal (mitigates non-stationarity)
    "peer_agents": spaces.Tuple([
        spaces.Dict({
            "role": spaces.Discrete(4),  # gatherer, builder, farmer, defender
            "relative_position": spaces.Box(low=-50, high=50, shape=(3,), dtype=np.int32),
            "current_skill": spaces.Discrete(6),
            "current_goal_id": spaces.Discrete(64),
        }) for _ in range(4)  # max 4 peers
    ])
})
```

**Why these specific choices:**

- **9×9×9 voxel grid** (729 voxels) is large enough to see relevant terrain and small enough to keep observation tensors compact. Plan4MC used similar sizes.
- **4 channels per voxel** (block id, walkable, interactable, light level) gives the policy semantic structure without raw block IDs becoming the entire feature space.
- **Peer agent current goals are included.** This is the single most important MARL non-stationarity mitigation. If the gatherer can see that the builder is currently `placing_wall`, the builder's pathing looks intentional rather than random.

### Action space

Parameterized discrete. The skill is the high-level intent; the parameters specialize it.

```python
action_space = spaces.Dict({
    # 0: Idle, 1: Navigate, 2: Mine, 3: Place, 4: Attack, 5: Interact
    "skill_type": spaces.Discrete(6),

    # Block / item / entity class the action targets
    "target_class": spaces.Discrete(50),

    # Spatial offset relative to agent position (-15..+15 on each axis)
    "spatial_offset": spaces.Box(low=-15, high=15, shape=(3,), dtype=np.int32),
})
```

**Skill semantics:**

| `skill_type` | Meaning | Parameters used | Executor |
|--------------|---------|-----------------|----------|
| 0 Idle | Do nothing this tick | — | — |
| 1 Navigate | Move to `agent_pos + spatial_offset` | `spatial_offset` | Baritone |
| 2 Mine | Mine the nearest block of `target_class` within `spatial_offset` range | `target_class`, `spatial_offset` | Mod's mining logic |
| 3 Place | Place `target_class` at `agent_pos + spatial_offset` | `target_class`, `spatial_offset` | Mod's place logic |
| 4 Attack | Attack the nearest entity of `target_class` within range | `target_class`, `spatial_offset` | Mod's combat logic |
| 5 Interact | Open crafting table / chest / use item | `target_class` | Mod's interaction logic |

This gives roughly 6 × 50 × 30³ ≈ 8 million actions — but the policy learns conditional distributions, not flat enumerations, and most combinations are masked as illegal by the environment.

### Handling variable-duration actions

Mining obsidian takes 9.4 seconds at vanilla speed; mining dirt takes 0.15. Two-step pattern:

1. Agent emits a `Mine` action.
2. Mod begins the mining animation server-side.
3. Subsequent observations report `current_skill=mining, progress=0.4` until complete.
4. The Python environment continues stepping; the policy can choose to wait (emit `Idle`) or switch tasks.

This is more flexible than blocking the env step until completion and matches how real players multi-task.

---

## 4. Multi-Agent RL Frameworks

### Stack recommendation

**PettingZoo (env API) → MARLlib (algorithm wrapper) → Ray RLlib (distributed engine).**

This stack:
- Lets you define one environment and run heterogeneous agents through it.
- Maps each role (gatherer, builder, farmer, defender) to its own policy network via `policy_mapping_fn`.
- Scales to dozens of parallel rollout workers without writing distributed code.
- Has the best debugging story in the field (Ray Dashboard, TensorBoard integration, replay buffers).

### Why not the alternatives

- **EPyMARL:** rigid, SMAC-focused, struggles with heterogeneous observation spaces.
- **Raw RLlib without MARLlib:** doable but requires writing MAPPO from scratch with all the scaling tricks (mini-batch SGD iterations, advantage normalization across agents) that MARLlib already implements correctly.
- **CleanRL:** great for single-agent learning, no multi-agent story.

### Algorithm choice: MAPPO

For cooperative multi-agent settings with shared rewards, **MAPPO (Multi-Agent PPO)** is the current default.

- Stable training, well-tuned hyperparameters available.
- CTDE — centralized critic during training, decentralized actors during execution.
- MARLlib's implementation includes the algorithm-specific tricks (input normalization, value clipping, mini-batch SGD ordering) that academic papers gloss over.

QMIX is an alternative for fully-discrete action spaces with stronger non-stationarity guarantees but requires more careful tuning.

### Heterogeneous policy mapping

```python
def policy_mapping_fn(agent_id, episode, **kwargs):
    role = agent_id.split("_")[0]  # "gatherer_1" -> "gatherer"
    return f"{role}_policy"

config = {
    "multiagent": {
        "policies": {
            "gatherer_policy": (None, obs_space, act_space, {}),
            "builder_policy":  (None, obs_space, act_space, {}),
            "farmer_policy":   (None, obs_space, act_space, {}),
            "defender_policy": (None, obs_space, act_space, {}),
        },
        "policy_mapping_fn": policy_mapping_fn,
    }
}
```

All four agents share observation/action space shapes but each role learns its own policy weights. The centralized critic during training sees all four agents' states.

### Scaling on the 9800X3D workstation

- 4–6 parallel headless Fabric server instances in Docker.
- 1 Ray worker per server, plus 1 trainer worker.
- 16 cores total: 6 for servers, 6 for Ray rollouts, 2 for trainer, 2 spare for OS / dashboard.
- 64GB DDR5 is ample: ~4GB per server, ~8GB for the trainer, headroom for the replay buffer.
- The RTX 4080 16GB handles policy inference and gradient updates. PyTorch will not be the bottleneck for MLP policies of this scale.

---

## 5. LLM-as-Planner Architecture

The LLM is event-driven, not tick-driven. This is the single most important rule.

### The two-loop pattern

**Fast loop (RL controller):** runs at 20 TPS (or 500 TPS during training). Executes pre-trained skill policies. Reports success / failure / timeout events.

**Slow loop (LLM planner):** runs only when triggered. Triggers are:
1. Initial human prompt arrives ("build a small wooden house with a farm").
2. A task in the DAG completes, unblocking its dependents.
3. A task fails (skill returned error or timeout).
4. A catastrophic event occurs (agent died, structure griefed).

The LLM never ticks in the fast loop. The fast loop is RL all the way down.

### Plan representation: typed JSON DAG

The LLM emits a directed acyclic graph of tasks, validated by a Pydantic schema:

```python
from pydantic import BaseModel
from typing import Literal

class Task(BaseModel):
    task_id: int
    role_target: Literal["gatherer", "builder", "farmer", "defender"]
    skill_call: str  # "harvest", "till_soil", "construct_wall", etc.
    parameters: dict
    dependencies: list[int] = []

class Plan(BaseModel):
    plan_id: str
    tasks: list[Task]
```

Example plan for "build a small wooden house":

```json
{
  "plan_id": "house_v1",
  "tasks": [
    {
      "task_id": 1,
      "role_target": "gatherer",
      "skill_call": "harvest",
      "parameters": {"target_block": "oak_log", "quantity": 20},
      "dependencies": []
    },
    {
      "task_id": 2,
      "role_target": "gatherer",
      "skill_call": "harvest",
      "parameters": {"target_block": "stone", "quantity": 10},
      "dependencies": []
    },
    {
      "task_id": 3,
      "role_target": "builder",
      "skill_call": "construct_floor",
      "parameters": {"corner": [100, 64, 100], "size": [7, 7]},
      "dependencies": [1]
    },
    {
      "task_id": 4,
      "role_target": "builder",
      "skill_call": "construct_walls",
      "parameters": {"corner": [100, 64, 100], "size": [7, 4, 7]},
      "dependencies": [3]
    }
  ]
}
```

### Three planning modes

Following Steve-Evolving's pattern:

| Mode | Trigger | Cost | Behavior |
|------|---------|------|----------|
| **Fresh** | New goal, no cached plan | High (full LLM call) | Decompose from scratch |
| **Reuse** | Goal matches cached plan | None | Pull from plan cache, optionally re-parameterize |
| **Replan** | Task failure | Medium | Send error context + partial plan, request fix |

The plan cache is a simple key-value store keyed on goal description embeddings. For common goals ("build wooden house," "expand farm") this drops LLM costs by an order of magnitude after the first run.

### System prompt template

```
You are the central planner for a multi-agent Minecraft village.

Available roles: gatherer, builder, farmer, defender
Available skills: <skill_library_schema_json>
Current world state: <world_state_summary>
Current inventories: <inventory_summary>
Peer agent statuses: <agent_status_summary>

Constraint: Output ONLY a JSON object matching the Plan schema. No prose.

If replanning, the previous plan and the error are below:
<previous_plan>
<error_log>

Goal: <user_goal>
```

The schema is validated on the Python side before any task is dispatched. Invalid plans trigger a retry with the validation error appended to the prompt.

### Resource conflict resolution

The LLM might inadvertently assign two agents to the same crafting table simultaneously. Two safeguards:

1. **Plan-time constraint hints.** The system prompt enumerates singleton resources (crafting table, furnace) and reminds the LLM not to schedule them concurrently.
2. **Runtime arbitration.** The Fabric mod's skill executor maintains a lock table for singleton resources. Conflicting `Interact` calls are queued, not failed.

The second is the real fix. The first is hygiene.


---

## 6. Reward Shaping

Reward design is where projects die quietly. RL agents will exploit any underspecified reward function with breathtaking creativity.

### The reward-hacking failure modes to plan against

| Naive reward | Exploit |
|--------------|---------|
| `+1 if inventory[wood] > 0` | Agent gets one log and stops. |
| `+1 per pickup(wood)` event | Agent drops and re-picks up the same log forever. |
| `+1 per wood in inventory each tick` | Agent hoards forever, never delivers. |
| `+10 per chop_tree action` | Agent chops the wrong species, or chops blocks that drop nothing. |

### The fix: delta inventory + dense progression + step penalty

For the gatherer role:

```
R_t = α · Δ(inv[target_item])      # +α per net unit gained, −α per unit dropped
    + β · Δ(distance_to_target)    # small dense signal toward nearest target block
    + γ                            # constant per-tick penalty, γ < 0
    + δ · 𝟙(quota_met)             # one-shot bonus when LLM-assigned quota hit
```

Suggested initial values:
- `α = +10` per net unit of the assigned target item
- `β = -0.1` per block of distance reduction (note: closer = smaller distance = positive reward via the delta)
- `γ = -0.01` per tick (encourages speed, penalizes loitering)
- `δ = +50` one-shot when the LLM-assigned quantity is reached

### Why this works

- **Delta inventory eliminates the drop-and-pickup exploit.** Dropping the item produces negative reward.
- **Including the LLM's current quota in the observation** is what lets one policy generalize across "gather 10 wood," "gather 30 stone," "gather 5 chicken." The target item is part of the agent's input, not baked into the policy.
- **The per-tick penalty** keeps the agent from learning to wander pleasantly through the woods.
- **The quota bonus** is what signals task completion and triggers the LLM to issue the next task.

### Extending to other roles

| Role | Primary reward signal | Notes |
|------|----------------------|-------|
| **Gatherer** | Delta inventory of target item | As above |
| **Builder** | Block placement matching the target structure pattern | Reward + per correct block, − per incorrect / misplaced |
| **Farmer** | Delta of harvested crops + tilled soil count | Penalize trampling crops |
| **Defender** | Damage dealt to hostile mobs − damage taken | Sparse positive on mob kills |

### Global cooperative reward

Add a small shared component to every agent's reward when village-level milestones complete (house built, farm yielding, no successful mob raids in N ticks). This is what teaches cooperation. Keep it small (~10–20% of total reward magnitude) — if the global signal dominates, you get lazy agents (see next section).

### Open question for Milestone 2

Should the gatherer be penalized for breaking blocks that are not its current target (e.g., chopping non-target trees while pathing through a forest)? Pro: prevents environmental griefing. Con: may discourage opportunistic resource collection. Recommendation: start without the penalty, add it if the gatherer becomes a wood-chipper.

---

## 7. Multi-Agent Failure Modes

Multi-agent RL is not single-agent RL × N. Four specific failure modes will appear without explicit mitigation.

### Non-stationarity

As agent A's policy updates, the environment from agent B's perspective changes. B's policy gradients become unreliable. Training oscillates or diverges.

**Mitigation:** CTDE (centralized training with decentralized execution). The critic during training has access to the global state including all agents' observations and actions. The actors only see their local observations. MAPPO and QMIX both implement this correctly.

**Additional mitigation:** include peer agents' current goals in each agent's observation. This makes peer behavior partially predictable from the agent's own perspective.

### Credit assignment

The village completes a house. Every agent gets +50 reward. Which agent's actions actually mattered? The naive joint reward distributes credit equally and confuses the policy gradients.

**Mitigation:** value decomposition. QMIX's mixing network learns a non-linear factorization of the joint value function that respects monotonicity (individual best response aligns with team best response). MAPPO's centralized critic implicitly learns this through the advantage function.

**Additional mitigation:** hybrid rewards. Each agent gets a large individual reward (delta inventory, blocks placed correctly) plus a small global reward for village milestones. The individual rewards give clear local signal; the global reward shapes cooperation at the margins.

### Lazy agents

If the team is succeeding without me, why bother? An agent learns to stand still and collect the shared reward. VDN (additive value decomposition) is famously vulnerable to this.

**Mitigation:** primarily individual rewards (per above). Plus QMIX's monotonicity constraint, which prevents the joint Q-function from rewarding any single agent's inaction. Plus intrinsic motivation methods (LAIES, ICES) that estimate each agent's causal contribution — overkill for v1 but worth knowing about.

### Communication degeneration

If agents have a learned communication channel, they may converge on a degenerate protocol (e.g., always send the same symbol regardless of context). This does not apply to v1 because there is no learned channel — communication is mediated by the LLM's task DAG. Worth flagging for Phase 5 if/when learned communication is added.

### Centralized critic and the LLM's plan

Open question: should the centralized critic have read-access to the LLM's current DAG state? Arguments for: it gives the critic explicit context for why agents are doing what they're doing, improving credit assignment. Arguments against: it tightly couples the RL training to the LLM's representation, making the LLM hard to swap. Recommendation: start without; add if credit assignment is visibly broken.

---

## 8. Server Performance and Accelerated Training

RL needs millions of transitions. At vanilla 20 TPS, one million ticks takes 14 hours. Acceleration is mandatory.

### Tick warping with Carpet Mod

Carpet's `/tick warp <duration>` command runs the server as fast as the CPU allows for a fixed duration, then returns to 20 TPS. The custom Fabric mod must hook into this to keep observations and actions synchronized.

Realistic speeds on the 9800X3D:
- Single agent, empty world: 500–1000 TPS
- 4 agents, small village area: 200–400 TPS
- 4 agents, complex village with redstone and mobs: 80–150 TPS

This is per-instance. Run 4–6 instances in parallel and aggregate throughput climbs accordingly.

### The Python loop is probably the real bottleneck

This is the gap in Gemini's original analysis. A server running at 500 TPS produces observations faster than a naive Python loop can process them. The actual training throughput is limited by:

1. WebSocket message parsing (JSON decode in Python).
2. PyTorch policy inference per agent per tick.
3. Ray RLlib's experience batching.

Mitigations:
- **Batch policy inference** across agents and across parallel server instances. Never call `.forward()` on one observation at a time.
- **Use msgpack instead of JSON** for the observation channel if profiling shows JSON parsing is dominant. Drop-in replacement, ~5× faster decode.
- **Profile end-to-end early.** The first thing to measure in Milestone 2 is wall-clock transitions per second across the whole pipeline.

### Headless server configuration

```
-XX:+UseZGC
-XX:+ZGenerational
-Xms4G -Xmx4G
--nogui
```

Generational ZGC keeps GC pauses under 1ms, which matters when the RL loop is timing-sensitive. Pin to a specific CPU core via Docker's `cpuset-cpus` to avoid context-switching overhead.

### Docker orchestration

```yaml
# docker-compose.yml sketch
services:
  mc-server-1:
    image: village-fabric-server:latest
    cpuset: "0"
    mem_limit: 5g
    volumes:
      - /mnt/ramdisk/world-snapshot-1:/server/world
    ports:
      - "25565:25565"  # Minecraft
      - "8765:8765"    # WebSocket
  mc-server-2:
    image: village-fabric-server:latest
    cpuset: "1"
    mem_limit: 5g
    volumes:
      - /mnt/ramdisk/world-snapshot-2:/server/world
    ports:
      - "25566:25565"
      - "8766:8765"
  # ... repeat for 4–6 instances
```

Each server gets its own CPU core, its own RAM-backed world directory, and its own port pair.

---

## 9. Episode Reset and Determinism (gap filled)

Gemini flagged this as an open question and did not answer it. It is critical enough to lock in before Milestone 1.

### The reset problem

At the end of each training episode, the world must return to a known initial state. The naive approach — re-generate chunks via worldgen — takes 10–60 seconds per reset. At 1000 episodes per training run, that is a quarter of total wall-clock time spent on resets.

### Snapshot-and-restore

The solution is to pre-generate one or more known-good world states and restore by directory copy:

1. **At setup time:** generate the world once, place initial structures, save it to disk.
2. **At training time:** mount that saved world to a RAM disk (`tmpfs`). Each Docker server gets a writable overlay.
3. **At episode end:** unmount the overlay, discard it, mount a fresh one. World is back to initial state in <500ms.

On Linux this is `tmpfs` + `overlayfs`. Each server instance gets its own overlay; the underlying snapshot is shared read-only.

### Determinism

For reproducible debugging, every training run should be seedable. Three sources of non-determinism must be controlled:

1. **Minecraft world seed** — fixed per training run.
2. **Mob spawning RNG** — Carpet Mod can fix this.
3. **Python/PyTorch RNG** — `torch.manual_seed`, `numpy.random.seed`, RLlib's `seed` config.

With all three fixed, a training run should produce bit-identical results across re-runs. This is invaluable when debugging "agent_3 did something weird at tick 84291."

---

## 10. Debugging and Observability (gap filled)

The single biggest predictor of project success is the quality of the debugging infrastructure built in Milestone 1. Build it before you need it.

### What to log per episode

- Every observation, every action, every reward, indexed by (agent_id, tick).
- LLM plan state transitions (plan created, task dispatched, task completed/failed, replan triggered).
- Server-side events (chunks loaded, mobs spawned, blocks placed/broken).
- Policy network gradients and activation statistics (RLlib does this by default with TensorBoard).

Storage: SQLite for the structured event log, Parquet for the bulk observation/action tensors. Both compress well.

### Replay

Given a (seed, run_id, episode_id), the dashboard should be able to:
- Replay the episode at 1×, 4×, or 0.25× speed.
- Scrub to a specific tick.
- Show each agent's observation tensor, action distribution, and reward at the scrubbed point.
- Overlay the LLM's DAG state at that tick.

This is non-negotiable. Without replay, debugging is just guessing.

### Dashboard (Tauri)

Suggested panel layout:

| Panel | Content |
|-------|---------|
| **Top-down map** | Village layout, agent positions, current LLM-assigned target blocks |
| **Agent strip** | One row per agent: role, current skill, current goal, inventory bar, current reward |
| **LLM plan view** | DAG visualization, completed/in-progress/pending tasks |
| **Reward curves** | Per-agent and aggregate reward over time |
| **Server stats** | TPS, memory, WebSocket message rate |
| **Replay controls** | Play/pause/scrub, episode selector |

Reuse the Tauri 2 + React 19 + Rust stack already in muscle memory.

### Determinism harness

Build a small test that runs the same (seed, plan) twice and asserts bit-identical observation streams. Run it in CI. If determinism breaks, you want to know immediately, not three weeks into a training run.

---

## 11. Bedrock Compatibility via Geyser

Deferred to Phase 5+. The reasoning:

### What Geyser does and does not do

Geyser is a protocol translator that sits between Bedrock clients (UDP) and Java servers (TCP), translating packets in real time. Floodgate handles the authentication mapping. Together they make a Java server appear as a Bedrock-compatible server to Bedrock clients.

What works: vanilla blocks, vanilla items, vanilla entities, vanilla UI.

What breaks:
- **Custom modded entities** render as generic placeholders or not at all.
- **Custom client-side packets** are dropped — Bedrock cannot parse them.
- **Movement around edge-case blocks** (bamboo, scaffolding) desyncs.
- **Some inventory UIs** (book editing, complex anvil interactions) misbehave.

### Implications for the village architecture

The WebSocket bridge between the Fabric mod and Python is completely outside the Minecraft protocol. Geyser does not touch it. That layer is safe.

The risk is in the *visual* representation of agents to Bedrock players. If the project ships custom entity models for the AI villagers, Bedrock players will see broken placeholders. Mitigations:

1. **Use vanilla entity types only.** Render AI agents as vanilla villagers, vanilla iron golems, etc. Use nametags and skins for differentiation.
2. **No custom client-side UI.** All status information goes through chat or the Tauri dashboard, never through custom client packets.
3. **Avoid modded blocks.** The village uses vanilla blocks only.

If those three constraints are followed throughout development, adding Geyser in Phase 5 becomes a low-risk drop-in. If they are violated, retrofitting Bedrock support will require either rewriting large portions of the mod or shipping a custom Bedrock resource pack.

### Recommendation

Treat Geyser compatibility as an architectural constraint from day one (vanilla-only entities, no custom client packets) but do not actually ship Geyser until Phase 5. This keeps the option open without the development overhead.

---

## 12. Reference Repositories

Ranked by relevance to this specific project.

| Repo | Why it matters |
|------|---------------|
| [Replicable-MARL/MARLlib](https://github.com/Replicable-MARL/MARLlib) | The MAPPO/QMIX implementation you will actually use. Read the YAML configs and the policy mapping examples. |
| [mindcraft-bots/mindcraft](https://github.com/mindcraft-bots/mindcraft) | Closest existing multi-agent LLM-driven Minecraft project. Read the agent coordination logic. |
| [zju-vipa/odyssey](https://github.com/zju-vipa/odyssey) | 40 primitive + 183 compositional skills. The skill library design is the reference. |
| [cnsdqd-dyb/VillagerAgent](https://github.com/cnsdqd-dyb/VillagerAgent) | Graph-based multi-agent LLM framework. Task decomposer and state manager patterns. |
| [Dan-Mizu/Not-Enough-Management](https://github.com/Dan-Mizu/Not-Enough-Management/) | Clean blueprint for server-side Fabric mod with WebSocket JSON-RPC. The protocol layer to copy. |
| [MineDojo/Voyager](https://github.com/MineDojo/Voyager) | LLM prompting patterns for structured Minecraft output. Read the system prompts. |
| [PrismarineJS/mineflayer](https://github.com/prismarinejs/mineflayer) | Even though you are not using it, the source is the canonical reference for Minecraft physics, pathfinding heuristics, and inventory state. |
| [openbmb/agentverse](https://github.com/openbmb/agentverse) | Multi-agent simulation framework. Useful patterns for emergent social behavior. |
| [Farama-Foundation/PettingZoo](https://github.com/Farama-Foundation/PettingZoo) | The env API you will implement. Study the Parallel API examples. |
| [gnembon/fabric-carpet](https://github.com/gnembon/fabric-carpet) | Carpet Mod source. Read the tick warp implementation and the fake player API. |

---

## Revised Milestone Plan

The original milestone plan was directionally correct but skipped foundational infrastructure (snapshots, debugging, determinism). Revised:

### Milestone 0: Infrastructure foundation (week 1)

- Fabric mod scaffold with mixins into the server tick loop.
- Carpet Mod integrated, `/tick warp` working.
- WebSocket + JSON-RPC server inside the mod, smoke-tested from a Python client.
- Baritone embedded, exposed via the mod API.
- RAM-disk world snapshot + overlay system working for fast resets.
- Docker setup for 4 parallel server instances.

**Exit criteria:** Python can connect to 4 servers in parallel, send `Navigate` actions, receive observations, trigger resets in <1s.

### Milestone 1: The data sandbox (week 2)

- PettingZoo environment wrapping the WebSocket bridge.
- Full observation and action space implemented and validated.
- Episode logging to SQLite + Parquet.
- Determinism harness in CI.
- Tauri dashboard skeleton showing live agent positions and observations.

**Exit criteria:** A scripted policy (no learning yet) can run for 10k ticks, log everything, and the dashboard renders state correctly.

### Milestone 2: Single-agent gatherer (weeks 3–6)

- One gatherer agent, independent PPO via Ray RLlib (not MARLlib yet — start simple).
- Delta-inventory reward implemented.
- Trains to reliably harvest 20 wood blocks on a fresh world.

**Exit criteria:** Gatherer policy achieves >80% success rate on the harvest task over 100 evaluation episodes.

### Milestone 3: Multi-agent + roles (months 2–3)

- Add builder and farmer roles, each with its own policy.
- Migrate to MARLlib + MAPPO with centralized critic.
- LLM planner integrated, emits typed JSON DAGs via the orchestrator CLI.
- Three planning modes (fresh, reuse, replan) implemented.

**Exit criteria:** Three agents can collaboratively complete "build a small wooden house" within 10 minutes of wall-clock training-time gameplay.

### Milestone 4: Coordination and defense (month 4)

- Add defender role.
- Hybrid reward structure (individual + small global).
- Mob threat handling, basic combat policy.
- LLM replanning on agent death.

**Exit criteria:** Village survives a night with mob spawning enabled, defenders respond to threats without LLM intervention.

### Milestone 5: Persistence and Bedrock (month 5+)

- 24/7 server runs with checkpointing.
- Geyser/Floodgate integration.
- Tauri dashboard production polish.
- Long-horizon experiments (week-long village evolution).

---

## Risks and Showstoppers

### High-severity risks

**LLM/RL synchronization fallacy.** If the planner is accidentally placed in the inner loop, the system freezes or bankrupts the API budget. Mitigation: hard-code the planner as event-driven only. Add a wall-clock budget per episode that caps LLM calls.

**Reward hacking propagation.** The first reward function will be wrong. The agent will exploit it. Mitigation: every reward function gets a 1000-episode exploitation hunt before being trusted. Log inventory deltas, position changes, and skill calls during the hunt. Look for repetitive patterns.

**Python loop bottleneck.** Tick-warp gets the server to 500 TPS but Python can only process 50 transitions/second. Mitigation: profile end-to-end in Milestone 1. Batch inference. msgpack if needed. Async WebSocket consumption.

### Medium-severity risks

**Centralized critic memory explosion.** The critic sees all 4 agents' observations concatenated. Tensor sizes balloon. Mitigation: cap peer-agent observation slots, use attention-based aggregation if it becomes a problem.

**Baritone variable execution time.** Navigate actions take 50ms to 5s. Mitigation: multi-tick skill abstraction with progress reporting (per the action space section).

**Plan cache staleness.** A cached plan for "build wooden house" assumes the gatherer has access to trees. After 10 hours of training the trees are gone. Mitigation: world-state preconditions in the cache key, not just goal description.

### Showstoppers (would force a rewrite)

**Fabric API breakage between Minecraft versions.** Mitigation: pin a specific Minecraft version (1.21.x). Do not chase the latest snapshot.

**Determinism loss.** If training runs become non-reproducible, debugging becomes impossible. Mitigation: determinism harness in CI from day one.

---

## Reading List (priority-ranked)

Read these in order. Each one informs the next architectural decision.

1. **[Voyager paper](https://voyager.minedojo.org/)** — the canonical LLM-as-planner architecture, even though we're not copying it directly.
2. **[Odyssey repo and docs](https://github.com/zju-vipa/odyssey)** — closest existing parallel planning-acting architecture.
3. **[MARLlib paper](https://www.jmlr.org/papers/volume24/23-0378/23-0378.pdf)** and **docs** — required for understanding the framework you're using.
4. **[QMIX paper](https://arxiv.org/abs/1803.11485)** and the value decomposition lineage — the theoretical foundation for credit assignment.
5. **[MINDCraft / MineCollab paper](https://mindcraft-minecollab.github.io/)** — action-by-action multi-agent LLM collaboration.
6. **[GITM paper](https://arxiv.org/abs/2305.17144)** — validates symbolic over visual observation.
7. **[Plan4MC paper](https://arxiv.org/abs/2303.16563)** — skill graph + intrinsic RL hierarchy.
8. **[Not-Enough-Management source](https://github.com/Dan-Mizu/Not-Enough-Management/)** — the Fabric mod WebSocket blueprint.
9. **[PettingZoo Parallel API docs](https://pettingzoo.farama.org/api/parallel/)** — the env interface you're implementing.
10. **[Geyser limitations wiki](https://geysermc.org/wiki/geyser/current-limitations/)** — read before designing any custom packets or entities.

---

## Open Questions for the Next Synthesis Pass

These are the items to feed into the Kimi swarm's results when they return:

1. **Episode reset performance** — has anyone in the open-source world already solved snapshot+overlay for Fabric? Any prior art beats reinventing this.
2. **Python inference loop optimization** — concrete profiling examples from existing Minecraft RL projects, not just "use batching."
3. **Baritone embedding patterns** — has anyone exposed Baritone via an RPC interface? Saving the wrapper code would shave a week.
4. **Defender role observation needs** — symbolic observations may not be sufficient for fast combat. Survey what XLand and similar combat-focused RL projects used.
5. **Deterministic replay in MARL** — concrete patterns for reproducible multi-agent training runs, especially with parallel rollouts.
6. **Orchestrator CLI integration shape** — should the LLM planner live inside the existing orchestrator CLI as a tool, or run as a separate service the orchestrator can call? Affects deployment and dashboard architecture.

---

*End of report.*