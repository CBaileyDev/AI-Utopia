## 4. Multi-Agent RL Frameworks

Selecting a Multi-Agent Reinforcement Learning (MARL) framework for a heterogeneous-agent Minecraft village is not merely a library choice — it constrains the observation space design, the debugging workflow, the rate at which curriculum stages can be iterated, and ultimately whether the project ships or stalls. This chapter evaluates seven frameworks and libraries against the constraints of a solo developer running 8–16 concurrent agents on a single workstation (AMD Ryzen 9800X3D, 64 GB DDR5, NVIDIA RTX 4080 16 GB). The analysis proceeds in four parts: a comparative review of each framework (Section 4.1), a hardware-specific scalability study (Section 4.2), a debugging and developer-experience assessment (Section 4.3), and a concrete stack recommendation with an algorithm-selection flowchart (Section 4.4).

### 4.1 Framework Comparison

The evaluation covers PettingZoo (the environment standard), Ray RLlib (the training engine), MARLlib (a stalled wrapper), EPyMARL (a lightweight alternative), HARL (the 2024 algorithmic frontier for heterogeneity), the HeMAC benchmark (the first standardized heterogeneous testbed), and BenchMARL (a Facebook Research benchmarking harness). Each is scored on heterogeneous-agent support, scalability on the target workstation, debugging infrastructure, learning curve, maintenance status, and algorithm variety.

#### 4.1.1 PettingZoo: Parallel API, Heterogeneous Agent Support, AEC vs. Parallel Tradeoffs

PettingZoo is not a training framework — it is an environment Application Programming Interface (API) standard maintained by the Farama Foundation (the same organization that stewards Gymnasium). Its role in the stack is analogous to Gymnasium in single-agent RL: it defines how environments expose observations, actions, and rewards to any downstream trainer. PettingZoo offers two APIs: the Agent Environment Cycle (AEC) API, which steps one agent at a time and is suited for turn-based or strictly sequential interactions, and the Parallel API, which steps all agents simultaneously and is the natural choice for a cooperative village-building scenario where gatherers, builders, farmers, and defenders act concurrently [^25^].

The Parallel API explicitly supports different observation and action spaces between agents, a capability that is non-negotiable for the village use case:

> "This API is based around the paradigm of Partially Observable Stochastic Games (POSGs) and the details are similar to RLlib's MultiAgent environment specification, except we allow for different observation and action spaces between the agents." [^25^]

The `observation_space(agent)` and `action_space(agent)` methods are invoked per-agent at initialization, enabling completely different representations: gatherers receive terrain-view patches (e.g., $\mathbb{R}^{32 \times 32 \times 4}$), builders receive local block-context windows ($\mathbb{R}^{16 \times 16 \times 8}$), farmers receive crop-state grids ($\mathbb{R}^{16 \times 16 \times 5}$), and defenders receive mob-detection views ($\mathbb{R}^{32 \times 32 \times 3}$). Action spaces follow the same pattern — gatherers use a 9-element `Discrete` space (8 movement directions plus collect), while builders use a `MultiDiscrete` tensor for parameterized block placement [^137^].

Creating a custom PettingZoo environment requires implementing approximately five methods (`reset`, `step`, `observation_space`, `action_space`, `render`). The Farama Foundation provides well-structured tutorials, and the stable API means the environment definition will not break when the training framework updates [^137^]. PettingZoo integrates with RLlib, Stable Baselines3, TorchRL, and BenchMARL, making it the unambiguous environment-layer choice.

#### 4.1.2 Ray RLlib Multi-Agent: Policy Mapping, Fractional GPU, Debugging Infrastructure

Ray RLlib is the multi-agent reinforcement learning library within the Anyscale Ray distributed computing framework. It is actively maintained, industry-proven (deployed at Shopify, Uber, and Ant Group), and provides the most mature multi-agent training API available to a solo developer [^23^]. RLlib's new API (Ray 2.x+) replaces the legacy `ModelV2` stack with `RLModule` and `MultiRLModuleSpec`, which cleanly separate policy architectures per agent type while allowing variable sharing between policies [^11^].

The `policy_mapping_fn` is the central abstraction for heterogeneous agents: it maps each agent ID to a named policy at runtime. For the village scenario, this means `gatherer_0` through `gatherer_3` all route to a single `gatherer_policy`, while `builder_0` through `builder_2` route to `builder_policy`, and so on. Per-policy algorithm overrides allow different learning rates, clip parameters, and network architectures for each role — builders might use a lower learning rate (1e-4) than gatherers (3e-4) because their action space is more complex and their feedback is sparser [^11^]:

```python
config.multi_agent(
    policy_mapping_fn=lambda agent_id, episode, **kwargs: (
        "gatherer_policy" if "gatherer" in agent_id
        else "builder_policy" if "builder" in agent_id
        else "farmer_policy" if "farmer" in agent_id
        else "defender_policy"
    ),
    algorithm_config_overrides_per_module={
        "gatherer_policy": PPOConfig.overrides(lr=3e-4),
        "builder_policy": PPOConfig.overrides(lr=1e-4),
    },
)
```

RLlib supports fractional GPU allocation via `num_gpus_per_learner=0.5`, which leaves headroom for environment rendering or auxiliary models on the same RTX 4080 [^141^]. The multi-GPU training stack introduced in Ray 2.5+ is overkill for a single-GPU setup but demonstrates the architecture's upward scalability [^142^]. A notable limitation is that RLlib does not yet vectorize multi-agent environments [^11^], which caps sample throughput compared to vectorized alternatives such as VMAS or BenchMARL. For 8–16 agents in a Minecraft environment where simulation is already bottlenecked by the Java server, this limitation is rarely the binding constraint.

The learning curve is steep: a solo developer should budget 2–3 days to understand `policy_mapping_fn`, `MultiRLModuleSpec`, `AlgorithmConfig`, and Ray resource allocation before productive experimentation begins. However, once mastered, the configuration system is powerful and reproducible.

#### 4.1.3 MARLlib: Unmaintained Since Late 2023, Verified Benchmark Claims but Not Recommended

MARLlib is a wrapper around Ray RLlib that exposes 18 MARL algorithms across 15+ environments through a unified API. Its benchmark paper claims training efficiency of 3 minutes 29 seconds for 1 million steps on SMAC's MMM2 map with 5 Ray workers [^3^]. This claim is verified on comparable hardware (RTX A6000, Threadripper PRO 5945WX), but it comes with important caveats: the benchmark uses 11.2 GB of system RAM and 5 GB of VRAM [^3^], and more critically, it was measured on homogeneous SMAC agents with parameter sharing. Heterogeneous agents with separate policies multiply both memory usage and wall-clock time.

The decisive factor against MARLlib is maintenance status. The last release (v1.0.3) was in April 2023, the last meaningful code update was September 2023, and the last documentation update was May 2024 [^197^]. The project has 47 open issues with zero recent pull request merges. RLlib's API has evolved significantly since MARLlib's last release — the `RLModule` stack, fractional GPU support, and new multi-agent configuration patterns are not accessible through MARLlib's abstraction layer. For a solo developer in 2025, MARLlib adds a dependency layer with functional loss rather than gain.

#### 4.1.4 EPyMARL: Lightweight Alternative, 9 Algorithms, Gymnasium Update July 2024

EPyMARL (Extended PyMARL) is a University of Edinburgh project that extended the original PyMARL codebase with 5 additional algorithms, environment support beyond SMAC, and — crucially for heterogeneous agents — configurable parameter sharing. The original PyMARL assumed all agents shared parameters and identical observation shapes; EPyMARL removes both constraints [^26^]. The July 2024 v2.0.0 release migrated from the deprecated Gym 0.21 API to Gymnasium, added native PettingZoo and VMAS support, and introduced Weights & Biases logging [^198^].

EPyMARL uses Python multiprocessing rather than Ray. Its training throughput is approximately 1.5x slower than MARLlib (5:29 vs. 3:29 for 1M steps at 5 workers), but it uses less than half the memory (8.4 GB vs. 11.2 GB RAM; 2.2 GB vs. 5.0 GB VRAM) [^3^]. For the RTX 4080, this lower GPU footprint actually leaves more memory for larger policy networks. EPyMARL supports 9 algorithms: IQL, VDN, QMIX, QTRAN, IA2C, IPPO, MADDPG, MAA2C, and MAPPO, plus Pareto-AC [^30^].

The project is backed by published benchmarks (NeurIPS 2021 Datasets & Benchmarks track) and has 24 open issues that are actively triaged [^198^]. EPyMARL is a good choice for researchers who want to hack algorithms in a simpler, non-Ray codebase. It is not ideal for production systems that need to scale beyond 16 agents or leverage the latest RLlib features.

#### 4.1.5 HARL (2024): HAPPO/HATRPO with Proven Monotonic Improvement for True Heterogeneity

HARL (Heterogeneous-Agent Reinforcement Learning) is a family of algorithms — HAPPO, HATRPO, HAA2C, HADDPG, HATD3, HAD3QN, and HASAC — explicitly designed for heterogeneous agents. The accompanying Journal of Machine Learning Research (JMLR) 2024 paper proves that all HARL algorithms enjoy monotonic improvement guarantees and convergence to Nash equilibrium under the HAML (Heterogeneous-Agent Mirror Learning) framework [^27^]. This is the only algorithm family in this survey with theoretical guarantees specifically for true heterogeneity.

The key innovation is a sequential update scheme: agents update one at a time in random order, with each agent's update conditioned on the already-updated policies of earlier agents. A multi-agent advantage decomposition lemma enables correct credit assignment without parameter sharing [^27^]. Empirically, the paper demonstrates that on a 17-agent Humanoid control task where each agent controls a dissimilar body part, MAPPO fails completely while HAPPO succeeds [^27^]. This result is directly relevant to the village scenario where gatherers, builders, farmers, and defenders have fundamentally different action spaces and physical capabilities.

The computational overhead of sequential updates is modest: HAPPO is approximately 2x slower per update step than MAPPO on MAMuJoCo tasks (8.6 s vs. 4.9 s on HalfCheetah 2x3; 76.7 s vs. 71.3 s on Humanoid 17x1), but improved sample efficiency often means fewer total updates are required [^27^]. The codebase has 913 GitHub stars, was last updated in October 2024, and has 131 forks but only two core contributors — characteristic of an academic project rather than an industry-backed framework [^88^].

#### 4.1.6 HeMAC Benchmark (2025): Standardized Heterogeneous Testbed, IPPO>MAPPO Findings

HeMAC (Heterogeneous Multi-Agent Cooperation), published at the European Conference on Artificial Intelligence (ECAI) 2025, provides the first standardized benchmark specifically designed for heterogeneous MARL [^1^]. Its findings have direct implications for algorithm selection in the village scenario. On the "Complex Fleet" task — a high-heterogeneity scenario with dissimilar agent capabilities — the results show IPPO outperforming MAPPO, and QMIX failing entirely:

> "While agents using advanced algorithms such as MAPPO excel in simpler cooperative tasks, their performance declines as heterogeneity increases, with IPPO outperforming them in highly diverse scenarios." [^1^]

The QMIX failure is particularly instructive: QMIX assumes a shared action-value space and homogeneous agent structure, which breaks when agents have fundamentally different observation and action dimensions [^1^]. This validates the decision to avoid value-decomposition methods for the village scenario and instead start with independent learning approaches that respect per-role action spaces.

#### 4.1.7 BenchMARL, Newer Frameworks, and When to Go Direct PyTorch

BenchMARL (Facebook Research, 2024–present) is a benchmarking library built on TorchRL that provides rigorous, reproducible MARL algorithm comparison [^79^]. Its unique "agent grouping" mechanism allows agents of the same type to benefit from vectorized training while heterogeneous agents keep separate data entries [^79^]. BenchMARL is actively maintained (last commit February 2026) and has 623 GitHub stars [^215^]. It supports 10+ algorithms (MAPPO, IPPO, MADDPG, IDDPG, MASAC, ISAC, IQL, VDN, QMIX) and integrates with VMAS, which can run "tens of thousands of parallel environments on accelerated hardware" [^164^]. BenchMARL is best used as an evaluation harness to rigorously compare IPPO vs. MAPPO vs. HAPPO on the village environment, rather than as the primary training framework.

Direct PyTorch implementation becomes the right choice only if framework limitations dominate development time. Valid reasons to go custom include: implementing a novel algorithm not available in any framework (e.g., transformer-based policies with Minecraft-specific tokenization), needing full control over the training loop for research purposes, or finding that debugging distributed framework issues takes longer than writing the training code from scratch. The TorchRL tutorial provides a complete MAPPO/IPPO training loop using `MultiAgentMLP` and `ClipPPOLoss` that can be adapted for heterogeneous agents with `share_params=False` [^105^]. For a solo developer, however, the recommended path is to start with a framework and go custom only when a concrete limitation is hit.

**Table 1: Framework Comparison for Heterogeneous MARL (4 Roles, 8–16 Agents, Single Workstation)**

| Framework | Hetero. Support | Maintenance | Algorithms | Learning Curve | VRAM (5 workers) | Solo-Dev Score |
|-----------|----------------|-------------|------------|----------------|-----------------|----------------|
| PettingZoo (env API) | Excellent [^25^] | Active (Farama) | N/A | Moderate [^137^] | N/A | 10/10 (required) |
| Ray RLlib (trainer) | Excellent [^11^] | Very active [^23^] | 30+ | Steep | ~5 GB [^3^] | 9/10 |
| HARL (HAPPO) | Excellent (designed for it) [^27^] | Moderate [^88^] | 10 | Moderate | ~3 GB (est.) | 8/10 |
| BenchMARL | Excellent [^79^] | Very active [^215^] | 10+ | Moderate-steep | Varies | 7/10 |
| EPyMARL | Good [^26^] | Moderate [^198^] | 9 | Moderate | 2.2 GB [^3^] | 6/10 |
| Direct PyTorch/TorchRL | Full control [^105^] | Self-maintained | Unlimited | Steep | User-controlled | 5/10 |
| MARLlib | Good (via RLlib) [^3^] | Stalled [^197^] | 18 | Steep | 5.0 GB [^3^] | 3/10 |
| MAPPO Official | Limited [^87^] | Minimal [^87^] | 1 | Moderate | 2.2 GB [^3^] | 3/10 (reference only) |

The scoring reflects the solo-developer constraint. Ray RLlib earns the highest training-framework score because its documentation maturity, community support, and debugging infrastructure reduce the time from "I want to try this" to "I can see it learning." HARL scores highly on algorithmic fit but lower on ecosystem breadth. MARLlib's stall since late 2023 makes it a poor investment for a new project, despite its verified speed claims [^3^] [^197^]. BenchMARL excels as a benchmarking tool but requires learning TorchRL and Hydra configuration, adding overhead that may not pay off during initial prototyping.

### 4.2 Scalability Analysis

Scalability for this project has two dimensions: whether the RTX 4080 can hold the policy networks for 8–16 heterogeneous agents in GPU memory, and whether the 9800X3D can run enough parallel environment instances to feed the learner with sufficient sample throughput.

#### 4.2.1 RTX 4080 16 GB Capacity: 8–16 Agents with Fractional GPU Allocation

The RTX 4080's 16 GB VRAM is the binding GPU constraint. Four separate policies (one per agent type) with small convolutional neural network (CNN) backbones and fully-connected heads fit comfortably within 6–8 GB. Each policy network for a role with a $32 \times 32 \times 4$ observation and 9 discrete actions requires roughly 2–3 million parameters (a 3-layer CNN with 32–64 filters plus a 256-unit Multi-Layer Perceptron (MLP) head), which at float32 precision occupies approximately 40–50 MB of GPU memory for weights and 200–400 MB during training when gradients, optimizer states, and replay buffers are included. With 4 policies, this totals 1–2 GB for network weights and 4–6 GB during active training.

RLlib's fractional GPU allocation (`num_gpus_per_learner=0.5` or `0.25`) allows the learner to share the GPU with environment rendering or auxiliary processes [^141^]. For the village scenario, where the primary GPU consumer is the policy learner (Minecraft simulation runs on the CPU), allocating the full GPU to the learner is appropriate, but fractional allocation provides flexibility if a vision encoder or large language model inference is later co-located on the same device.

The main risk is not fitting 4 policies but rather the lack of multi-agent environment vectorization in RLlib [^11^]. Without vectorization, each environment rollout executes sequentially within a worker, which means the GPU spends more time waiting for CPU-bound environment steps than it would with batched vectorized rollouts. This GPU underutilization is acceptable when the environment itself (the Minecraft server) is the bottleneck, but it becomes a concern if the environment is lightweight and the policy is large.

#### 4.2.2 Ryzen 9800X3D: Parallel Environment Rollouts with Ray Tuning

The Ryzen 9800X3D (8 cores, 16 threads, 96 MB L3 cache) is well-suited for running 4–6 parallel Minecraft server instances while leaving cores for RLlib's rollout workers. Chapter 8 (Performance Engineering) established that 4–6 parallel Fabric server instances at 20 ticks per second (TPS) is realistic on this processor. Each RLlib rollout worker can manage one or two environment instances; with `num_rollout_workers=4` and `num_envs_per_worker=2`, the system runs 8 concurrent environment instances across the 16 threads.

Ray's resource scheduler automatically distributes workers across CPU cores. The 9800X3D's large L3 cache benefits the frequent context switches between environment simulation and policy inference. Setting `num_cpus_per_worker=2` ensures each worker has sufficient CPU headroom for both the Minecraft server process and the Python environment wrapper. The 64 GB of DDR5 RAM accommodates the combined footprint: 4–6 Minecraft instances at 2–3 GB each (12–18 GB), RLlib worker memory (5–10 GB), OS and background processes (4–8 GB), leaving 30+ GB of headroom.

#### 4.2.3 Realistic Throughput Expectations: Samples/Second with 4–8 Parallel Environments

Sample throughput in this setup is dominated by Minecraft server simulation time, not RL framework overhead. A single Minecraft Fabric server running at 20 TPS produces one observation-action-reward tuple per agent per 50 ms. With 12 agents (4 gatherers + 3 builders + 3 farmers + 2 defenders) across 8 parallel environments, the theoretical maximum sample generation rate is $12 \times 8 \times 20 = 1{,}920$ agent-steps per second. In practice, environment reset overhead, episode truncation, and the Python-Java bridge latency reduce this by 30–50%, yielding an effective throughput of approximately 1,000–1,300 agent-steps per second.

The RL framework's training throughput must match or exceed this sample generation rate to avoid a learner-starvation bottleneck. With a `train_batch_size` of 4,096 and 1,000 agent-steps per second, the learner performs one gradient update every ~4 seconds. This is a healthy ratio: the learner is not idling, nor is it so fast that it overfits to a small batch. If sample generation is slower than expected, `train_batch_size` can be reduced to 2,048, or `num_envs_per_worker` can be increased to 4 if RAM permits.

**Table 2: Scalability Analysis on Target Workstation (9800X3D + 64 GB RAM + RTX 4080 16 GB)**

| Component | Resource | Allocation | Headroom | Bottleneck? |
|-----------|----------|------------|----------|-------------|
| GPU (RTX 4080) | 16 GB VRAM | 6–8 GB (4 policies + training state) [^141^] | 8–10 GB | No |
| GPU compute | 48 SM, 2.5 GHz | ~40% utilized (non-vectorized envs) [^11^] | Significant | Partial (vectorization gap) |
| CPU cores | 8C / 16T | 4 workers × 2 cores + 6 Minecraft instances | 2–4 threads | No |
| System RAM | 64 GB | 12–18 GB (servers) + 10 GB (RLlib) + 6 GB (OS) | 30+ GB | No |
| Env. throughput | 1,920 steps/s theoretical | 1,000–1,300 steps/s effective | — | Yes (Java bridge) |
| Learner throughput | Configurable | 1 update / 4s at batch 4096 | — | No |
| Storage (NVMe) | 1 TB+ recommended | Checkpoints: ~200 MB each | — | No |

The Java-Python bridge is the binding throughput bottleneck, not the RL framework or the GPU. This is a crucial observation: optimizing the Minecraft server side (chunk loading, entity count, tick rate) yields more training speedup than switching between RL frameworks. MARLlib's 3:29 for 1M steps [^3^] is irrelevant if the environment itself takes 30 minutes to produce those samples. A solo developer should therefore invest engineering effort in bridge efficiency and server optimization before worrying about framework-level throughput differences.

### 4.3 Debugging and Developer Experience

Debugging MARL systems is categorically harder than debugging single-agent RL: errors can arise from the environment wrapper, the policy network, the reward function, inter-agent coordination, or the distributed training infrastructure, and symptoms (diverging losses, flat reward curves, lazy agents) have multiple possible causes. The framework choice directly affects how quickly a solo developer can isolate and fix these issues.

#### 4.3.1 TensorBoard Integration, Episode Replay, Checkpoint Management

RLlib provides TensorBoard logging out of the box via `PPOConfig().debugging(log_level="INFO")`, with per-policy loss curves, value function estimates, KL divergence, entropy, and custom metrics. Episode replay is supported through RLlib's `evaluate()` method with checkpoint restoration, which allows visual inspection of agent behavior at arbitrary training iterations. Checkpoint management is built into `tune.run()` with configurable frequency (`checkpoint_freq=50`) and automatic best-checkpoint tracking [^11^].

EPyMARL offers Sacred experiment management (with optional MongoDB logging) and, since the July 2024 update, Weights & Biases integration with a simple plotting script for run data [^198^]. HARL provides TensorBoard logging and episode statistics tracking but has less mature tooling overall [^88^]. BenchMARL integrates with Weights & Biases and provides automatic experiment checkpointing through its Hydra configuration system [^79^].

#### 4.3.2 Why RLlib Wins for Solo Developers: Documentation Maturity and Community Support

The decisive advantage of RLlib for a solo developer is not any single technical feature but the cumulative effect of documentation breadth, community size, and error-searchability. RLlib's multi-agent documentation includes complete code examples, API references, and migration guides from the legacy to the new API [^11^]. The Anyscale blog publishes regular deep-dives on scaling patterns [^142^]. The Ray community on GitHub Discussions and Stack Overflow has thousands of answered questions, meaning that most error messages are searchable.

By contrast, HARL's two-contributor academic team cannot provide the same level of community support [^88^]. EPyMARL's smaller user base means fewer answered questions online [^198^]. MARLlib's stalled state means that issues related to newer Ray versions receive no responses [^197^]. For a solo developer who will encounter novel errors at every stage of integration, the ability to find a Stack Overflow answer or a GitHub issue describing the exact same traceback is worth weeks of debugging time.

The one significant caveat is distributed debugging: Ray worker errors propagate poorly, and stack traces can be truncated across process boundaries. The mitigation is to start with `num_rollout_workers=0` (local mode) during initial environment debugging, then incrementally scale workers once the environment and policy mapping are stable.

### 4.4 Recommendation

#### 4.4.1 Stack Verdict: PettingZoo + Ray RLlib with IPPO, Graduate to MAPPO/QMIX as Needed

The recommended stack for the heterogeneous Minecraft village project is **PettingZoo (environment API) + Ray RLlib (training framework) + IPPO (starting algorithm)**. This combination is the only one that simultaneously satisfies all hard constraints: heterogeneous observation and action spaces, active maintenance, mature debugging infrastructure, and sufficient scalability on the target hardware.

The algorithm choice follows directly from the HeMAC benchmark findings [^1^]: IPPO (Independent Proximal Policy Optimization) — one PPO instance per agent type — is the correct starting point because it is the simplest algorithm that respects per-role action spaces and has empirically outperformed MAPPO on highly heterogeneous tasks. Each agent type learns its own policy with its own observation space; no observation padding or forced action-space sharing is required. The "independent" label is slightly misleading: agents train within the same environment and experience each other's state transitions, so coordination can emerge through the shared environment dynamics even without an explicit centralized critic.

If IPPO fails due to poor cross-role coordination (e.g., builders do not wait for wood delivery from gatherers), the escalation path is: (1) MAPPO with a centralized critic that conditions on concatenated global state, (2) HAPPO from the HARL framework for sequential updates with theoretical convergence guarantees [^27^], (3) COMA-style counterfactual credit assignment if lazy agents emerge [^305^]. Value decomposition methods (QMIX, VDN) should be avoided for this use case because they assume homogeneous agents and shared action-value spaces, and HeMAC confirms QMIX fails entirely under high heterogeneity [^1^].

The starter code structure consists of three files: `village_env.py` (the PettingZoo environment with per-role `observation_space` and `action_space` methods), `train_village.py` (the RLlib configuration with `policy_mapping_fn` and `MultiRLModuleSpec`), and `eval_village.py` (checkpoint loading, episode rendering, and metric logging). This structure mirrors the code examples in Sections 4.1.1 and 4.1.2 and can be bootstrapped from RLlib's PettingZoo integration tutorial [^12^].

#### 4.4.2 Starter Code Structure and Algorithm Selection Flowchart

The algorithm selection process is governed by the heterogeneity of the agent population and the coordination patterns that emerge during training. The following flowchart encodes the decision logic as a state table:

**Table 3: Algorithm Selection Flowchart — From IPPO to Advanced Methods**

| Stage | Condition | Algorithm | Configuration | Escalate When |
|-------|-----------|-----------|---------------|---------------|
| 1 (Start) | 4 roles, different obs/action spaces | **IPPO** [^1^] | One policy per role type; `share_params=False` | Role coordination fails; builders idle waiting for gatherers |
| 2 | Cross-role dependencies visible in reward | **MAPPO** [^352^] | Centralized critic on concatenated global state; `train_batch_size=4096` | MAPPO epochs >10 cause oscillation; lazy agents emerge |
| 3 | True heterogeneity causing MAPPO collapse | **HAPPO** [^27^] | Sequential update via HARL; `algo_name="happo"`; per-role learning rates | Sequential updates too slow; credit assignment still noisy |
| 4 | Lazy agents or credit assignment failure | **IPPO + COMA critic** [^305^] | Add counterfactual baseline; monitor per-agent Q-value variance | COMA scales poorly beyond 8 agents; local minima |
| 5 | Coordination requires learned communication | **IPPO + DIAT comm.** [^327^] | Differentiable inter-agent transformer; sparse gated escalation | Communication degenerates; premature convergence |
| Avoid | Homogeneous agents, shared action space only | ~~QMIX/VDN~~ [^1^] | N/A — not applicable to heterogeneous village | HeMAC: QMIX "fails entirely" under heterogeneity |

The escalation logic is empirical, not theoretical. Stage 1 (IPPO) should be trained for at least 5–10 million environment steps before concluding that coordination failure requires Stage 2 (MAPPO). Many apparent coordination problems are actually reward-shaping problems: if the reward function does not provide positive feedback for successful handoffs (e.g., a builder receiving wood and placing a block), no algorithm will learn coordination. The algorithm flowchart assumes the reward function has already been validated through single-agent pre-training of each role.

BenchMARL [^79^] should be introduced at Stage 2 or 3 not as the primary training framework but as an evaluation harness: run the same PettingZoo environment through both RLlib (IPPO/MAPPO) and BenchMARL (IPPO/MAPPO/HAPPO) to verify that performance differences are due to algorithmic choices rather than implementation details. This cross-framework validation is a best practice for any MARL research project and is especially valuable when the project may eventually be published as a benchmark paper.

For a solo developer, the path of least resistance is clear: define the environment in PettingZoo, train with RLlib's IPPO, monitor per-role metrics, and escalate through the flowchart only when concrete failure modes are observed. This minimizes framework complexity, maximizes available documentation and community support, and leaves open the full algorithmic upgrade path as the project matures.
