# Executive Summary

This report synthesizes findings from 10 research dimensions — spanning 14+ prior agent projects, 8 MARL frameworks, 12 Fabric mods, 5 LLM planner architectures, and hardware benchmarking — into actionable engineering decisions for a solo developer building a multi-agent AI village in Minecraft.

---

### Key Findings

#### The three-tier architecture (LLM Planner → Goal Specification → Per-Role RL Policies) is optimal, unimplemented in any existing project, and represents a novel contribution

Every surveyed project falls into one of two camps: pure LLM agents (Voyager, GITM, JARVIS-1) that generate code or structured actions directly [^27^][^26^], or pure RL multi-agent systems (IPPO, MAPPO, QMIX) with no high-level planning layer [^1^][^295^]. The composition — an LLM planner emitting goal embeddings that condition per-role RL policies trained with CTDE (Centralized Training, Decentralized Execution) — does not exist in open source. The insight file flags this as "the highest-risk custom component" requiring an estimated 2–3x more engineering time than a monolithic approach [Insight 1]. The upside: a 4-role cooperative village with heterogeneous agents (gatherer, builder, farmer, defender) would be a genuinely novel contribution at the intersection of three under-explored research areas [^188^].

#### Py4J via UnionClef outperforms gRPC for single-machine deployment; no production gRPC mod exists

The user's proposed gRPC/websocket bridge adds serialization complexity with zero benefit for a single-workstation setup. UnionClef ships a production Py4J two-way bridge enabling direct Java object method calls from Python, actively maintained through April 2026 [^180^]. No production-ready gRPC mod exists for current Fabric versions — `fabric-grpc-api` targets only 1.20.1 and has not been updated in approximately 3 years [^51^]. Py4J is strictly more ergonomic for a solo developer: it eliminates protobuf schema maintenance and allows calling Baritone's `baritone.api` directly from the Python orchestrator [^150^].

#### Drop MARLlib in favor of direct Ray RLlib — the wrapper adds a dependency layer with no functional gain

MARLlib has been effectively unmaintained since late 2023, with its last release in April 2023 and documentation last updated May 2024 [^197^]. It is a thin wrapper over Ray RLlib that has not kept pace with RLlib's API evolution. The recommended stack is PettingZoo for environment definition (its Parallel API natively supports different observation and action spaces per agent [^25^]) plus `ray.rllib`'s multi-agent API directly. RLlib provides mature policy mapping, fractional GPU allocation on the RTX 4080 [^141^], and TensorBoard integration out of the box [^11^].

#### Hardware supports 4–6 parallel server instances — the bottleneck is software architecture, not compute

The Ryzen 9800X3D (Cinebench 2024 single-core score 133) with 64GB DDR5 and RTX 4080 16GB can comfortably run 4–6 parallel Fabric server instances at 20 TPS each, with each instance consuming ~2–3GB RAM in a village-sized world [^462^][^518^]. The RTX 4080 handles 8–16 agents during training [^141^]. The real constraints are algorithmic: credit assignment across heterogeneous roles, non-stationarity from simultaneous learning, and lazy agent dynamics [^295^][^301^][^303^] — all of which would persist on any hardware.

#### Bedrock support imposes a server-side-only constraint that eliminates the richest observation sources — decide before writing code

Geyser emulates a vanilla Java client; if a vanilla client cannot join the Fabric server, Geyser cannot work [^389^]. This forces all mods to be strictly server-side, ruling out client-side Fabric mods that would provide rendered view data, full HUD state, and precise player telemetry [^389^]. Server-side plugins can access world state, inventories, and entity positions, but not the agent's camera or client-side rendering. Floodgate-Fabric for Bedrock authentication is now actively maintained [^432^], so the Geyser stack itself is viable — but the observation-space limitation is permanent. This is a showstopper-level architectural fork that must be resolved before the first line of bridge code.

---

### Recommendation Highlights

**1. Start with one gatherer, not four roles.** The practical path is: Milestone 1 = single gatherer collecting wood and stone reliably; Milestone 2 = add builder coordination; Milestone 3 = add farmer; Milestone 4 = add defender; Milestone 5 = full village with LLM planner. Every additional role multiplies the state space, action space, and failure-mode surface simultaneously [Insight 9]. The curriculum learning evidence supports progressive scaling 2 → 3 → 4 agents [^374^].

**2. Use Py4J + UnionClef as the bridge, not gRPC.** Study UnionClef's Py4J gateway implementation as the reference pattern. Combine it with Baritone for pathfinding and low-level action primitives. This eliminates an entire category of serialization work and gives direct access to Java APIs from Python [^180^][^150^].

**3. Build three training phases, not one.** Phase 1: per-role policy pre-training in isolation (gatherer learns to gather). Phase 2: multi-agent CTDE with curriculum scaling (2 agents → 3 → 4). Phase 3: LLM planner integration with fine-tuning. Skipping Phase 1 leads to the "everything fails at once" debugging nightmare where credit assignment collapse, non-stationarity, and lazy agents are indistinguishable [Insight 6][^305^][^301^].

**4. Design the observation space as hybrid symbolic + local pixels, and architect reward verification at the system level.** Use symbolic observations for the LLM planner (inventory, block positions, entity states) and hybrid symbolic + local pixel patches for RL policies — this pattern is validated by JARVIS-1 and Optimus-3 [^169^]. For rewards, implement potential-based shaping with automatic verification (checking inventory changes are permanent) and KL-regularization to a pretrained behavior prior, rather than patchwork reward-function engineering [Insight 7].

---

### Open Questions Requiring Immediate Human Decisions

**Bedrock or no Bedrock?** If Bedrock player support is a hard requirement, all observation extraction must be server-side, which limits the agent's perceptual richness. If abandoned, client-side mods unlock significantly richer observations (camera feed, precise player state, full HUD). This decision is irreversible without a full rewrite of the bridge layer.

**Which algorithm baseline?** The evidence points to IPPO as the simplest starting point (HeMAC benchmark shows it outperforms MAPPO in high-heterogeneity scenarios [^1^]), with QMIX/VDN as the fallback if credit assignment fails, and HARL HAPPO as the nuclear option if heterogeneity causes training collapse [^27^]. This is an empirical tuning question — budget time for algorithm comparison in Phase 2.

**LLM cost budget?** A Voyager-style GPT-4 planner can cost thousands of dollars per experiment [^13^]. The operational range is wide: $0.20–50/hour depending on model choice, replanning frequency, and caching. AgenticCache-style plan caching achieves 79% cost savings [^297^]. Decide on a monthly LLM budget before Phase 3 — it determines whether GPT-4o-mini, fine-tuned open-source models, or a mix is the right choice.
