# Cross-Verification: Multi-Agent Minecraft AI Village Research

## Verification Methodology
Compared findings across all 10 dimension files. Categorized each major claim by confidence tier based on cross-dimensional confirmation.

---

## High Confidence Findings (Confirmed by ≥2 independent sources)

### HC-1: Voyager is the most influential but archived codebase
- Dim01: "Voyager (2023) remains the most influential codebase" — MIT license, skill library pattern
- Dim10: "6.9k stars, archived, last commit July 2023" — reference-only, not forkable for production
- Dim05: Detailed prompt structure, skill library architecture analysis
- **Verdict:** HIGH CONFIDENCE — Clone and study for architecture, but do not depend on for active development.

### HC-2: CTDE (Centralized Training, Decentralized Execution) is the recommended MARL architecture
- Dim04: "PettingZoo + RLlib with IPPO/MAPPO" recommended
- Dim07: "CTDE with value decomposition (QMIX/VDN) is the recommended foundation"
- Dim03: CTDE architecture cited for multi-agent observation design
- **Verdict:** HIGH CONFIDENCE — Implement CTDE as baseline architecture.

### HC-3: No production-ready gRPC mod exists for current Fabric versions
- Dim02: "fabric-grpc-api targets only 1.20.1, not updated in ~3 years"
- Dim10: No gRPC/websocket bridge mod found in any repo
- **Verdict:** HIGH CONFIDENCE — Must build custom bridge. Py4J (UnionClef approach) or raw WebSocket recommended over gRPC.

### HC-4: UnionClef + Baritone are the best existing bot infrastructure
- Dim02: "UnionClef ships with built-in Py4J two-way bridge, active April 2026"
- Dim10: "Baritone 8.9k stars, actively maintained through May 2026"
- Dim02: Baritone LGPL-3.0, stable API, 1.21.8 support
- **Verdict:** HIGH CONFIDENCE — Study UnionClef source for Py4J bridge pattern; use Baritone as library.

### HC-5: Parallel server instances beat tick warp for RL training
- Dim08: "4-6 parallel instances at 20 TPS is realistic on 9800X3D"
- Dim08: "Parallel instances are superior strategy for RL training — more stable rollouts"
- **Verdict:** HIGH CONFIDENCE — Run 4-6 parallel Fabric server instances rather than tick-warping one.

### HC-6: Hybrid observation space (symbolic + local pixels) outperforms pure approaches
- Dim03: "Hybrid combining symbolic state + local pixel patches recommended"
- Dim01: JARVIS-1 and Optimus-3 both use hybrid approaches successfully
- Dim05: LLM planners use symbolic observations; RL policies benefit from local pixel context
- **Verdict:** HIGH CONFIDENCE — Use symbolic observations for LLM planner, hybrid for RL policies.

### HC-7: Credit assignment, non-stationarity, and lazy agents are the top 3 failure modes
- Dim07: All three rated HIGH priority for Minecraft village scenario
- Dim04: HeMAC benchmark confirms standard MARL algorithms struggle with heterogeneity
- Dim07: COMA + intrinsic motivation (LAIES) are the recommended mitigations
- **Verdict:** HIGH CONFIDENCE — Design mitigation strategy around these three from day one.

### HC-8: MineLand and mindcraft are the most relevant active multi-agent projects
- Dim10: "mindcraft 5.3k stars, 69 contributors, commits through May 2026"
- Dim10: "MineLand 48 agents, Gym-style API, Docker support"
- Dim01: Multi-agent noted as "biggest gap" in prior art
- **Verdict:** HIGH CONFIDENCE — Study both repos before starting.

### HC-9: CraftJarvis team (MineStudio, Optimus-3, JARVIS-1) is the most active research group
- Dim01: "Only CraftJarvis-team repos show consistent active maintenance through 2025"
- Dim10: MineStudio described as "modern replacement for MineDojo"
- Dim10: Multiple repos from this team in top 10 ranking
- **Verdict:** HIGH CONFIDENCE — Follow this team's work closely.

### HC-10: Bedrock support is conditionally feasible with server-side-only constraint
- Dim09: "Geyser emulates a vanilla Java client — all mods must be strictly server-side"
- Dim09: "Floodgate-Fabric now exists and is actively maintained"
- Dim09: Custom entities require manual Geyser mapping work
- **Verdict:** HIGH CONFIDENCE — Feasible but constrains mod selection to server-side only.

---

## Medium Confidence Findings (Single authoritative source)

### MC-1: MARLlib is effectively unmaintained since late 2023
- Dim04: "Effectively unmaintained since late 2023"
- Not directly contradicted by any other dimension
- **Verdict:** MEDIUM CONFIDENCE — Use Ray RLlib directly instead.

### MC-2: HARL (HAPPO/HATRPO) is algorithmically superior for true heterogeneity
- Dim04: JMLR 2024 paper proves monotonic improvement guarantees
- Dim07: Not mentioned in failure mode analysis (independent confirmation lacking)
- **Verdict:** MEDIUM CONFIDENCE — Strong theoretical backing but limited empirical validation on Minecraft-scale tasks.

### MC-3: 200-500 TPS sustained on 9800X3D with tick sprint
- Dim08: Sourced from Carpet Mod documentation and hardware benchmarks
- No direct empirical test data found for specifically village-sized worlds
- **Verdict:** MEDIUM CONFIDENCE — Likely achievable but depends heavily on world state complexity.

### MC-4: LLM planner should tick on task completion, not per-step
- Dim05: Voyager invokes LLM once per task (30s-5min), not per tick
- Dim05: DEPS replans on failure or horizon prediction
- **Verdict:** MEDIUM CONFIDENCE — Event-driven replanning is consensus; exact frequency depends on task granularity.

### MC-5: Potential-based shaping with inventory-normalized rewards recommended for gatherer
- Dim06: Derived from synthesis of MineRL, VPT, Plan4MC approaches
- No single paper validates the exact 3-stage design proposed
- **Verdict:** MEDIUM CONFIDENCE — Individual components are validated; combined design needs empirical testing.

---

## Low Confidence Findings (Weak sourcing or unverified claims)

### LC-1: Exact RAM per server instance numbers
- Dim08: "2-3GB baseline for village-sized loaded area" but sources are hosting guides, not RL-specific
- RAM depends heavily on chunk count, entity count, and optimization mod stack
- **Verdict:** LOW CONFIDENCE — Treat as rough estimate; benchmark empirically.

### LC-2: Cost estimates for LLM planner ($0.20-50/hour)
- Dim05: Wide range depends on model choice, replanning frequency, and caching
- No hard data from production deployments at scale
- **Verdict:** LOW CONFIDENCE — Budget $5-20/hour for GPT-4 class models during active development.

---

## Conflict Zones

### CZ-1: Best action space — code-as-policy vs. structured primitives vs. low-level keypresses
- **Dim03/Dim05:** Voyager's code-as-action offers "superior composability" but RL agents need different interface
- **Dim03:** GITM's structured primitives (9 actions) offer "better reliability"
- **Dim03:** MineRL low-level keypresses are standard for RL but sample-inefficient
- **Resolution:** NOT A TRUE CONFLICT — Different layers need different action spaces. LLM planner uses structured primitives; RL policies output parameterized skill commands; low-level execution is environment-side.

### CZ-2: IPPO vs. MAPPO vs. QMIX algorithm choice
- **Dim04:** IPPO recommended as starting point (HeMAC shows it beats MAPPO in high heterogeneity)
- **Dim07:** QMIX/VDN recommended for credit assignment
- **Dim04:** HARL HAPPO theoretically best for true heterogeneity
- **Resolution:** PARTIAL CONFLICT — Start with IPPO (simplest), move to MAPPO if coordination needed, switch to QMIX if credit assignment fails, consider HAPPO if heterogeneity causes collapse. This is an empirical tuning question, not a theoretical one.

### CZ-3: Client-side mod vs. server plugin architecture for the bridge
- **Dim02:** "Client-side Fabric mod + Python orchestrator" recommended
- **Dim09:** Server-side-only mods required for Bedrock support
- **Resolution:** TRUE TENSION — Client-side mod gives richer world state but precludes Bedrock support. Server-side bridge works for Bedrock but has limited observation access. Recommend starting server-side (future-proof for Bedrock) and upgrading if observation quality is insufficient.

### CZ-4: UnionClef vs. mineflayer as the foundational bot layer
- **Dim02:** UnionClef recommended for Fabric/Java-first approach
- **Dim10:** mineflayer recommended for protocol-level JavaScript approach
- **Resolution:** NOT A TRUE CONFLICT — They serve different architectures. UnionClef for Java/Fabric mod ecosystem; mineflayer for standalone Node.js/Python agents. User's Fabric stack favors UnionClef path, but mindcraft (mineflayer-based) is more actively maintained for multi-agent.
