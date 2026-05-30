# Kimi agent-swarm research prompt (dispatched 2026-05-30)

The user can run a Kimi agent swarm (up to ~300 sub-agents). This is the
copy-paste prompt handed off to inform the **design fork** in `MORNING_BRIEF.md`
(thin reactive controller + smart producers vs end-to-end learned search) and the
multi-agent village architecture. Findings, when they come back, feed the fork
decision + the Explorer/Scout role design.

---

```
ROLE: You are a research swarm producing a decision-grade technical report for an applied RL engineering project. Fan out across the questions below, then SYNTHESIZE into one coherent report. Prefer primary sources (papers, official docs, real project repos/postmortems) over blog rehash. Cite every non-obvious claim with a source (title + link/venue + year). Flag disagreement between sources. Prefer 2021-2026 work but include seminal older work. NO filler, NO generic "it depends" — give concrete techniques, named algorithms, hyperparameter ranges, and worked tradeoffs an engineer can act on.

PROJECT CONTEXT (so your recommendations fit):
- Goal: a persistent multi-agent Minecraft (Java 1.21.1) "village" of specialized per-role RL agents (Lumberjack/gatherer, Miner, Farmer, Soldier, Explorer/Scout) that cooperatively run a village.
- Stack: Python RL = Ray RLlib PPO, new API stack, per-role LSTM RLModule (custom: CoreEncoder→SharedBackbone→RoleEncoder→ActorHead + a CTDE centralized critic). Action space is a multi-head Dict (discrete skill_type ∈ {NAVIGATE,HARVEST,DEPOSIT,SEARCH,WAIT,...}, a target/instance pointer, spatial params, plus unused comm heads). Observation is an egocentric vector (a 32×32 local resource grid flattened, a top-8 "nearest resources" list with per-(x,z)-column topmost-within-±3 vertical scan, inventory, scalars) — crucially BLIND beyond a ~16-block perception radius.
- Training is in a fast headless SIM that mirrors the real Minecraft skill/arena byte-for-byte (RLGym/RocketSim + Craftax pattern); sim-trained policies are validated by a sim→real transfer gate. Single role (M1B gatherer) is solved + transfers 3/3.
- CURRENT DECISION (this report must inform it): we built a "decision-core" that demotes the do-everything HARVEST skill so the POLICY picks which resource instance to mine + must NAVIGATE/explore when nothing is visible (blind 2-cluster arena: one resource cluster visible, one beyond perception → requires a blind explore hop). A controlled ablation showed: a ZERO-LEARNING scripted follower (HARVEST nearest-visible; else move toward an oracle bearing cue) ties-or-beats the trained PPO policy in EVERY condition; the perception action-mask (which forces NAVIGATE when blind) is load-bearing; and PPO does not converge (return oscillates, persistently high entropy, intermittent non-finite-KL). Whether a *properly stabilized* PPO on a *search-requiring* arena can learn directed exploration is UNTESTED. The fork: (A) accept a thin reactive controller and put intelligence in PRODUCERS (a dedicated Explorer/Scout role + memory that emit "the forest is that way" bearings from PARTIAL info), vs (B) make one policy genuinely learn to search end-to-end.

DELIVER A REPORT WITH THESE SECTIONS:

§1 — SPARSE-REWARD EXPLORATION LEARNING (feasibility of fork B).
  Q1. For a discrete-action, partially-observed, sparse-reward navigation/foraging task where the target is OUTSIDE the observation radius (must search blind), which methods reliably make an RL policy learn DIRECTED exploration? Compare: intrinsic motivation (RND, ICM, NGU, BYOL-Explore, count/pseudo-count), episodic-memory exploration, frontier-based + learned exploration, Go-Explore, and recurrent-memory agents. For EACH: when it works, failure modes, compute cost, and whether it needs a metric/occupancy map. Rank by suitability for a ~minutes-to-train fast-sim with an egocentric vector obs that is zero beyond 16 blocks.
  Q2. PPO-specific stabilization for multi-head action spaces WITH action masking: causes of non-finite/exploding KL when masking forces zero-probability actions; correct handling (kl_coeff scheduling vs fixed vs 0, entropy coefficient schedules, masking BEFORE vs after the distribution, per-head entropy, log-std clamping for unused Gaussian heads, invalid-action penalty vs hard mask). Concrete settings/recipes from real implementations.
  Q3. Behavior-cloning / offline warm-start to bootstrap search before RL (BC→PPO, DAgger, AWAC, IQL, decision-transformer-as-prior). Does warm-starting from a scripted non-oracle searcher (e.g., a spiral/lawnmower sweep) demonstrably teach learnable search that pure PPO can't discover? Evidence + recipes.

§2 — "WHERE DOES THE INTELLIGENCE LIVE" (fork A vs B, architecture).
  Q4. Hierarchical / hybrid architectures for long-horizon open-world agents: options framework, feudal/HRL, HTN & GOAP planners over RL skills, LLM-as-planner over learned low-level policies. For each, what is delegated to the planner/producer vs the learned policy, and the empirical tradeoffs (sample efficiency, generalization, debuggability) vs end-to-end RL.
  Q5. Survey how COMPARABLE open-world / Minecraft agent systems architect the policy ↔ planner ↔ skill ↔ memory ↔ role stack: Voyager, MineDojo/MineCLIP, MineRL BASALT winners, DreamerV3-on-Minecraft (diamond), Plan4MC, GITM (Ghost in the Minecraft), JARVIS-1, Mineflayer/Mindcraft-style LLM-bot frameworks, and non-Minecraft persistent multi-agent worlds (Generative Agents / "AI town", AI-Economist). For each: is exploration learned or scripted? is "where to go" produced by a map/memory/planner or by the control policy? Extract the consensus pattern for "find a distant resource."

§3 — THE EXPLORER/SCOUT PRODUCER (if fork A).
  Q6. How to PRODUCE good exploration targets/bearings from PARTIAL observation + accumulated memory: online occupancy/semantic mapping, frontier selection, learned exploration policies that output a goal for a lower controller, and biome/resource-locating heuristics. What's the simplest thing that reliably emits "resource is in direction θ / at frontier F" for a downstream reactive controller?
  Q7. Multi-agent information sharing: how do cooperating agents share discovered locations (a shared map / blackboard / emergent comm), and does sharing scout-discovered bearings measurably help foragers? CTDE-compatible patterns.

§4 — SYNTHESIS + RECOMMENDATION.
  Given the project context and ablation result, give a reasoned recommendation between fork A and fork B (or a staged hybrid), the single highest-leverage next experiment for each fork, and the concrete techniques (named, with settings) to try first. State what evidence would change the recommendation.

OUTPUT: markdown, ~2000-4000 words, sectioned as above, with a sources list. Lead §4 with a 5-bullet executive summary. Where sources conflict, say so explicitly rather than averaging.
```
