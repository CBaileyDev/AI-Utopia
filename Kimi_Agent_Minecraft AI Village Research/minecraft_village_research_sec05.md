## 5. LLM-as-Planner Architectures

The central question facing any hybrid LLM–RL system is deceptively simple: what exactly does the language model output, and how does the downstream controller consume it? In the context of Minecraft village construction, where a single high-level goal ("build a village") must decompose into hundreds of low-level actions across multiple specialized agents, the planner–controller interface becomes the load-bearing architectural decision. This section analyzes how five representative systems structure that interface, extracts the design dimensions that matter, and specifies a concrete subgoal-format schema for an LLM + per-role RL policy architecture.

### 5.1 Architectural Patterns from Existing Systems

The dominant pattern across state-of-the-art Minecraft agents is a hierarchical decomposition: an LLM handles high-level task decomposition into sub-goals, while a low-level controller — variously an RL policy, a scripted API executor, or a JavaScript interpreter — handles primitive action execution [^27^][^26^][^59^][^60^]. This separation is nearly universal because LLMs struggle with precise low-level motor control but excel at reasoning about task structure. The variation lies in what the LLM emits, how often it is invoked, and what feedback closes the loop.

#### 5.1.1 Voyager: Event-Driven Planning with Code-as-Action

Voyager, introduced by Wang et al. (May 2023) from NVIDIA, Caltech, and Stanford, is the canonical LLM-powered embodied lifelong learning agent in Minecraft [^27^]. Its architecture comprises three interacting components: an automatic curriculum that proposes progressively harder tasks, an ever-growing skill library of executable JavaScript code stored as a vector database keyed by GPT-3.5-generated description embeddings, and an iterative prompting mechanism that refines code through up to four rounds of execution feedback [^27^][^31^].

The prompt structure is instructive. The system prompt establishes the agent's role as a Minecraft assistant with code generation capabilities, enumerating Mineflayer control primitives (`exploreUntil`, `mineBlock`, `craftItem`, `placeItem`, `smeltItem`, `killMob`, chest interactions) and pathfinder goals [^22^][^31^]. The per-iteration user prompt assembles a dense context block: code from the last round, JavaScript execution errors, chat log feedback, biome and time, nearby blocks and entities, health and hunger, position and equipment, inventory state, chest contents, the current curriculum task, retrieved skills from the skill library, and critique from the self-verification module [^22^]. The LLM responds with an explanation (if applicable), a step-by-step plan, and complete JavaScript code using the provided APIs.

Voyager operates on an event-driven planning cycle rather than a fixed tick. The LLM is invoked once per curriculum task — typically every 30 seconds to 5 minutes of gameplay — with up to 4 iterative refinement rounds before the curriculum abandons the task and proposes an alternative [^27^]. This sparse invocation pattern is cost-efficient relative to per-step planners but still consumes substantial API budget because each planning iteration involves one GPT-4 call for code generation plus one GPT-4 call for self-verification. The action vocabulary is *code-as-action*: complete JavaScript programs that call Mineflayer APIs, offering temporally extended and compositional behaviors but requiring API knowledge and interpreter reliability [^27^].

Failure handling follows a three-tier mechanism: environment feedback (intermediate progress messages from the bot), execution errors (JavaScript interpreter output), and self-verification (a separate GPT-4 critic that checks task completion against agent state and provides corrective critique) [^27^]. New skills are committed to the vector-indexed library only after self-verification confirms success, enabling persistent cross-world transfer.

#### 5.1.2 GITM: Hierarchical Decomposer–Planner–Interface

Ghost in the Minecraft (GITM), from OpenGVLab (May 2023), replaces Voyager's code generation with a rigid three-layer hierarchy: an LLM Decomposer, an LLM Planner, and an LLM Interface [^26^]. The Decomposer (GPT-3.5-turbo) breaks high-level goals into exactly depth-2 plan trees using Minecraft Wiki knowledge. The Planner generates sequences of structured action primitives. The Interface executes these via hand-written scripts requiring no LLM involvement [^26^][^224^].

GITM's Decomposer prompt is deliberately constrained: the plan tree must be exactly depth 2, each step described in one line, and bottom-level sub-goals must be basic actions [^238^]. The Planner prompt includes functional descriptions of nine structured actions (`equip`, `explore`, `approach`, `mine/attack`, `dig_down`, `go_up`, `build`, `craft/smelt`, `apply`), query illustration, response format specification (`[Explanation, Thought, Action List]`), and interaction guidelines for correcting failed actions [^26^]. The action vocabulary is thus *structured primitives* with deterministic, hand-scripted execution — offering reliability at the cost of expressiveness relative to Voyager's generated code.

Planning occurs at two timescales: decomposition runs once per high-level goal, and action planning runs once per sub-goal [^26^]. Failure handling is feedback-driven: after each structured action, the Interface reports success or failure, the reason is included in the feedback message, and the Planner regenerates an action sequence using chain-of-thought reasoning [^26^]. Memory is text-based — successful action lists are summarized and retrieved for similar sub-goals, with no vector embeddings [^26^]. GITM's key efficiency advantage is that it requires only a single CPU node with 32 cores, no GPU, and operates on GPT-3.5-turbo for decomposition — making it "10,000x more efficient than prior RL methods" in infrastructure terms [^26^][^241^].

#### 5.1.3 DEPS: Descriptor-Explainer-Planner-Selector Loop

DEPS (Describe, Explain, Plan and Select), from Tsinghua and CUHK (NeurIPS 2023), introduces an interactive replanning loop based on LLMs and was the first zero-shot multi-task agent to complete 70+ Minecraft tasks [^59^][^217^]. Its architecture comprises four distinct modules: a Descriptor that summarizes the current state as text when failures occur; an Explainer that uses an LLM to self-explain why the previous plan failed; a Planner that regenerates the plan incorporating error information; and a Selector — critically, a *learned* neural network with an Impala CNN backbone — that ranks parallel candidate sub-goals by estimating remaining steps to completion [^59^].

The LLM planner outputs *natural language sub-goals* (e.g., "mine oak wood", "craft stone pickaxe") rather than direct actions or code. These sub-goals are consumed by a goal-conditioned controller, $\pi(a|s,g)$, trained with RL, which translates natural language goals into keyboard and mouse actions [^59^]. The LLM is invoked once per sub-goal failure, not every tick; the controller runs at environment tick rate. Typical task completion requires 3,000–12,000 environment steps with 5–20 LLM calls total [^59^].

DEPS's prompt structure includes an initial planning prompt that specifies the agent's role and requests sequences of goals as code-like function definitions, and a replanning prompt that chains Descriptor output (current state summary), Explainer output (error analysis), and revised plan generation [^59^][^60^]. The interactive replanning loop — sub-goal fails, Descriptor summarizes, Explainer diagnoses, Planner regenerates, Selector ranks — proved highly effective, increasing Minecraft task success by 52.74% over baseline planning [^59^]. However, the base system has no persistent memory across sessions, and the Codex API dependency (since discontinued) required migration to newer models [^166^].

#### 5.1.4 JARVIS-1: Multimodal Key-Value Memory with CLIP Retrieval

JARVIS-1 (November 2023) from the CraftJarvis team augments multimodal LLM planning with a multimodal key-value memory system, achieving nearly perfect performance on 200+ tasks and a 5x improvement on ObtainDiamondPickaxe over prior records [^60^][^62^]. Unlike the text-only systems above, JARVIS-1 embeds visual observations (RGB screenshots) into the planning prompt via a MineCLIP visual encoder, and its planning module operates on both visual and textual state [^60^].

The planning prompt instructs the LLM (GPT-4) to extract action names, types, goal objects, tools, and ranks from input comprising current inventory, biome, health/hunger, and embedded screenshots [^60^]. The LLM responds with code-like function calls (`mine`, `craft`, `place`, `attack`) containing natural language sub-goals. Before execution, a self-check module simulates the plan step-by-step, predicts inventory state after each step, and verifies preconditions — a proactive failure detection mechanism unique to JARVIS-1 [^60^]. On execution failure, environment feedback triggers self-explanation and memory-augmented replanning.

The memory architecture is the system's distinguishing feature. Keys are multimodal: task description plus observation or situation at memory creation time. Values are successfully executed plans. Retrieval is two-stage: CLIP text embedding similarity for candidate selection, followed by CLIP visual embedding similarity for ranking [^60^]. The memory grows with gameplay, and empirical validation shows success rates increase with memory size — a genuine lifelong learning property [^60^]. The LLM is invoked once per task to generate sub-goals, then only on replan events, with 10–30 LLM calls typical for long-horizon tasks.

#### 5.1.5 Optimus-3: End-to-End MoE Eliminating API Cost

Optimus-3 (June 2025) represents a fundamentally different approach: an end-to-end Mixture-of-Experts (MoE) model based on Qwen2.5-VL that eliminates external LLM API calls entirely at inference time [^5^]. Its architecture features a Task Router (Sentence-BERT classifier) that routes queries to task-specific experts (Planning, Captioning, Embodied QA, Grounding, Reflection), a Shared Knowledge Expert always activated for cross-task knowledge, and a VPT action head for low-level Minecraft control [^5^].

Because Optimus-3 is a trained model rather than a prompt-based system, it has no traditional "prompt structure" or "tick frequency" in the LLM-as-planner sense. Inference runs once per environment step locally on GPU. Planning is integrated into the model via the Planning expert. Failure handling uses a dedicated Reflection task expert and GRPO-based training with dependency-aware rewards for iterative self-correction [^5^]. There is no explicit external memory — knowledge is encoded in MoE parameters — but task expansion is supported by adding new task experts without catastrophic forgetting [^5^].

The performance gains are substantial: +20% on Planning, +3% on Long-Horizon Action, +66% on Captioning, +76% on Embodied QA, +3.4x on Grounding, and +18% on Reflection over previous state-of-the-art [^5^]. The 15% success rate on Diamond Group tasks and 35% success with 69% completion rate on Diamond Sword substantially exceed JARVIS-1 and GPT-4o baselines [^42^]. For operational cost, Optimus-3 represents the zero-API-cost extreme: inference is local GPU compute only, making it the most cost-effective option at scale for long-running agents, though it requires significant upfront training investment (7B parameter activated model, 32GB VRAM for server deployment) [^5^][^125^].

### 5.2 Interface Design Decisions

The five systems present a design space with five key dimensions: prompt composition, action vocabulary, invocation frequency, failure handling strategy, and operational cost. Table 1 synthesizes the architectural comparison.

**Table 1: Cross-System Architectural Comparison**

| Dimension | Voyager | GITM | DEPS | JARVIS-1 | Optimus-3 |
|---|---|---|---|---|---|
| Planner LLM | GPT-4 | GPT-3.5/4 | GPT-3.5/4 | GPT-4 | End-to-end MoE |
| Action vocabulary | JavaScript code (Mineflayer) | 9 structured primitives | Natural language sub-goals | Code-like functions + NL goals | Low-level K/M via VPT head |
| LLM invocation | Per task (30 s–5 min) | Per sub-goal | Per sub-goal failure | Per task + replan | Per env step (local) |
| LLM calls per task | 4–8 | 3–10 | 5–20 | 10–30 | 0 (local inference) |
| Failure detection | GPT-4 critic self-verify | Action success/failure feedback | Descriptor + Explainer loop | Self-check + env feedback | Reflection expert |
| Memory type | Vector skill library (embeddings) | Text summaries | Horizon-predictive selector (learned) | Multimodal key-value (CLIP) | MoE parameters |
| Est. cost per task | $2–10 | $0.50–5 | $0.10–2 | $1–5 | $0 (GPU amortized) |
| Key strength | Lifelong learning, compositionality | Efficiency, no GPU | Interactive replanning | Multimodal memory, lifelong | Zero API cost, generalist |
| Key weakness | High API cost, no vision | LiDAR only, superhuman stats | No cross-session memory | Complex retrieval, incomplete code | Requires 32 GB VRAM |

Several patterns emerge from this comparison. First, every successful system separates planning from execution — the LLM never directly controls keyboard and mouse. Second, memory architecture is the primary differentiator: vector libraries (Voyager), text summaries (GITM), learned selectors (DEPS), and multimodal key-value stores (JARVIS-1) each represent a point on the spectrum of retrievable experience. Third, cost correlates almost linearly with invocation frequency and model capability.

#### 5.2.1 Prompt Structure: System Prompt vs. Per-Tick Context

Across all prompt-based systems, the prompt decomposes into a relatively stable system instruction and a dynamic per-invocation context block. What goes into each determines the model's reasoning quality and context window consumption. Table 2 disaggregates the components.

**Table 2: Prompt Composition Across Systems**

| Component | Voyager | GITM | DEPS | JARVIS-1 |
|---|---|---|---|---|
| **System prompt** | Role definition, Mineflayer API list, pathfinder goals, utility functions | Role definition, action interface docs, response format `[Explanation, Thought, Action List]`, interaction guidelines | Role definition ("generate goal sequences"), output format (Python-like function defs) | Role definition, action extraction instructions, multimodal input format |
| **Per-invocation context** | Last code + errors + chat log + biome/time + nearby blocks/entities + health/hunger + position + inventory + chests + curriculum task + retrieved skills + critic critique | Target object + quantity + wiki knowledge + previous action feedback + current agent state (inventory, biome, depth) | Current inventory + visual observation + target object + history dialogue | Current inventory + biome + health/hunger + screenshot (MineCLIP) + task description |
| **Unique additions** | Retrieved skills (top-5 from vector DB) + self-verification critique | Structured action definitions + text-based wiki knowledge | Descriptor/explainer history in context | Multimodal memory retrieval (CLIP-ranked) + self-check simulation |
| **Est. tokens per call** | 3,000–8,000 | 1,500–4,000 | 2,000–5,000 | 4,000–10,000 (multimodal) |

The universal components validate a minimal prompt template: role definition, environment state (inventory, health, position, nearby entities and blocks), task specification, execution history, and domain knowledge [^27^][^26^][^59^][^60^]. The differentiating components offer the highest leverage for improvement: Voyager's retrieved skills reduce the need for the LLM to reason from scratch about known tasks; JARVIS-1's multimodal memory retrieval enables recognition of visually similar situations; DEPS's descriptor-explanation chain maintains a structured failure narrative in context that prevents repeated errors.

#### 5.2.2 Action Vocabulary: Code vs. Primitives vs. Natural Language

Three paradigms emerged from the surveyed systems. *Code-as-action* (Voyager) offers maximum expressiveness and composability — generated JavaScript can call any Mineflayer API, compose skills via function calls, and persist in a library for reuse. However, it requires the LLM to reason about JavaScript syntax, Mineflayer API semantics, and runtime error recovery simultaneously, which proved difficult for models below GPT-4 [^27^].

*Structured primitives* (GITM) restrict the LLM to nine hand-defined actions with deterministic execution. This constrains the LLM's reasoning to parameter selection rather than code generation, dramatically improving reliability at the cost of flexibility [^26^]. The nine actions cover the essential Minecraft interaction space but cannot express novel behaviors without manual extension.

*Natural language sub-goals* (DEPS, JARVIS-1) occupy a middle ground. The LLM outputs goals like "obtain 64 oak logs" or "craft stone pickaxe," and a separate goal-conditioned controller translates these into low-level actions. This decouples planning from execution cleanly: the LLM reasons about task structure, while the RL policy handles motor control conditioned on goal embeddings [^59^]. For an LLM + per-role RL policy architecture, this paradigm is the clear match — the LLM specifies what each role should achieve, and the role-specific policy determines how.

#### 5.2.3 Tick Frequency: Cost-Speed Trade-offs

The invocation frequency spectrum spans from per-tick (prohibitively expensive) to per-task (most efficient). ReAct-style agents that invoke an LLM every environment step face costs of $20–100+ per hour of Minecraft gameplay [^220^]. Voyager's per-task invocation (4–8 calls every 30 seconds to 5 minutes) achieves the sparsest LLM usage among prompt-based systems, with estimated costs of $2–10 per task using GPT-4 [^27^]. GITM and DEPS operate at intermediate frequencies, with DEPS's failure-driven replanning proving empirically efficient — only 5–20 LLM calls for tasks requiring thousands of environment steps [^59^].

Event-driven replanning is the consensus pattern: the LLM is invoked when something goes wrong or a major phase boundary is reached, not on a fixed schedule [^59^][^60^][^27^]. For a multi-role village-building system, this translates to invoking the LLM planner when a sub-goal fails, when all sub-goals in a phase complete, or when environmental conditions change significantly (e.g., nightfall, attack, resource exhaustion).

#### 5.2.4 Failure Handling and Replanning

The failure handling mechanisms form a clear hierarchy of sophistication. Voyager uses a separate GPT-4 critic for self-verification — reliable but doubles LLM cost per iteration [^27^]. GITM uses structured action success/failure feedback piped back to the Planner for chain-of-thought regeneration — simple and effective but limited to action-level failures [^26^]. DEPS's Descriptor-Explainer loop provides the richest failure narrative: the Descriptor summarizes state and outcome, the Explainer locates the error in the prior plan, and the Planner incorporates this explanation into a revised plan [^59^]. JARVIS-1 adds a proactive self-check layer that simulates execution before acting, catching precondition violations before they become runtime failures [^60^].

The empirical lesson is that the most robust systems combine *proactive* verification (check before executing) with *reactive* environment feedback (adapt after failure) [^60^]. For a system with per-role RL policies, the reactive path is the natural one: when a role policy reports failure (timeout, health critical, tool broken), that structured report flows to the LLM planner which generates a revised sub-goal sequence.

#### 5.2.5 LLM Cost Analysis

Operational cost is the primary constraint on LLM-as-planner architectures. GPT-4-class agents at moderate replanning frequencies cost $5–15 per hour; dense replanning with full multimodal context pushes costs to $20–50 per hour [^13^][^27^]. The cost drivers, in order of impact, are: call frequency (per-tick >> per-sub-goal >> per-task), model choice (GPT-4 >> GPT-4o-mini >> GPT-3.5 >> open-source), context length (multimodal >> code + skills >> text-only), and self-verification overhead (extra LLM call per iteration).

Reduction strategies with validated effectiveness include: plan caching (AgenticCache achieves 79% cost savings by reusing validated plans for similar states) [^297^]; two-model routing (directing 80% of routine tasks to GPT-4o-mini and 20% requiring complex reasoning to GPT-4) [^261^]; fine-tuned open-source models (MineMA-8B achieves Voyager-plus-GPT-4o-mini performance at GPU cost only, with zero per-call API charges) [^13^]; and skill library reuse (Voyager's vector retrieval reduces the need for novel code generation) [^27^]. For sustained village-building operations, a fine-tuned open-source model or Optimus-3-style local inference is the only economically viable path at scale.

### 5.3 Recommended Interface Design

Drawing from the architectural patterns above, the recommended design for an LLM + per-role RL policy system is a hierarchical subgoal specification interface. The core principle is separation of concerns: the LLM reasons about *which role* to activate and *what sub-goal* to pursue; the RL policy reasons about *how* to achieve that goal through low-level actions. This mirrors DEPS's goal-conditioned policy design but with learned policy roles instead of a single monolithic controller [^59^][^244^].

#### 5.3.1 Subgoal Specification Format

The LLM planner outputs a structured JSON plan where each sub-goal specifies the role, target state, termination conditions, constraints, and fallback options. The schema is:

```json
{
  "$schema": "llm_plan_output",
  "plan_id": "plan_village_001_rev2",
  "high_level_goal": "build a village with 3 houses and a well",
  "subgoals": [
    {
      "subgoal_id": "sg_01",
      "role": "lumberjack",
      "goal_specification": {
        "target_state": {
          "inventory_delta": { "oak_log": 64 }
        },
        "termination_conditions": {
          "success_criteria": ["inventory_contains(oak_log, 64)"],
          "timeout_ticks": 6000,
          "failure_events": ["health_below(5)", "night_falls"]
        }
      },
      "constraints": {
        "preserve_items": ["iron_pickaxe"],
        "avoid_biomes": ["dark_forest"],
        "max_health_cost": 2.0,
        "tool_requirements": ["stone_axe_or_better"]
      },
      "fallback_subgoals": ["gather_birch_log", "gather_spruce_log"]
    },
    {
      "subgoal_id": "sg_02",
      "role": "miner",
      "goal_specification": {
        "target_state": {
          "inventory_delta": { "cobblestone": 64 }
        },
        "termination_conditions": {
          "success_criteria": ["inventory_contains(cobblestone, 64)"],
          "timeout_ticks": 8000,
          "failure_events": ["health_below(4)", "pickaxe_broken"]
        }
      },
      "constraints": {
        "tool_requirements": ["stone_pickaxe_or_better"]
      }
    },
    {
      "subgoal_id": "sg_03",
      "role": "builder",
      "goal_specification": {
        "target_state": {
          "structures_built": ["house_1", "house_2", "house_3", "well"]
        },
        "termination_conditions": {
          "success_criteria": ["structures_completed(4)"],
          "timeout_ticks": 12000,
          "failure_events": ["materials_exhausted"]
        }
      }
    }
  ],
  "dependencies": [
    { "before": "sg_01", "after": "sg_03" },
    { "before": "sg_02", "after": "sg_03" }
  ]
}
```

The `role` field activates one of the trained RL policies (`miner`, `lumberjack`, `builder`, `fighter`, `explorer`, `crafter`, `farmer`). The `goal_specification.target_state` defines the desired outcome in terms of inventory changes, location, or structures built. The `termination_conditions` block provides three complementary signals: `success_criteria` (explicit conditions that mark completion), `timeout_ticks` (hard limit before the policy must report failure), and `failure_events` (conditions that trigger immediate abort and replanning). The `constraints` block encodes safety and efficiency bounds — items to preserve, biomes to avoid, maximum acceptable health cost, and tool requirements. Finally, `fallback_subgoals` provides the LLM planner with alternative paths when a primary sub-goal is infeasible, enabling graceful degradation without a full replan cycle.

The `dependencies` array at the plan level encodes ordering constraints between sub-goals (e.g., resource gathering must precede construction), forming a directed acyclic graph that the execution scheduler traverses. This DAG structure, adapted from VillagerAgent's task decomposition framework [^100^], allows parallel execution where dependencies permit and enforces sequential ordering where required.

#### 5.3.2 RL Policy Input Format

Each per-role RL policy receives a structured input combining a goal embedding, multimodal observation, and execution context:

```json
{
  "$schema": "rl_policy_input",
  "role_id": "lumberjack",
  "goal_embedding": {
    "vector": [0.12, -0.05, 0.33, ...],
    "structured": {
      "target_state": { "inventory_delta": { "oak_log": 64 } },
      "termination_conditions": {
        "success_criteria": ["inventory_contains(oak_log, 64)"],
        "timeout_ticks": 6000,
        "failure_events": ["health_below(5)", "night_falls"]
      }
    }
  },
  "observation": {
    "visual": "base64_encoded_rgb_or_clip_embedding",
    "inventory": { "oak_log": 23, "stone_axe": 1 },
    "player_state": {
      "position": [100.5, 64.0, -200.3],
      "health": 18,
      "hunger": 15,
      "armor": 4
    },
    "nearby_entities": ["pig", "sheep", "zombie"],
    "nearby_blocks": ["oak_log", "grass_block", "dirt"],
    "time_of_day": "day",
    "biome": "plains"
  },
  "execution_context": {
    "ticks_elapsed": 3420,
    "plan_id": "plan_village_001_rev2",
    "subgoal_id": "sg_01",
    "previous_actions": ["equip(stone_axe)", "move_to(105,64,-195)", "chop(oak_log)"]
  }
}
```

The `goal_embedding.vector` field contains a fixed-dimension dense vector (e.g., 256-dimensional output from a sentence transformer encoding the natural language goal description) that conditions the policy network. The `goal_embedding.structured` field preserves the full goal specification for interpretability and for the policy's internal termination checking. The `observation` block follows the hybrid design recommended in prior analysis: symbolic state for precise inventory and position data, plus visual input (either raw RGB or a CLIP embedding) for pattern recognition [^63^][^5^]. The `execution_context` provides temporal grounding — elapsed ticks, plan and sub-goal identifiers, and the last $N$ actions for history-aware decision making.

#### 5.3.3 Failure Report Format

When an RL policy cannot complete its assigned sub-goal, it emits a structured failure report that feeds back to the LLM planner:

```json
{
  "$schema": "failure_report",
  "report_id": "fr_001",
  "plan_id": "plan_village_001_rev2",
  "subgoal_id": "sg_01",
  "role": "lumberjack",
  "status": "failure",
  "failure_details": {
    "failure_type": "attacked",
    "failure_tick": 2340,
    "final_state": {
      "position": [112.0, 64.0, -188.5],
      "health": 3,
      "inventory": { "oak_log": 23, "stone_axe": 1 },
      "nearby_threats": ["zombie", "skeleton"]
    },
    "descriptor_summary": "Failed to gather 64 oak logs. Health dropped to 3 after zombie attack at tick 2340. Inventory has 23 oak logs. Night fell at tick 2100.",
    "execution_trace": [
      { "tick": 2300, "action": "chop(oak_log)", "observation_summary": "mining oak_log, progress 0.7", "reward": 0.1 },
      { "tick": 2320, "action": "chop(oak_log)", "observation_summary": "zombie approaching from east", "reward": 0.0 },
      { "tick": 2340, "action": "retreat",
        "observation_summary": "health dropped to 3, zombie hit", "reward": -1.0 }
    ]
  },
  "partial_progress": {
    "inventory_delta": { "oak_log": 23 },
    "exploration_coverage": 0.4
  }
}
```

The critical field is `failure_details.failure_type`, which uses a closed vocabulary (`timeout`, `health_critical`, `tool_broken`, `inventory_full`, `path_blocked`, `resource_unavailable`, `attacked`, `unknown`). This typed signal enables the LLM planner to select an appropriate recovery strategy without re-analyzing the full execution trace — for example, `attacked` triggers an emergency retreat sub-goal followed by re-assignment with a fighter escort, while `resource_unavailable` triggers a fallback to alternative materials or exploration. The `descriptor_summary` is auto-generated from the execution trace using a lightweight template or small language model, providing natural language context for the LLM's replanning reasoning. The `partial_progress` block captures what was achieved before failure, enabling the LLM to resume from an intermediate state rather than restarting from scratch.

#### 5.3.4 Complete Prompt Template

The LLM planner receives a composite prompt assembled from persistent and dynamic components. Based on the universal patterns identified across Voyager, GITM, DEPS, and JARVIS-1 [^27^][^26^][^59^][^60^], the recommended template structure is:

```
[SYSTEM PROMPT — persistent]
You are a Minecraft strategic planner. Your role is to decompose
high-level goals into executable subgoals for specialized RL policies.
Available roles: {miner, lumberjack, builder, fighter, explorer, crafter, farmer}
Each role has a trained RL policy that executes low-level actions.

Output format: JSON following the subgoal specification schema.
Constraints:
- Subgoals must have clear success criteria
- Always include fallback_subgoals for resource-gathering tasks
- Timeout_ticks must not exceed 12000 (10 minutes at 20 TPS)
- Health-critical failure_events are mandatory for combat/exploration roles

[KNOWLEDGE BASE — retrieved per goal]
Crafting recipes relevant to current goal:
- oak_log ×1 → oak_planks ×4 (crafting table not required)
- oak_planks ×2 + stick ×2 → wooden_pickaxe ×1
- cobblestone ×3 + stick ×2 → stone_pickaxe ×1
...
Biome info and resource distributions for current region.
Threat assessment: hostile spawn rates, nearest dungeon locations.

[CURRENT STATE — refreshed per invocation]
Position: (100.5, 64.0, -200.3) | Biome: plains
Health: 18/20 | Hunger: 15/20
Inventory: oak_log:23, stone_axe:1, cobblestone:8
Equipment: stone_axe (held), leather_chestplate
Time: 12500 (afternoon) | Weather: clear
Nearby blocks (32-block radius): oak_log×12, grass_block, dirt, stone
Nearby entities: pig×3, sheep×2, zombie×1 (distance 25)

[PLAN HISTORY — accumulated]
Current plan: plan_village_001_rev2, executing: sg_01
Completed: []
Failed (with reports): []

[MEMORY RETRIEVAL — from vector DB]
Similar past plans: ["gather_wood_in_plains_with_axe", ...]
Successful strategies: ["avoid_dark_forest_at_night", ...]

[USER GOAL — top-level objective]
Build a village with 3 houses and a well.

[INSTRUCTION]
Generate a plan as JSON. Consider available resources, environmental
threats, fallback options, and optimal ordering based on dependencies.
```

This template balances completeness with token efficiency. The system prompt is loaded once per session. The knowledge base is retrieved based on goal relevance (following GITM's wiki-knowledge approach [^26^]). The current state block includes all fields found to be universally necessary across surveyed systems. Plan history and memory retrieval provide the contextual grounding that Voyager's skill library and JARVIS-1's multimodal memory demonstrated as critical for long-horizon performance [^27^][^60^]. The instruction concludes with explicit guidance to consider dependencies, threats, and fallbacks — steering the LLM toward robust planning without constraining it to a rigid template.

The interface described above — structured sub-goal specification in JSON, goal-conditioned RL policies with dense embeddings, typed failure reports with auto-generated summaries, and composite prompts with retrieved knowledge — provides the bridge between high-level LLM reasoning and low-level policy execution. It preserves the LLM's strengths in task decomposition and error diagnosis while delegating motor control to RL policies optimized for their specific roles. The next chapter turns to the design of those policies and their reward functions, beginning with the Gatherer role.
