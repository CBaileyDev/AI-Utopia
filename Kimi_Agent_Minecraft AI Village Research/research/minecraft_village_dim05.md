# Cluster 5: LLM-as-Planner Architectures for Minecraft Agents

## Research Report: LLM Planner / Low-Level Controller Interface Design

**Date:** July 2025
**Focus:** Extractable architectural patterns for LLM-as-planner systems in Minecraft, with concrete interface specifications for LLM + RL-policy hybrid agents.

---

## Executive Summary

- **The dominant pattern** across state-of-the-art Minecraft agents (Voyager, GITM, DEPS, JARVIS-1) is a hierarchical decomposition: LLM handles high-level task decomposition into sub-goals, while a low-level controller (RL policy, scripted API, or code executor) handles primitive action execution. This separation is nearly universal because LLMs struggle with precise low-level motor control but excel at reasoning about task structure [^27^][^26^][^59^][^60^].
- **Code-as-action (Voyager)** and **structured action primitives (GITM)** represent the two major action vocabulary paradigms. Code-as-action offers superior composability and interpretability; structured primitives offer better reliability and deterministic execution guarantees [^27^][^26^].
- **Memory architectures differ significantly**: Voyager uses a vector-indexed skill library; JARVIS-1 uses multimodal key-value memory with CLIP-based retrieval; GITM uses text-based memory of successful action lists; DEPS uses a learned horizon-predictive selector for sub-goal ranking [^27^][^60^][^26^][^59^].
- **Replanning is universally triggered by execution failure**, but the failure detection mechanisms vary: Voyager uses a GPT-4 critic for self-verification; DEPS uses a descriptor-explainer loop with environment feedback; GITM uses structured action success/failure feedback; JARVIS-1 uses self-check and environment feedback [^27^][^59^][^60^].
- **LLM API costs are the primary operational constraint** — Voyager-style agents with GPT-4 can cost thousands of dollars per experiment. Cost reduction strategies include: using smaller models for routine tasks (GPT-3.5/GPT-4o-mini), caching plans (AgenticCache achieves 79% cost savings), and using open-source models (LLaMA-3 fine-tuned variants) [^13^][^297^].
- **Recommendation**: For an LLM + per-role RL policy architecture, use a **subgoal specification interface** where the LLM outputs structured subgoal definitions (target state + constraints + timeout), and RL policies execute conditioned on goal embeddings. This mirrors DEPS's goal-conditioned policy design but with learned policy roles instead of a single monolithic controller [^59^][^244^].

---

## 1. Voyager — Automatic Curriculum, Skill Library, Code Generation

### 1.1 Overview
Voyager, introduced by Wang et al. (May 2023) from NVIDIA/Caltech/Stanford, is the first LLM-powered embodied lifelong learning agent in Minecraft [^27^]. It consists of three key components: (1) an automatic curriculum for open-ended exploration, (2) an ever-growing skill library of executable code, and (3) an iterative prompting mechanism for program improvement [^27^][^31^].

### 1.2 Prompt Structure

**System prompt** establishes the agent's role as a Minecraft assistant with code generation capabilities. The full prompt includes [^22^][^31^]:

1. **Guidelines for code generation**: Rules for generating JavaScript code using Mineflayer APIs
2. **Control primitive APIs**: A curated list of Mineflayer functions including:
   - `exploreUntil(bot, direction, maxTime, callback)`
   - `mineBlock(bot, name, count)`
   - `craftItem(bot, name, count)`
   - `placeItem(bot, name, position)`
   - `smeltItem(bot, itemName, fuelName, count)`
   - `killMob(bot, mobName, timeout)`
   - `getItemFromChest(bot, chestPosition, itemsToGet)`
   - `depositItemIntoChest(bot, chestPosition, itemsToDeposit)`
3. **Mineflayer pathfinder Goals**: `GoalNear`, `GoalXZ`, `GoalGetToBlock`, `GoalFollow`, `GoalPlaceBlock`, `GoalLookAtBlock`
4. **Mineflayer utility functions**: `bot.equip()`, `bot.consume()`, `bot.fish()`, `bot.sleep()`, `bot.activateBlock()`, `bot.lookAt()`, `bot.activateItem()`, `bot.useOn()`

**User prompt per iteration** contains [^22^]:
- Code from the last round (for iterative refinement)
- Execution error (from JavaScript interpreter)
- Chat log / environment feedback (from `bot.chat()` calls)
- Biome, time, nearby blocks, nearby entities
- Health, hunger, position, equipment
- Inventory (xx/36 slots)
- Chest contents
- Current task (from curriculum)
- Context (retrieved skills from skill library)
- Critique (from self-verification module)

The LLM responds with:
- **Explain** (if applicable): Analysis of why previous code failed
- **Plan**: Step-by-step plan to complete the task
- **Code**: Complete JavaScript code using the provided APIs

### 1.3 Tick Frequency
Voyager operates in **event-driven planning cycles**, not per-tick. The LLM is invoked:
- **Once per task** proposed by the automatic curriculum (typically every 30 seconds to 5 minutes of gameplay)
- **Up to 4 iterations** per task (iterative refinement loop)
- If the agent gets stuck after 4 rounds, the curriculum proposes a different task [^27^]

This makes Voyager relatively **sparse in LLM calls** compared to per-tick planners.

### 1.4 Action Vocabulary
**Code-as-action**: Voyager generates complete JavaScript programs that call Mineflayer APIs. This is the most expressive action space among surveyed systems — it enables temporally extended, compositional behaviors [^27^].

Example generated code [^27^]:
```javascript
async function craftStonePickaxe(bot) {
  await mineBlock(bot, "stone", 3);
  await craftItem(bot, "cobblestone", 3);
  // ... continues
}
```

### 1.5 Planner Failure Handling
Three-tier feedback mechanism [^27^]:
1. **Environment feedback**: Intermediate progress messages (e.g., "I cannot make an iron chestplate because I need: 7 more iron ingots")
2. **Execution errors**: JavaScript interpreter errors (syntax/runtime)
3. **Self-verification**: A separate GPT-4 "critic" agent checks if the task was completed successfully based on agent state. If failed, it provides critique suggesting how to complete the task.

The iterative loop runs up to 4 times before abandoning the task.

### 1.6 Memory / State Management
**Skill Library**: A vector database (keyed by GPT-3.5-generated description embeddings, valued by code programs) [^27^].
- Retrieval: Top-5 relevant skills queried by task description + environment feedback
- Addition: New skills committed only after self-verification confirms success
- Composition: Complex skills synthesized by composing simpler programs
- Storage: Persistent across sessions; transferable to new Minecraft worlds

### 1.7 LLM Cost
- **Primary model**: GPT-4 for code generation, curriculum generation, and self-verification
- **Secondary model**: GPT-3.5 for question-answering ("due to budgetary considerations") [^27^]
- Estimated cost: **Thousands of dollars per experiment** with GPT-4; reproductions with GPT-4o-mini or GPT-3.5 are significantly cheaper but with reduced capability [^13^]
- Each planning iteration involves: 1x GPT-4 call for code generation + 1x GPT-4 call for self-verification + up to 4x iterative refinement calls

---

## 2. GITM — Structured Action Primitives, LLM Planner, Action Sequence Execution

### 2.1 Overview
Ghost in the Minecraft (GITM), from OpenGVLab (May 2023), uses a hierarchical LLM-based approach with three components: LLM Decomposer, LLM Planner, and LLM Interface [^26^]. It achieved 47.5% improvement in ObtainDiamond success rate and was the first agent to procure all items in the Minecraft Overworld technology tree [^26^][^224^].

### 2.2 Prompt Structure

**LLM Decomposer prompt** (GPT-3.5-turbo) [^238^]:
```
SYSTEM: You serve as an assistant that helps me play Minecraft.
I will give you my goal in the goal, please break it down as a 
tree-structure plan to achieve this goal.
Requirements:
1. The plan tree should be exactly of depth 2.
2. Describe each step in one line.
3. Index the two levels like '1.', '1.1.', '1.2.', '2.', etc.
4. Sub-goals at the bottom level should be basic actions.

USER: Target object: {quantity} {object name}
Knowledge: {related knowledge from wiki/recipes}
```

**LLM Planner prompt** includes [^26^]:
1. **Action Interface**: Functional descriptions of structured actions and parameters
2. **Query Illustration**: Structure of user queries
3. **Response Format**: `[Explanation, Thought, Action List]` where:
   - Explanation: reason for action failure
   - Thought: chain-of-thought planning
   - Action List: structured actions to execute
4. **Interaction Guideline**: How to correct failed actions based on feedback
5. **Previous feedback**: Success/failure of prior actions + current agent state

### 2.3 Tick Frequency
GITM plans at **two timescales**:
- **Decomposition**: Once per high-level goal (creates a sub-goal tree)
- **Action planning**: Once per sub-goal (generates a sequence of structured actions)
The planner does NOT run every game tick — it generates action sequences that are executed sequentially until completion or failure [^26^].

### 2.4 Action Vocabulary
**Structured action primitives** with format `(Name, Arguments, Description)` [^26^]:

| Name | Arguments | Description |
|------|-----------|-------------|
| `equip` | object | Equip object from inventory |
| `explore` | object | Move to find objects/entities |
| `approach` | object | Move close to visible object |
| `mine/attack` | object, tool | Attack/mine object with tool |
| `dig_down` / `go_up` | tool | Go down/up with tool |
| `build` | object, tool, material | Place objects per blueprint |
| `craft/smelt` | object, tool | Craft/smelting with tool |
| `apply` | tool, object | Apply tool on object |

### 2.5 Planner Failure Handling
**Feedback-driven replanning** [^26^]:
- After each structured action execution, the LLM Interface reports success/failure
- If execution fails, the failure reason is included in the feedback message
- The LLM Planner receives: (1) action success/failure info, (2) current agent state (inventory, biome, depth), (3) previous plan context
- The planner then generates a revised action sequence using chain-of-thought reasoning

### 2.6 Memory / State Management
**Text-based memory** [^26^]:
- Successful action lists are recorded and summarized into text-based memory
- Retrieval: When facing a similar sub-goal, the planner retrieves past successful plans as reference
- Format: Text summaries of (goal, action_sequence, outcome)
- No vector embeddings — purely text-based retrieval via LLM context

### 2.7 LLM Cost
- **Decomposer**: GPT-3.5-turbo (cheapest option)
- **Planner**: GPT-3.5-turbo or GPT-4 depending on complexity
- **Interface**: Hand-written scripts (no LLM)
- **Total**: Single CPU node with 32 CPU cores, no GPU needed — "10,000x more efficient than prior RL methods" [^26^][^241^]
- Estimated API cost: **~$10-50 per full tech tree run** (estimated from GPT-3.5-turbo pricing and published iteration counts)

---

## 3. DEPS — Describe-Explain-Plan-Select, Interactive Replanning

### 3.1 Overview
DEPS (Describe, Explain, Plan and Select), from Tsinghua/CUHK (NeurIPS 2023), is an interactive planning approach based on LLMs for open-world multi-task agents [^59^][^217^]. It achieved the milestone of the first zero-shot multi-task agent completing 70+ Minecraft tasks [^59^].

### 3.2 Prompt Structure

**Initial planning prompt** [^60^]:
```
System: You are a helper agent in Minecraft. You need to generate
sequences of goals for a certain task in Minecraft. Just refer to
history dialogue to give the plan. Do not explain or give any
other instruction.

User: My current inventory has {inventory}. {visual observation}.
How to obtain {target} in Minecraft step-by-step?

Assistant: def {target_fn}(initial_inventory={}):
  mine(obj={...}, tool=...)
  craft(obj={...}, materials={...}, tool=...)
  ...
```

**Replanning prompt** (on failure) [^59^]:
- **Descriptor** output (current state summary): "Failed to mine iron ore. Current position: caves. Inventory: wooden pickaxe (durability low). No stone pickaxe available."
- **Explainer** output (error analysis): "The goal 'mine iron ore' requires at least a stone pickaxe. The wooden pickaxe cannot mine iron ore and is about to break."
- **Revised plan**: Updated sub-goal sequence incorporating the explanation

### 3.3 Tick Frequency
DEPS operates in **sub-goal execution cycles**:
- LLM planner invoked: **Once per sub-goal failure** (not every tick)
- The goal-conditioned controller `π(a|s,g)` runs at every tick
- Selector evaluates parallel sub-goals: **Once per plan revision**
- Typical task completion: 3000-12000 steps with 5-20 LLM calls total [^59^]

### 3.4 Action Vocabulary
The LLM outputs **natural language sub-goals** (not direct actions) [^59^]:
- Examples: "mine oak wood", "craft 4 wooden planks", "craft stone pickaxe"
- These are consumed by a goal-conditioned controller (MC-Controller or STEVE-1)
- The controller translates natural language goals into keyboard/mouse actions

### 3.5 Planner Failure Handling
**Interactive replanning loop** [^59^][^65^]:
1. Sub-goal `g_k` fails during execution
2. **Descriptor** (`f_DESC`): Summarizes current state `s_t` and execution outcome into text `d_t`
3. **Explainer** (`f_LM`): Analyzes `d_t` to locate errors in previous plan `P_{t-1}`
4. **Planner** (`f_LM`): Generates revised plan `P_t` incorporating explanation
5. **Selector** (`f_S`): Chooses most accessible sub-goal from parallel options
6. Controller executes new sub-goal

The LLM serves triple duty: Explainer + Planner + (in JARVIS-1 extension) Query Generator.

### 3.6 Memory / State Management
- **No persistent memory across sessions** in base DEPS
- **Horizon-predictive Selector**: A learned neural network (Impala CNN backbone) that predicts remaining steps to achieve each candidate sub-goal given current state [^59^]
- **In-session context**: Full history of (plan, description, explanation) maintained in prompt context
- Selector weights can be shared with controller policy parameters

### 3.7 LLM Cost
- **Planner model**: ChatGPT/GPT-4 (flexible)
- **Goal-conditioned controller**: Pre-trained RL model (MC-Controller or STEVE-1), no LLM at execution
- Estimated: **~5-20 LLM calls per task**, each with moderate context (few-thousand tokens)
- Cost per task: **$0.10-2.00** depending on model choice and task complexity

---

## 4. JARVIS-1 — Memory-Augmented Multimodal LLM

### 4.1 Overview
JARVIS-1 (November 2023) from the CraftJarvis team is a multimodal agent combining MineCLIP visual encoder with LLM planning, augmented with multimodal memory for life-long learning [^60^][^62^]. It achieved nearly perfect performance on 200+ tasks and 12.5% on ObtainDiamondPickaxe (5x prior records) [^62^].

### 4.2 Prompt Structure

**Planning prompt** (GPT-4) [^60^]:
```
System: Extract the action name, action type, goal object, tool and
action rank from the input text.

User (with visual observation + symbolic state):
  - Current inventory: {...}
  - Current biome: {...}
  - Current health/hunger: {...}
  - [Screenshot embedded via MineCLIP]
  Task: {task description}

Assistant: Plan as code-like function calls:
  def obtain_diamond_pickaxe(initial_inventory={}):
    mine(obj={"log":3}, tool=None)
    craft(obj={"planks":12}, materials={"log":3})
    craft(obj={"stick":4}, materials={"planks":2})
    craft(obj={"crafting_table":1}, materials={"planks":4})
    ...
```

**Self-check prompt**: Given initial plan, progressively simulate execution, predict resulting state after each step, evaluate if states satisfy goal preconditions [^60^].

**Self-explain prompt**: On failure, analyze error and locate bugs in original plan [^60^].

### 4.3 Tick Frequency
- **Planner**: Invoked once per task (generates K sub-goals), then **only on replan events**
- **Self-check**: Once per plan, before execution
- **Replanning**: On sub-goal execution failure (environment feedback triggers self-explain + replan)
- **Controller**: STEVE-1 or MineDojo API runs at environment tick rate

### 4.4 Action Vocabulary
Similar to Voyager — **code-like function definitions** with natural language sub-goals:
- `mine(obj={...}, tool=...)`
- `craft(obj={...}, materials={...}, tool=...)`
- `place(obj={...}, position=...)`
- `attack(target=..., tool=...)`
- Sub-goals are natural language: "obtain book/diamond/obsidian with empty inventory" [^60^]

### 4.5 Planner Failure Handling
**Two-layer failure recovery** [^60^]:
1. **Self-check** (proactive): Before execution, simulate plan step-by-step, predict inventory state after each step, verify preconditions
2. **Environment feedback** (reactive): On execution failure, feed failure info to MLM, use self-explanation to locate plan bugs, generate improved plan

### 4.6 Memory / State Management
**Multimodal key-value memory** [^60^]:
- **Keys**: Multimodal — (task description + observation/situation when memory was created)
- **Values**: Successfully executed plans
- **Query generation**: LLM decomposes task into sub-tasks via backward reasoning; sub-goals present in memory join current visual observation to form multimodal query
- **Retrieval**: Two-stage — (1) CLIP text embedding similarity for candidate selection, (2) CLIP visual embedding similarity for ranking
- **Self-improvement**: Memory grows with gameplay; multiple agents run in parallel to gather experiences
- **Life-long learning**: Success rates increase with memory size (empirically validated) [^60^]

### 4.7 LLM Cost
- **Primary model**: GPT-4 (best performance), ChatGPT (similar), fine-tuned LLaMA2-13B (comparable after fine-tuning)
- **Ablation**: Open-source LLaMA2-70B lacks Minecraft knowledge — requires fine-tuning on Minecraft text data [^60^]
- **Memory retrieval**: CLIP embeddings (no LLM call)
- Estimated: **~10-30 LLM calls per long-horizon task**, with rich multimodal context

---

## 5. MP5 — Active Perception, Multimodal Embodied System

### 5.1 Overview
MP5 (2024) is a five-module embodied system (Parser, Percipient, Planner, Performer, Patroller) built on MLLMs with an active perception scheme for situation-aware planning [^218^]. It achieves 91% success on complex context-dependent tasks and 22% on difficult process-dependent tasks.

### 5.2 Prompt Structure

**Parser prompt** (LLM with retrieval augmentation):
- Decomposes long-horizon task into sub-objectives
- Uses external memory for similar past decompositions

**Planner prompt** (LLM with retrieval augmentation):
- Schedules action sequences for each sub-objective
- Refines sub-objectives based on current situation
- Receives: task description, current inventory, perceived environment info

**Percipient prompt** (LoRA-enabled MLLM):
- Answers visual questions about observed images
- Open-ended visual concept understanding
- Multi-round Q&A with Patroller for active perception

### 5.3 Tick Frequency
MP5 operates with **frequent perception-action cycles**:
- **Active perception**: Multi-round Q&A between Percipient and Patroller during action execution
- **Planning**: Once per sub-objective, refined after each active perception cycle
- **Patroller checks**: After each Percipient response and after each action execution
- This is more frequent than Voyager/GITM but less than per-tick planning [^218^]

### 5.4 Action Vocabulary
**Mineflayer-based executable commands** via Performer module [^218^]:
- Structured action sequences from Planner translated into executable commands
- Actions include: move, look, attack, use, craft, place, mine, jump, equip
- Patroller verifies action appropriateness before execution

### 5.5 Planner Failure Handling
**Patroller-mediated feedback loop** [^218^]:
- Patroller checks responses from Percipient, Planner, and Performer
- Verifies plans against perceived environmental information
- Provides feedback on better strategy choices
- Enables situation-aware replanning when context changes

### 5.6 Memory / State Management
- **Parser and Planner**: Augmented with external memory (RAG)
- **Temporary Env. Info Set**: Resets per active perception cycle; stores current perceived information
- **Percipient**: LoRA-fine-tuned MLLM for Minecraft-specific visual understanding
- No long-term episodic memory like JARVIS-1

### 5.7 LLM Cost
- **Parser/Planner/Patroller**: GPT-3.5-turbo or GPT-4
- **Percipient**: LoRA-enabled multimodal LLM (local inference)
- Multiple LLM calls per sub-objective due to active perception Q&A rounds
- Estimated: **~20-50 LLM calls per complex task**

---

## 6. Optimus-3 (2025) — MoE with Task Routing, Multimodal Reasoning, RL Enhancement

### 6.1 Overview
Optimus-3 (June 2025) is the first generalist agent in Minecraft with comprehensive perception, planning, action, grounding, and reflection capabilities [^5^]. It introduces a task-level routing Mixture-of-Experts (MoE) architecture and multimodal reasoning-augmented RL.

### 6.2 Prompt Structure / Architecture
Optimus-3 is an **end-to-end trained model**, not a prompt-based system. Its architecture includes [^5^]:
- **Task Router**: Sentence-BERT classifier routes queries to task-specific experts
- **Shared Knowledge Expert**: Always activated, captures cross-task knowledge
- **Task-Specific Experts**: Dedicated experts for Planning, Captioning, Embodied QA, Grounding, Reflection
- **ViT**: Visual encoding
- **VPT action head**: Low-level action control in Minecraft

### 6.3 Tick Frequency
- **Inference**: Once per environment step for action generation
- **Planning**: Integrated into the model — no separate LLM API calls
- **MoE routing**: Per-query, not per-token (task-level routing)

### 6.4 Action Vocabulary
- **Low-level actions**: Direct keyboard/mouse outputs via VPT action head
- **Text outputs**: Planning, captioning, QA, reflection via MoE LLM decoder
- **Bounding boxes**: For grounding tasks

### 6.5 Planner Failure Handling
- **Reflection task expert**: Dedicated expert for self-reflection on failures
- **Reinforcement learning**: GRPO-based training with dependency-aware rewards for planning
- **Self-correction**: Model learns to iteratively refine reasoning via RL [^5^]

### 6.6 Memory / State Management
- **Knowledge-enhanced data generation pipeline**: Automated dataset creation using knowledge graphs + STEVE-1 + expert models
- **No explicit external memory** — knowledge encoded in MoE parameters
- Task expansion supported by adding new task experts without forgetting

### 6.7 LLM Cost
- **Training**: Significant compute for MoE training (7B parameter activated model)
- **Inference**: Local GPU inference — **zero API cost per gameplay hour**
- **Data generation**: One-time GPT-4/STEVE-1 calls for dataset creation
- Most cost-effective at scale for long-running agents

### 6.8 Performance
Outperforms previous SOTA: +20% Planning, +3% Long-Horizon Action, +66% Captioning, +76% Embodied QA, +3.4x Grounding, +18% Reflection [^5^].

---

## 7. AutoGPT / AgentGPT — Adaptations for Minecraft

### 7.1 Overview
AutoGPT is a popular open-source framework for autonomous AI agents that decomposes high-level goals into subgoals and executes them in a ReAct-style loop [^222^][^22^]. In Voyager's evaluation, AutoGPT was adapted for Minecraft by using GPT-4 for task decomposition with agent states, environment feedback, and execution errors as observations [^22^].

### 7.2 Prompt Structure
AutoGPT uses a **goal-decomposition loop** [^222^]:
```
System: You are an autonomous agent. Given a goal, decompose it
into actionable sub-goals and execute them sequentially.

Current state: {agent state}
Completed tasks: {...}
Failed tasks: {...}
Next sub-goal: {...}
```

### 7.3 Tick Frequency
**Per-sub-goal invocation** — similar frequency to Voyager's curriculum but without automatic curriculum generation. The LLM is called for each sub-goal decomposition and execution check.

### 7.4 Action Vocabulary
In Minecraft adaptations, AutoGPT typically outputs **natural language sub-goals** or **structured JSON actions** that map to environment APIs [^22^].

### 7.5 Planner Failure Handling
AutoGPT has **limited failure recovery** compared to specialized Minecraft agents:
- No built-in skill library (key limitation per Voyager comparison) [^22^]
- No self-verification module
- Relies on ReAct-style reasoning traces for error recovery
- Prone to infinite loops (documented 14% autonomous completion rate, 86% termination in loops) [^296^]

### 7.6 Memory / State Management
- Short-term: Conversation context within LLM window
- Long-term: File-based memory storage (can write/read files)
- No structured skill library or multimodal memory

### 7.7 LLM Cost
AutoGPT is **notoriously expensive** for Minecraft tasks:
- Averaged **47 LLM calls per task** for simple tasks [^296^]
- **~$12 in API costs per simple task** (GPT-4-turbo pricing) [^296^]
- Minecraft adaptation would be significantly higher due to longer task horizons
- Estimated: **$50-200+ per hour of Minecraft gameplay**

---

## 8. ReAct / Reflexion — Applied to Embodied Tasks

### 8.1 Overview
**ReAct** (Yao et al., 2022) interleaves reasoning traces with actions using chain-of-thought prompting [^220^]. **Reflexion** (Shinn et al., 2023) extends ReAct with self-reflection — after task completion, a reflection model generates verbal reinforcement cues for future trials [^248^][^250^].

### 8.2 Prompt Structure

**ReAct instruction** [^220^]:
```
You should first think about the current condition and plan for your
future actions, and then output your action in this turn.

Format:
Thought: {reasoning about current state}
Action: {next action}
```

**Reflexion adds** (after episode) [^248^]:
```
Evaluator: Scores the trajectory with a reward signal
Self-Reflection Model: Generates verbal feedback:
  "The previous approach failed because X. Next time, try Y instead."
This reflection is stored in memory and provided as context for the next episode.
```

**ReflAct** (2025 enhancement) [^220^]:
```
Instruction: "You should first reflect on the agent's state in
relation to the task goal, and then output the action for this turn."

Format:
Reflection: {assessment of current state vs. goal}
Action: {next action}
```

### 8.3 Tick Frequency
- **ReAct**: LLM invoked **every step** (expensive for Minecraft)
- **Reflexion**: LLM every step + additional reflection LLM call after each episode
- **ReflAct**: LLM every step with reflection-format reasoning

### 8.4 Action Vocabulary
Natural language actions or structured primitives depending on the environment adapter:
- ALFWorld: `go to {location}`, `pick up {obj}`, `use {obj} on {obj}`
- Minecraft (via Mineflayer): API function calls or high-level commands

### 8.5 Planner Failure Handling
- **ReAct**: Relies on reasoning traces to recover from errors mid-episode
- **Reflexion**: Explicit self-reflection after failure, stored in episodic memory for future trials
- **ReflAct**: Goal-oriented reflection at every step prevents deviation (36.4% improvement over ReAct on ALFWorld) [^220^]

### 8.6 Memory / State Management
- **ReAct**: Context window only (no persistent memory)
- **Reflexion**: Long-term memory of self-reflections, organized by task
- **ReflAct**: Reflection + action pairs stored in trajectory history

### 8.7 LLM Cost
- **ReAct**: Highest cost due to per-step LLM calls
- Minecraft embodiment: **$20-100+ per hour** depending on environment complexity
- Reflexion adds one LLM call per episode for reflection generation

---

## 9. Cross-System Comparison Table

| Dimension | Voyager | GITM | DEPS | JARVIS-1 | MP5 | Optimus-3 |
|-----------|---------|------|------|----------|-----|-----------|
| **Planner LLM** | GPT-4 | GPT-3.5/4 | GPT-3.5/4 | GPT-4 | GPT-3.5/4 | End-to-end MoE |
| **Action vocab** | Code (JS) | Structured primitives | Natural language sub-goals | Code-like functions | Mineflayer cmds | Low-level actions |
| **Tick frequency** | Per-task (~30s-5min) | Per-sub-goal | Per-sub-goal failure | Per-task + replan | Per-sub-objective | Per environment step |
| **LLM calls/task** | 4-8 | 3-10 | 5-20 | 10-30 | 20-50 | 0 (local) |
| **Failure detection** | Self-verification (GPT-4 critic) | Action success/failure feedback | Descriptor+Explainer loop | Self-check + env feedback | Patroller verification | Reflection expert |
| **Memory type** | Vector skill library | Text memory | Horizon-predictive selector | Multimodal key-value memory | RAG-augmented | MoE parameters |
| **Cost/task** | $2-10 | $0.50-5 | $0.10-2 | $1-5 | $2-10 | $0 (after training) |
| **Strength** | Lifelong learning, compositionality | Efficiency, tech tree completion | Interactive replanning, zero-shot | Multimodal memory, life-long learning | Active perception, situation awareness | Generalist, no API cost |
| **Weakness** | High API cost | Depends on LiDAR, limited perception | No cross-session memory | Complex retrieval pipeline | Many LLM calls | Requires training data |

---

## 10. Recommended Interface Design: LLM Planner + Per-Role RL Policies

Based on the architectural patterns extracted from the surveyed systems, we recommend a **hierarchical subgoal specification interface** that combines the best elements of DEPS's goal-conditioned policies, Voyager's code-as-action composability, and JARVIS-1's multimodal memory.

### 10.1 Design Philosophy

**Core insight from the survey**: The most successful systems separate **planning** (what to do) from **execution** (how to do it). When the LLM only specifies subgoals and a low-level controller handles motor execution, both components can be optimized independently [^59^][^26^][^244^].

For a system with **per-role RL policies** (e.g., a mining policy, a combat policy, a building policy), the LLM should communicate:
1. **Which role** to activate (policy selection)
2. **What subgoal** to achieve (goal specification)
3. **What constraints** to respect (safety/efficiency bounds)
4. **How long** to attempt before giving up (timeout)

### 10.2 LLM Output Format (JSON Schema)

```json
{
  "$schema": "llm_plan_output",
  "description": "Structured plan output from LLM planner to RL controller",
  "type": "object",
  "properties": {
    "plan_id": {
      "type": "string",
      "description": "Unique identifier for this plan revision"
    },
    "high_level_goal": {
      "type": "string",
      "description": "Original user-specified goal, e.g., 'build a village'"
    },
    "subgoals": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "subgoal_id": { "type": "string" },
          "role": {
            "type": "string",
            "enum": ["miner", "lumberjack", "builder", "fighter", "explorer", "crafter", "farmer"],
            "description": "Which RL policy role to activate"
          },
          "goal_specification": {
            "type": "object",
            "properties": {
              "target_state": {
                "type": "object",
                "description": "Desired state after subgoal completion",
                "properties": {
                  "inventory_delta": {
                    "type": "object",
                    "example": { "oak_log": 16, "stone": 32 }
                  },
                  "location": {
                    "type": "object",
                    "properties": {
                      "type": { "type": "string", "enum": ["coordinates", "biome", "structure", "relative"] },
                      "value": { "type": "string", "example": "plains_biome" }
                    }
                  },
                  "structures_built": {
                    "type": "array",
                    "items": { "type": "string" },
                    "example": ["house_foundation"]
                  }
                }
              },
              "termination_conditions": {
                "type": "object",
                "properties": {
                  "success_criteria": {
                    "type": "array",
                    "items": { "type": "string" },
                    "example": ["inventory_contains(oak_log, 16)", "distance_to_water < 10"]
                  },
                  "timeout_ticks": { "type": "integer", "example": 6000 },
                  "failure_events": {
                    "type": "array",
                    "items": { "type": "string" },
                    "example": ["health_below(3)", "inventory_full"]
                  }
                }
              }
            }
          },
          "constraints": {
            "type": "object",
            "properties": {
              "preserve_items": { "type": "array", "items": { "type": "string" } },
              "avoid_biomes": { "type": "array", "items": { "type": "string" } },
              "max_health_cost": { "type": "number" },
              "tool_requirements": { "type": "array", "items": { "type": "string" } }
            }
          },
          "fallback_subgoals": {
            "type": "array",
            "description": "Alternative subgoals if this one fails",
            "items": { "type": "string" },
            "example": ["gather_spruce_log_instead", "trade_for_wood"]
          }
        },
        "required": ["subgoal_id", "role", "goal_specification"]
      }
    },
    "dependencies": {
      "type": "array",
      "description": "Ordering constraints between subgoals",
      "items": {
        "type": "object",
        "properties": {
          "before": { "type": "string" },
          "after": { "type": "string" }
        }
      }
    }
  },
  "required": ["plan_id", "high_level_goal", "subgoals"]
}
```

**Example plan for "build a village"**:
```json
{
  "plan_id": "plan_village_001_rev2",
  "high_level_goal": "build a village with 3 houses and a well",
  "subgoals": [
    {
      "subgoal_id": "sg_01",
      "role": "lumberjack",
      "goal_specification": {
        "target_state": { "inventory_delta": { "oak_log": 64 } },
        "termination_conditions": {
          "success_criteria": ["inventory_contains(oak_log, 64)"],
          "timeout_ticks": 6000,
          "failure_events": ["health_below(5)", "night_falls"]
        }
      },
      "constraints": { "avoid_biomes": ["dark_forest"], "max_health_cost": 2.0 },
      "fallback_subgoals": ["gather_birch_log", "gather_spruce_log"]
    },
    {
      "subgoal_id": "sg_02",
      "role": "miner",
      "goal_specification": {
        "target_state": { "inventory_delta": { "cobblestone": 64 } },
        "termination_conditions": {
          "success_criteria": ["inventory_contains(cobblestone, 64)"],
          "timeout_ticks": 8000,
          "failure_events": ["health_below(4)", "pickaxe_broken"]
        }
      },
      "constraints": { "tool_requirements": ["stone_pickaxe_or_better"] }
    },
    {
      "subgoal_id": "sg_03",
      "role": "builder",
      "goal_specification": {
        "target_state": { "structures_built": ["house_1", "house_2", "house_3", "well"] },
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

### 10.3 RL Policy Input Format (Goal Embedding + Observation)

```json
{
  "$schema": "rl_policy_input",
  "description": "Input to per-role RL policy from planner",
  "type": "object",
  "properties": {
    "role_id": { "type": "string", "example": "lumberjack" },
    "goal_embedding": {
      "type": "object",
      "description": "Dense vector + structured goal representation",
      "properties": {
        "vector": {
          "type": "array",
          "items": { "type": "number" },
          "description": "Fixed-dim goal embedding (e.g., 256-dim from sentence transformer)"
        },
        "structured": {
          "type": "object",
          "description": "Original goal_specification from LLM plan"
        }
      }
    },
    "observation": {
      "type": "object",
      "properties": {
        "visual": {
          "type": "string",
          "description": "Base64-encoded RGB frame or CLIP embedding"
        },
        "inventory": {
          "type": "object",
          "description": "Current inventory state"
        },
        "player_state": {
          "type": "object",
          "properties": {
            "position": { "type": "array", "items": { "type": "number" } },
            "health": { "type": "number" },
            "hunger": { "type": "number" },
            "armor": { "type": "number" }
          }
        },
        "nearby_entities": {
          "type": "array",
          "items": { "type": "string" }
        },
        "nearby_blocks": {
          "type": "array",
          "items": { "type": "string" }
        },
        "time_of_day": { "type": "string" },
        "biome": { "type": "string" }
      }
    },
    "execution_context": {
      "type": "object",
      "properties": {
        "ticks_elapsed": { "type": "integer" },
        "plan_id": { "type": "string" },
        "subgoal_id": { "type": "string" },
        "previous_actions": {
          "type": "array",
          "items": { "type": "string" },
          "description": "Last N actions for temporal context"
        }
      }
    }
  },
  "required": ["role_id", "goal_embedding", "observation"]
}
```

### 10.4 Failure Reporting Format (RL Policy → LLM)

```json
{
  "$schema": "failure_report",
  "description": "Report from RL policy back to LLM planner on subgoal failure",
  "type": "object",
  "properties": {
    "report_id": { "type": "string" },
    "plan_id": { "type": "string" },
    "subgoal_id": { "type": "string" },
    "role": { "type": "string" },
    "status": { "type": "string", "enum": ["success", "failure", "partial", "timeout"] },
    "failure_details": {
      "type": "object",
      "properties": {
        "failure_type": {
          "type": "string",
          "enum": ["timeout", "health_critical", "tool_broken", "inventory_full",
                   "path_blocked", "resource_unavailable", "attacked", "unknown"]
        },
        "failure_tick": { "type": "integer" },
        "final_state": {
          "type": "object",
          "properties": {
            "position": { "type": "array", "items": { "type": "number" } },
            "health": { "type": "number" },
            "inventory": { "type": "object" },
            "nearby_threats": { "type": "array", "items": { "type": "string" } }
          }
        },
        "execution_trace": {
          "type": "array",
          "description": "Sequence of (action, observation, reward) tuples",
          "items": {
            "type": "object",
            "properties": {
              "tick": { "type": "integer" },
              "action": { "type": "string" },
              "observation_summary": { "type": "string" },
              "reward": { "type": "number" }
            }
          }
        },
        "descriptor_summary": {
          "type": "string",
          "description": "Natural language summary of what happened (auto-generated)",
          "example": "Failed to gather 64 oak logs. Health dropped to 3 after zombie attack at tick 2340. Inventory has 23 oak logs. Night fell at tick 2100."
        }
      }
    },
    "partial_progress": {
      "type": "object",
      "description": "What was achieved before failure",
      "properties": {
        "inventory_delta": { "type": "object" },
        "structures_built": { "type": "array" },
        "exploration_coverage": { "type": "number" }
      }
    }
  },
  "required": ["plan_id", "subgoal_id", "status"]
}
```

### 10.5 Replanning Trigger Design

Based on the survey, we recommend a **multi-criteria trigger** [^59^][^60^][^27^]:

| Trigger | Source | Priority | Action |
|---------|--------|----------|--------|
| **Subgoal timeout** | RL policy reports timeout | Critical | Replan with adjusted goal or fallback |
| **Health critical** | `health < threshold` | Critical | Emergency replan (retreat/survive first) |
| **Tool broken** | Inventory change detected | High | Replan to craft/replace tool |
| **Inventory full** | Inventory check | Medium | Insert "deposit items" subgoal |
| **Unexpected attack** | Environment event | High | Interrupt current subgoal, activate fighter role |
| **State deviation** | Progress tracking shows <50% after 50% timeout | Medium | Adjust subgoal parameters |
| **Periodic review** | Every 5 minutes or every N subgoals | Low | Full plan reassessment opportunity |

### 10.6 Prompt Context Design (What the LLM Sees)

The LLM planner should receive [^27^][^59^][^60^]:

```
[SYSTEM PROMPT]
You are a Minecraft strategic planner. Your role is to decompose
high-level goals into executable subgoals for specialized RL policies.
Available roles: {miner, lumberjack, builder, fighter, explorer, crafter, farmer}
Each role has a trained RL policy that executes low-level actions.

You output structured JSON plans following the schema: {...}

[KNOWLEDGE BASE]
- Crafting recipes relevant to current goal
- Biome information and resource distributions
- Threat assessments for current region

[CURRENT STATE]
- Position: (x, y, z), Biome: {biome}
- Health: {health}/20, Hunger: {hunger}/20
- Inventory: {inventory}
- Equipment: {held_item, armor}
- Time: {time}, Weather: {weather}
- Nearby blocks (32-block radius): [...]
- Nearby entities: [...]

[PLAN HISTORY]
- Current plan: {plan_id}, executing subgoal: {subgoal_id}
- Completed subgoals: [...]
- Failed subgoals with reports: [...]

[MEMORY RETRIEVAL]
- Similar past plans (from vector DB): [...]
- Successful strategies for similar situations: [...]

[USER GOAL]
{high_level_goal}

[INSTRUCTION]
Generate a revised plan (or initial plan) as JSON. Consider:
1. What resources are already available
2. What threats exist in the environment
3. What fallback options exist for each subgoal
4. Optimal ordering of subgoals based on dependencies
```

### 10.7 LLM Cost Estimation

| Configuration | Calls per hour | Cost per hour (GPT-4) | Cost per hour (GPT-4o-mini) |
|---------------|---------------|----------------------|---------------------------|
| **Sparse replan** (replan only on failure, ~10 calls/hour) | 10 | ~$2-5 | ~$0.20-0.50 |
| **Periodic review** (+ every 5 min review, ~22 calls/hour) | 22 | ~$5-10 | ~$0.50-1.00 |
| **Dense replan** (+ proactive adaptation, ~50 calls/hour) | 50 | ~$15-30 | ~$1.50-3.00 |
| **With open-source model** (fine-tuned LLaMA-3-8B) | N/A | ~$0 (GPU amortized) | ~$0 (GPU amortized) |

**Cost reduction strategies** (based on survey findings):
1. **AgenticCache**: Cache validated plans, achieve 79% cost savings [^297^]
2. **Two-model pattern**: Route 80% of tasks to GPT-4o-mini, 20% to GPT-4 [^261^]
3. **Fine-tuned open-source**: MineMA-8B achieves Voyager+GPT-4o-mini performance at GPU cost only [^13^]
4. **Skill library reuse**: Voyager's approach reduces new-skill generation by retrieving existing ones [^27^]

---

## 11. Key Findings Summary

### 11.1 Prompt Structure Patterns

**Universal components across all systems**:
1. **System instruction**: Role definition + output format specification
2. **Environment state**: Inventory, health, position, nearby entities/blocks
3. **Task specification**: Current goal/sub-goal
4. **History**: Previous actions and outcomes
5. **Knowledge**: Crafting recipes, biome info, external wiki knowledge [^27^][^26^][^59^][^60^]

**Differentiation**:
- Voyager adds **retrieved skills** and **critique** from self-verification
- DEPS adds **descriptor summary** and **explanation** of failure
- GITM adds **text-based knowledge** from wiki and **structured action definitions**
- JARVIS-1 adds **multimodal memory retrieval** (visual + text)

### 11.2 Action Vocabulary Taxonomy

Three paradigms emerged:
1. **Code-as-action** (Voyager): Most expressive, composable, but requires API knowledge
2. **Structured primitives** (GITM): Most reliable, clear semantics, hand-scripted execution
3. **Natural language sub-goals** (DEPS, JARVIS-1): Most flexible, consumed by goal-conditioned policies

For LLM + RL policy systems, **natural language sub-goals with structured JSON constraints** offers the best balance — interpretable by humans, parseable by systems, and flexible enough for diverse RL policy interfaces.

### 11.3 Planner Failure Handling Patterns

| System | Detection | Diagnosis | Recovery |
|--------|-----------|-----------|----------|
| Voyager | GPT-4 critic | Self-verification critique | Iterative code refinement |
| GITM | Action feedback | LLM chain-of-thought | Regenerate action sequence |
| DEPS | Controller failure | LLM explainer | Revised plan + selector |
| JARVIS-1 | Self-check + env feedback | Self-explanation | Memory-augmented replan |
| MP5 | Patroller verification | Active perception | Situation-aware replan |

**Key insight**: The most robust systems combine **proactive self-check** (verify before executing) with **reactive environment feedback** (adapt after failure) [^60^].

### 11.4 Memory Architecture Patterns

| System | Storage | Retrieval | Lifetime |
|--------|---------|-----------|----------|
| Voyager | Vector DB (skill embeddings) | Similarity search | Persistent, cross-world |
| GITM | Text summaries | Text matching | Session-level |
| DEPS | Selector network weights | State-conditioned | Learned, persistent |
| JARVIS-1 | Multimodal key-value | CLIP similarity | Persistent, grows with play |
| MP5 | RAG external KB | Text similarity | External |

### 11.5 LLM Cost Patterns

**Cost drivers** (in order of impact):
1. **Call frequency**: Per-tick (ReAct) >> per-subgoal (GITM) >> per-task (Voyager)
2. **Model choice**: GPT-4 >> GPT-4o-mini >> GPT-3.5 >> open-source
3. **Context length**: Multimodal (JARVIS-1) >> code + skills (Voyager) >> text only (GITM)
4. **Self-verification**: Extra LLM call per iteration (Voyager critic)

**Practical ranges for Minecraft agents**:
- Budget ($0.50-2/hour): Fine-tuned open-source or GPT-4o-mini with caching
- Standard ($5-15/hour): GPT-4o-mini or GPT-3.5 with moderate replanning
- Premium ($20-50/hour): GPT-4 with full multimodal context and frequent replanning

---

## 12. Open Questions

1. **How fine-grained should the LLM-RL interface be?** DEPS uses natural language sub-goals; Voyager generates full programs. For per-role RL policies, should the LLM specify high-level goals only, or also provide low-level action hints?

2. **What is the optimal replanning frequency?** Too sparse leads to persistent failure; too dense wastes API budget. Is there a principled way to determine replanning triggers based on uncertainty estimates?

3. **How to handle multi-role coordination?** When multiple RL policies need to coordinate (e.g., builder needs lumberjack to deliver wood), should the LLM manage the handoff, or should policies learn to communicate?

4. **Can we eliminate LLM API costs entirely?** Optimus-3 shows end-to-end trained models can match LLM-based planners. What is the minimum scale of training data needed to replace LLM planning for a specific task distribution?

5. **How to transfer learned skills across world seeds?** Voyager's skill library transfers to new worlds. Can RL policy parameters similarly transfer, or do they overfit to specific terrain?

6. **What is the right memory granularity?** JARVIS-1 stores full plans; Voyager stores code functions; GITM stores text summaries. For per-role RL policies, should memory store successful trajectories, successful parameter updates, or something else?

7. **How to balance exploration vs. exploitation in plan generation?** Voyager's automatic curriculum drives exploration, but for goal-directed tasks ("build a village"), how should the LLM balance following proven plans vs. exploring novel approaches?

---

## References

[^27^] Wang, G., et al. (2023). "Voyager: An Open-Ended Embodied Agent with Large Language Models." arXiv:2305.16291. https://arxiv.org/abs/2305.16291

[^31^] Wang, G., et al. (2023). "VOYAGER: An Open-Ended Embodied Agent with Large Language Models." PDF. https://arxiv.org/pdf/2305.16291

[^22^] Wang, G., et al. (2023). "An Open-Ended Embodied Agent with Large Language Models." Appendix with full prompts. https://arxiv.org/html/2305.16291

[^26^] Zhu, X., et al. (2023). "Ghost in the Minecraft: Generally Capable Agents for Open-World Environments via Large Language Models with Text-based Knowledge and Memory." arXiv:2305.17144. https://arxiv.org/pdf/2305.17144

[^224^] OpenGVLab. (2023). "GITM GitHub Repository." https://github.com/OpenGVLab/GITM

[^238^] GITM paper (extended). "Ghost in the Minecraft." https://hal.science/hal-04107105v2/file/GITM_hal.pdf

[^59^] Wang, Z., et al. (2023). "Describe, Explain, Plan and Select: Interactive Planning with Large Language Models Enables Open-World Multi-Task Agents." NeurIPS 2023. https://arxiv.org/abs/2302.01560

[^217^] Wang, Z., et al. (2023). "DEPS." NeurIPS 2023 Proceedings. https://proceedings.neurips.cc/paper_files/paper/2023/hash/6b8dfb8c0c12e6fafc6c256cb08a5ca7-Abstract-Conference.html

[^58^] Wang, Z., et al. (2023). "DEPS Extended Version." https://arxiv.org/html/2302.01560v3

[^60^] Wang, Z., et al. (2023). "JARVIS-1: Open-world Multi-task Agents with Memory-Augmented Multimodal Language Models." arXiv:2311.05997. https://arxiv.org/html/2311.05997

[^62^] JARVIS-1 Discussion. Reddit r/singularity. https://www.reddit.com/r/singularity/comments/17uk4dx/jarvis1_openworld_multitask_agents_with/

[^218^] Qin, Y., et al. (2023). "MP5: A Multi-modal Open-ended Embodied System in Minecraft via Active Perception." arXiv:2312.07472. https://arxiv.org/html/2312.07472v4

[^5^] Optimus-3 Team. (2025). "Optimus-3: Towards Generalist Multimodal Minecraft Agents with Scalable Task Experts." arXiv:2506.10357. https://arxiv.org/html/2506.10357v1

[^42^] Optimus-3 (v2). (2025). "Optimus-3: Dual-Router Aligned Mixture-of-Experts Agent." https://arxiv.org/html/2506.10357v2

[^220^] Kim, J., et al. (2025). "ReflAct: World-Grounded Decision Making in LLM Agents." EMNLP 2025. https://aclanthology.org/2025.emnlp-main.1697.pdf

[^248^] Shinn, N., et al. (2023). "Reflexion: Language Agents with Verbal Reinforcement Learning." NeurIPS 2023. https://www.promptingguide.ai/techniques/reflexion

[^250^] Shinn, N. (2023). "Reflexion GitHub Repository." https://github.com/noahshinn/reflexion

[^222^] Significant Gravitas. (2023). "AutoGPT: Build, Deploy, and Run AI Agents." https://github.com/significant-gravitas/autogpt

[^296^] Comanesh, A. (2026). "The Planning-Rubicon." Medium. https://medium.com/@anicomanesh/the-planning-rubicon-why-the-vast-majority-of-ai-agents-are-just-expensive-chatbots-part-i-fa0409a10d8e

[^244^] (2024). "Guiding RL Agents with High-Level Language Prompts." arXiv:2410.08632. https://arxiv.org/html/2410.08632v1

[^13^] (2025). "Empowering Minecraft Agents with Open-World Skills." IJCAI 2025. https://www.ijcai.org/proceedings/2025/0022.pdf

[^261^] Morph. (2026). "LLM API Comparison 2026." https://www.morphllm.com/llm-api

[^297^] (2026). "AgenticCache: Reducing LLM Inference Latency and Cost for Embodied Agents." arXiv:2604.24039. https://arxiv.org/html/2604.24039v1

[^275^] (2025). "Large Language Model Agents: A Comprehensive Survey." https://www.preprints.org/manuscript/202512.2119

[^277^] (2024). "A Survey on Large Language Model Based Game Agents." arXiv:2404.02039. https://arxiv.org/html/2404.02039v2

[^288^] (2026). "Experience Transfer for Multimodal LLM Agents in Minecraft Game." arXiv:2604.05533. https://arxiv.org/html/2604.05533v1

[^289^] (2025). "A Scalable Design Pattern for Agentic AI Systems." arXiv:2505.06817. https://arxiv.org/html/2505.06817v1

[^221^] (2025). "Diving into Minecraft Embodied AI - WoRV Tech Blog." https://worv.ghost.io/en-diving-into-minecraft-embodied-ai-exploring-recent-research-in-team-craftjarvis/

[^293^] PrismarineJS. "Mineflayer Documentation." https://prismarinejs.github.io/mineflayer/

[^287^] (2023). "Empowering Agents with Imagination for Creative Tasks." arXiv:2312.02519. https://arxiv.org/pdf/2312.02519

[^295^] (2025). "AgentSquare: Automatic LLM Agent Search in Modular Design Space." https://fi.ee.tsinghua.edu.cn/publications/7e80fb36-f064-11f0-b382-2aaffa21d846.pdf

[^14^] Voyager Project Page. https://voyager.minedojo.org/

[^166^] CraftJarvis. "MC-Planner GitHub." https://github.com/CraftJarvis/MC-Planner

[^239^] Nottingham, K., et al. (2023). "DECKARD Agent." https://deckardagent.github.io/

[^243^] (2025). "Option Discovery Using LLM-guided Semantic Hierarchical Reinforcement Learning." arXiv:2503.19007. https://arxiv.org/html/2503.19007v1

[^245^] (2024). "Dynamic Symbolic Representation and LLM to Enhance Hierarchical RL." https://openreview.net/pdf?id=DM6q3YPWqG

[^276^] Dagan, G., et al. (2024). "Plancraft: an evaluation dataset for planning with LLM agents." https://homepages.inf.ed.ac.uk/alex/papers/colm.pdf

[^290^] (2025). "VistaWise: Building Cost-Effective Agent with Cross-Modal Memory." EMNLP 2025. https://aclanthology.org/2025.emnlp-main.1111.pdf

---

*Report compiled from 20+ independent web searches across academic papers (arXiv, NeurIPS, IJCAI, EMNLP), GitHub repositories, and technical blog posts. Focus on extractable architectural patterns for production LLM+RL hybrid systems.*
