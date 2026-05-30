# Comprehensive Survey: Minecraft AI Agents — RL and LLM Approaches

**Research Dimension 1: Prior Art in Minecraft RL and LLM Agents**

**Date:** 2025-07-10
**Scope:** 14 major projects spanning 2019-2026, covering both reinforcement learning-based and LLM-based approaches

---

## Executive Summary

- **The field has shifted decisively from RL to LLM-based agents.** Projects from 2019-2022 (MineRL, VPT) focused on sample-efficient RL and imitation learning with limited success on long-horizon tasks. Starting in 2023, LLM-powered agents (Voyager, GITM, DEPS, JARVIS-1) achieved dramatically better results by using language models for planning, with structured low-level controllers for execution.

- **Voyager (2023) remains the most influential codebase** for anyone building a Minecraft agent today. Its code-as-policy paradigm, skill library with vector embeddings, and automatic curriculum have been adopted or extended by nearly every subsequent LLM agent project. The repo is actively maintained under MIT license. [^25^](https://github.com/MineDojo/Voyager)

- **For multi-agent village building, three projects stand out:** Voyager (composable skill library), Optimus-3 (latest MoE generalist with multi-task routing), and Odyssey (explicit multi-agent benchmark with building tasks). VillagerAgent (2024) is the most directly relevant multi-agent framework but was not in the original list. [^188^](https://aclanthology.org/2025.findings-emnlp.777.pdf)

- **Code availability is surprisingly good but maintenance varies.** Most major projects (Voyager, JARVIS-1, Optimus-3, Plan4MC, Odyssey, STEVE-1) have public GitHub repos. However, many repos have stale dependencies, abandoned issues, or incomplete feature releases. Only CraftJarvis-team repos (Optimus series, JARVIS-1) show consistent active maintenance through 2025. [^169^](https://github.com/CraftJarvis)

- **The biggest gap for village-building is multi-agent coordination.** Nearly all surveyed projects are single-agent. Multi-agent work (VillagerAgent, AgentVerse, Gated Coordination) is nascent and mostly from 2024-2026. This represents a major opportunity. [^188^](https://aclanthology.org/2025.findings-emnlp.777.pdf)

---

## Key Findings

### 1. MineRL (2019) — Guss et al., NeurIPS Competition Framework

**Paper:** "MineRL: A Large-scale Dataset of Minecraft Demonstrations" (IJCAI 2019) and "The MineRL Competition on Sample Efficient Reinforcement Learning Using Human Priors" (NeurIPS 2019 Competition Track) [^23^](https://ar5iv.labs.arxiv.org/html/2101.11071)

**Architecture Summary:**
MineRL is not an agent but a **benchmark framework and dataset** that enabled much of the subsequent Minecraft RL research. It provides: (1) the MineRL environment (wrapper around Malmo/Minecraft with human-like action space), (2) a large-scale dataset of 60M+ state-action pairs from human demonstrations across 9 tasks, and (3) a competition track focused on sample-efficient RL using human priors.

**Observation/Action Space:**
- **Observation:** RGB images (64x64), inventory state, compass angle, crafting grid
- **Action:** Discrete action space mimicking human controls — movement (forward/back/left/right/jump/sneak/sprint), camera (pitch/yaw), attack/use, and inventory/crafting actions. The full action space is large and compound, making exploration extremely difficult.
- **Tasks:** Hierarchical tasks including Treechop, ObtainIronPickaxe, and the grand challenge ObtainDiamond

**What Worked:**
- Released the largest-ever dataset of human demonstrations in an embodied domain at the time (60M+ state-action pairs) [^28^](https://arxiv.org/pdf/2007.02701)
- The competition successfully catalyzed research in sample-efficient RL, with winning solutions demonstrating that imitation learning alone could outperform RL methods on some tasks
- The MineRL environment became the de facto standard for RL research in Minecraft, used by dozens of subsequent papers

**What Failed:**
- Even with human demonstrations, the ObtainDiamond task remained extremely challenging. The best competition submissions achieved only modest success rates [^33^](http://proceedings.mlr.press/v123/scheller20a/scheller20a.pdf)
- Naive RL (Rainbow) was unable to outperform random policies on most tasks, demonstrating the extreme difficulty of exploration in the human action space [^28^](https://arxiv.org/pdf/2007.02701)
- The environment speed and Java dependency created significant engineering bottlenecks

**Code Availability:**
- **GitHub:** https://github.com/minerllabs/minerl
- **License:** MIT
- **Status:** The repo has seen limited maintenance since 2022. Many issues go unanswered. The project is effectively in maintenance mode. The underlying MineRL environment is built on Malmo which itself is deprecated.
- **Last meaningful commit:** 2022-2023 period

**Applicability to Multi-Agent:**
Low direct applicability. MineRL was designed for single-agent tasks. The environment does not natively support multi-agent scenarios. However, the dataset of human demonstrations could theoretically be used to train behavioral priors for multi-agent agents.

---

### 2. MineDojo (2022/2023) — Fan et al., Unified Benchmark

**Paper:** "MineDojo: Building Open-Ended Embodied Agents with Internet-Scale Knowledge" (NeurIPS 2022 Datasets and Benchmarks) [^134^](https://ar5iv.labs.arxiv.org/html/2206.08853)

**Architecture Summary:**
MineDojo is a **comprehensive simulation framework and benchmark suite** featuring 3000+ programmatic tasks built on top of Minecraft. Its key innovation is integrating internet-scale knowledge (YouTube videos, Wiki pages, Reddit discussions) with the environment. It includes:
- MineCLIP: A contrastive vision-language model trained on Minecraft YouTube videos to provide language-conditioned rewards
- 3000+ diverse tasks spanning combat, farming, mining, building, and survival
- A Gym-style API for agent development

**Observation/Action Space:**
- **Observation:** RGB (160x256), voxel observations, compass, inventory, biome info, GPS (privileged), and more. Highly configurable multimodal observations.
- **Action:** Compound action space with movement (forward/back/strafe), camera (pitch/yaw), attack, use, craft, equip, place, destroy, and more. More flexible than MineRL's action space.
- **Tasks:** 3000+ programmatic tasks defined by goal conditions, organized into categories

**What Worked:**
- MineCLIP provided an effective reward signal for language-conditioned RL, enabling training of multi-task agents without dense reward engineering [^24^](https://arxiv.org/html/2412.18293v3)
- The diversity of 3000+ tasks enabled systematic evaluation of agent generalization
- The internet-knowledge integration (Wiki, YouTube, Reddit) was a novel and influential contribution
- Served as the foundation for many subsequent projects (Plan4MC, DEPS, and others)

**What Failed:**
- The simulator is relatively slow compared to pure-Java alternatives like Mineflayer, hampering RL training throughput [^24^](https://arxiv.org/html/2412.18293v3)
- MineCLIP rewards, while useful, are noisy and imprecise — agents often exploit reward hacking
- The environment setup is complex (Java, Gradle, Python dependencies) and frequently breaks
- Many tasks are programmatically defined and don't capture the richness of open-ended human play

**Code Availability:**
- **GitHub:** https://github.com/MineDojo/MineDojo
- **License:** MIT (code), CC BY 4.0 (YouTube data), various for Wiki/Reddit
- **Status:** Limited active development. Issues pile up without resolution. The CraftJarvis team maintains a fork (MC-Simulator) that is more actively maintained. [^170^](https://github.com/CraftJarvis/MC-Simulator)
- **Last meaningful commit:** 2023

**Applicability to Multi-Agent:**
Medium. MineDojo supports multiple agents in principle but the multi-agent API is not well-documented or battle-tested. The task definitions are single-agent focused. For multi-agent village building, a fork or wrapper would be needed.

---

### 3. Voyager (2023) — Wang et al., GPT-4 Powered Lifelong Learning Agent

**Paper:** "Voyager: An Open-Ended Embodied Agent with Large Language Models" (NeurIPS 2023 Workshop) [^27^](https://arxiv.org/abs/2305.16291)

**Architecture Summary:**
Voyager is the landmark LLM-powered lifelong learning agent for Minecraft. It has three core components:
1. **Automatic Curriculum:** Uses GPT-4 to generate progressively harder tasks, maximizing exploration based on the agent's current state (inventory, location, completed/failed tasks)
2. **Skill Library:** An ever-growing vector database of executable JavaScript code (Mineflayer API calls), indexed by GPT-3.5-generated descriptions. Each skill is a reusable function like `craftStoneSword()` or `combatZombieWithSword()`.
3. **Iterative Prompting Mechanism:** A feedback loop where GPT-4 generates code, the code executes in Minecraft, environment feedback and execution errors are collected, and GPT-4 refines the code. A self-verification module confirms task completion before committing the skill to the library.

**Observation/Action Space:**
- **Observation:** Inventory state, agent stats (health, hunger), nearby entities, block types, biome info, chat messages. Uses Mineflayer's rich API for state extraction rather than raw pixels.
- **Action:** JavaScript code that calls Mineflayer APIs (e.g., `bot.dig()`, `bot.placeBlock()`, `bot.craft()`). This **code-as-policy** approach represents temporally extended actions as programs rather than low-level motor commands.
- **Key insight:** Code is compositional and interpretable — complex behaviors are built by composing simpler code blocks.

**What Worked:**
- Achieved **3.3x more unique items**, **2.3x longer distances traveled**, and **up to 15.3x faster tech tree milestone unlocking** compared to prior SOTA [^27^](https://arxiv.org/abs/2305.16291)
- The skill library effectively **eliminated catastrophic forgetting** — skills learned in one world transferred to new worlds
- Code-as-policy proved vastly more effective than low-level motor commands for long-horizon tasks
- The iterative prompting mechanism (incorporating environment feedback, execution errors, and self-verification) dramatically improved code correctness
- Only required blackbox GPT-4 queries — no fine-tuning or model parameter access needed

**What Failed:**
- **Critical dependency on GPT-4 API** — the system does not work effectively with weaker models. GPT-3.5 ablations showed dramatic performance drops [^31^](https://arxiv.org/pdf/2305.16291)
- **No visual perception** — Voyager relies entirely on symbolic state from Mineflayer, making it blind to visual details that matter for some tasks
- **Single-agent only** — no mechanism for multi-agent coordination or communication
- **Requires significant API costs** for GPT-4 queries during extended play sessions
- The skill library can grow unwieldy over time; without pruning, retrieval quality degrades

**Code Availability:**
- **GitHub:** https://github.com/MineDojo/Voyager
- **License:** MIT
- **Status:** The original repo has seen limited recent activity. However, multiple forks and extensions exist. The Co-voyager fork has 18 commits ahead of main. [^171^](https://github.com/Itakello/Co-voyager) The CraftJarvis team also maintains a Voyager-based codebase (Odyssey builds on Voyager).
- **Last meaningful activity:** 2023, with community forks in 2024-2025

**Applicability to Multi-Agent:**
**HIGH.** Voyager's architecture is the most adaptable to multi-agent among all surveyed projects:
- The skill library can be **shared across agents** — each agent contributes skills and retrieves from a communal library
- Code-as-policy enables **composable multi-agent behaviors** — agents can call each other's skills
- The automatic curriculum can be extended to generate **multi-agent tasks**
- The iterative prompting mechanism can incorporate **inter-agent communication**
- **Recommendation:** Clone Voyager as a starting point for multi-agent village building. Its modular design (curriculum, skill library, prompting) maps cleanly to multi-agent extensions.

---

### 4. GITM (2023) — Zhu et al., Ghost in the Minecraft

**Paper:** "Ghost in the Minecraft: Generally Capable Agents for Open-World Environments via Large Language Models with Text-based Knowledge and Memory" [^32^](https://arxiv.org/abs/2305.17144)

**Architecture Summary:**
GITM takes a hierarchical LLM-based approach with three components:
1. **LLM Decomposer:** Breaks down high-level goals (e.g., "ObtainDiamond") into sub-goals using text-based knowledge from the internet (Minecraft Wiki)
2. **LLM Planner:** Plans sequences of **structured actions** for each sub-goal. Structured actions are high-level cognitive primitives: `{equip, explore, approach, mine/attack, dig_down, go_up, build, craft/smelt, apply}`. Each has clear semantics and feedback.
3. **LLM Interface:** Translates structured actions into actual keyboard/mouse operations and processes raw observations.

**Observation/Action Space:**
- **Observation:** LiDAR rays (5-degree intervals, horizontal and vertical), voxel observations (10-unit radius), inventory, life status, and agent location. **No RGB images used** — the agent operates primarily on symbolic spatial data.
- **Action:** 9 structured actions (equip, explore, approach, mine/attack, dig_down, go_up, build, craft/smelt, apply) that are translated to keyboard/mouse inputs. The explore action uses BFS (on ground) or DFS (underground) traversal strategies.
- **Note:** Breaking speed and strength were set to 100, effectively giving the agent superhuman mining capabilities.

**What Worked:**
- Achieved **+47.5% improvement** in ObtainDiamond success rate over all previous methods, reaching the highest success rate at the time [^32^](https://arxiv.org/abs/2305.17144)
- First agent to **procure ALL items in the Minecraft Overworld technology tree**
- Operates on a **single CPU node with 32 cores** — no GPU needed, making it dramatically more accessible than RL-based approaches
- The structured action abstraction proved highly effective — LLMs reason better at the cognitive level than at the motor-control level

**What Failed:**
- The LiDAR-only observation (no RGB) means the agent misses visual information (biome types, dropping items, visual details of blocks). The paper acknowledges this limitation.
- Setting breaking speed/strength to 100 gives the agent superhuman abilities not available to human players
- The agent relies heavily on Minecraft Wiki knowledge — when the LLM lacks domain knowledge, planning fails
- Single-agent only with no learning mechanism — doesn't improve from experience
- Hardcoded explore strategies (BFS/DFS) are inefficient for large-scale terrain

**Code Availability:**
- **GitHub:** https://github.com/OpenGVLab/GITM
- **License:** Not clearly specified
- **Status:** The repo received initial code in 2023 but has limited ongoing maintenance. Multiple issues remain unresolved.
- **Last meaningful activity:** 2023

**Applicability to Multi-Agent:**
Low-Medium. GITM's hierarchical decomposition could be extended to multi-agent task allocation, but the architecture is fundamentally single-agent. The structured action space doesn't include communication primitives. The heavy reliance on internet knowledge and superhuman physical abilities makes it less suitable for realistic multi-agent scenarios.

---

### 5. DEPS (2023) — Wang et al., Describe-Explain-Plan-Select

**Paper:** "Describe, Explain, Plan and Select: Interactive Planning with Large Language Models Enables Open-World Multi-Task Agents" (NeurIPS 2023) [^59^](https://arxiv.org/abs/2302.01560)

**Architecture Summary:**
DEPS is an interactive planning framework with four key components:
1. **Descriptor:** Summarizes the current situation as text when failures occur
2. **Explainer:** Uses LLM (GPT-4/Codex) to self-explain why the previous plan failed
3. **Planner:** Regenerates the plan incorporating error information
4. **Selector:** A **learnable module** (not just LLM prompting) that ranks parallel candidate sub-goals based on estimated completion steps, refining the plan for feasibility

The agent combines this planner with goal-conditioned low-level controllers.

**Observation/Action Space:**
- **Observation:** Visual observations (RGB) processed through vision encoders, plus inventory and status info
- **Action:** High-level sub-goals (in natural language) are generated by the LLM planner and then executed by goal-conditioned controllers (trained with RL). The controllers map goals to low-level actions.
- **Key innovation:** The Selector module adds learned feasibility ranking to LLM planning

**What Worked:**
- First zero-shot multi-task agent to accomplish **70+ Minecraft tasks**, nearly doubling overall performance of baselines [^59^](https://arxiv.org/abs/2302.01560)
- The interactive error correction (describe + explain + replan) proved highly effective for long-horizon planning
- The learned Selector module significantly improved plan feasibility by considering agent state
- Generalized to non-Minecraft domains (ALFWorld, tabletop manipulation)
- GPT-4 planner with DEPS augmentation achieved ~90% success on MT1 tasks vs ~50% for vanilla GPT-4 [^58^](https://arxiv.org/html/2302.01560v3)

**What Failed:**
- The planning latency is high — each replan requires multiple LLM API calls, making real-time interaction slow
- Goal-conditioned controllers still fail on some low-level tasks; the system degrades when controllers fail repeatedly
- The Codex API dependency was cancelled by OpenAI, requiring migration to newer models [^166^](https://github.com/CraftJarvis/MC-Planner)
- Single-agent only

**Code Availability:**
- **GitHub Planner:** https://github.com/CraftJarvis/MC-Planner
- **GitHub Controller:** https://github.com/CraftJarvis/MC-Controller
- **GitHub Simulator (MineDojo fork):** https://github.com/CraftJarvis/MC-Simulator
- **License:** MIT
- **Status:** Part of the CraftJarvis ecosystem. The MC-Planner repo received its last meaningful update in March 2023 with a note about Codex API cancellation. The CraftJarvis team has since moved to newer projects (JARVIS-1, Optimus series).

**Applicability to Multi-Agent:**
Medium. The interactive planning framework (describe-explain-plan) could be extended to multi-agent coordination. The Selector's learned feasibility ranking could help allocate tasks among agents. However, the architecture requires significant refactoring for multi-agent.

---

### 6. Plan4MC (2023) — Yuan et al., Skill Graph with LLM Pre-Generation

**Paper:** "Plan4MC: Skill Reinforcement Learning and Planning for Open-World Minecraft Tasks" (NeurIPS 2023 FMDM Workshop) [^99^](https://arxiv.org/abs/2303.16563)

**Architecture Summary:**
Plan4MC is a **demonstration-free RL agent** that combines skill learning with LLM-assisted planning:
1. **Skill Learning:** Three types of fine-grained basic skills trained with RL:
   - Finding-skills: Exploration policies that locate items
   - Manipulation-skills: Policies for interacting with items/mobs (harvesting, attacking, mining)
   - Crafting-skills: Hardcoded crafting recipes
2. **Skill Graph:** An LLM (GPT) pre-generates a directed graph of skill dependencies before task execution
3. **Skill Search:** A DFS-based search algorithm walks the skill graph to generate executable skill plans

**Observation/Action Space:**
- **Observation:** RGB images (160x256) processed through MineCLIP encoder, plus inventory and status
- **Action:** Compressed discrete action space (12x3 actions) including movement, camera, attack, and use
- **Policy architecture:** MineAgent-style (MineCLIP image encoder + MLPs) with LSTM for Finding-skills

**What Worked:**
- **Most sample-efficient demonstration-free RL method** — solved 40 diverse tasks with only 7M environment steps [^99^](https://arxiv.org/abs/2303.16563)
- The Finding-skill innovation dramatically improved success rates — providing better initialization for Manipulation-skills (0.40 conditional success with Finding vs 0.25 without)
- The skill graph + search approach proved more reliable than direct LLM planning for long-horizon tasks [^72^](https://arxiv.org/html/2303.16563v2)
- Outperformed Interactive LLM baseline on Mine-Stones and Mine-Ores tasks requiring long-horizon planning

**What Failed:**
- Finding-skills are not goal-aware during exploration — the exploration policy is suboptimal for specific targets
- The skill graph depends on LLM domain knowledge — if the LLM lacks knowledge, the graph is incorrect
- Only 40 tasks were evaluated — far fewer than LLM-based agents
- Performance degrades on tasks requiring more than 10 sequential skills

**Code Availability:**
- **GitHub:** https://github.com/PKU-RL/Plan4MC
- **License:** MIT (assumed)
- **Status:** Released in 2023. Pre-trained models for skills included. Limited ongoing maintenance.
- **Last meaningful activity:** 2023

**Applicability to Multi-Agent:**
Medium-High. The skill graph concept is highly applicable to multi-agent coordination:
- Multiple agents can each specialize in different skills (division of labor)
- The skill graph can be extended to include communication and coordination primitives
- Skill search can allocate sub-tasks to different agents
- The RL-trained skills provide reliable low-level execution without LLM API costs

---

### 7. JARVIS-1 (2023) — Wang et al., Memory-Augmented Multimodal LLM

**Paper:** "JARVIS-1: Open-World Multi-Task Agents with Memory-Augmented Multimodal Language Models" (T-PAMI 2024) [^63^](https://arxiv.org/abs/2311.05997)

**Architecture Summary:**
JARVIS-1 is a multimodal agent built on pre-trained multimodal language models:
1. **Multimodal Perception:** Visual observations (RGB) + textual instructions fed into a multimodal language model (MLLM)
2. **Planning:** The MLLM generates language plans from visual + text input
3. **Goal-Conditioned Controllers:** Plans are dispatched to STEVE-1/VPT-based controllers for low-level execution
4. **Multimodal Memory:** A hybrid memory system combining:
   - Pre-trained knowledge (Minecraft facts)
   - Actual gameplay experiences (observations, actions, outcomes)
   - Multimodal state-action sequences

**Observation/Action Space:**
- **Observation:** Raw RGB pixels (visual) + textual inventory/state descriptions
- **Action:** Goal-conditioned policies (via STEVE-1 controller) that output keyboard/mouse actions at 20Hz
- **Key feature:** Unlike text-only LLM agents, JARVIS-1 can **see** the environment through visual input

**What Worked:**
- Achieved **nearly perfect performance** on 200+ tasks from the Minecraft Universe Benchmark [^63^](https://arxiv.org/abs/2311.05997)
- **12.5% success rate on ObtainDiamondPickaxe** — 5x improvement over previous records [^62^](https://www.reddit.com/r/singularity/comments/17uk4dx/jarvis1_openworld_multitask_agents_with/)
- Demonstrated **self-improvement** in lifelong learning experiments — performance increased with more gameplay
- The multimodal memory proved critical for performance improvement over time

**What Failed:**
- The released code is **incomplete** — multimodal descriptor and retrieval were not fully released. The GitHub README states these are "Coming Soon" since 2023. [^95^](https://github.com/CraftJarvis/JARVIS-1)
- Heavy reliance on STEVE-1/VPT controllers which have their own limitations
- Requires significant computational resources (GPU for vision model + API costs for LLM)
- The full system is complex with many moving parts, making replication difficult

**Code Availability:**
- **GitHub:** https://github.com/CraftJarvis/JARVIS-1
- **License:** Not explicitly stated
- **Status:** Released but incomplete. The team has since moved to OmniJARVIS and the Optimus series. Issues remain largely unaddressed.
- **Last meaningful activity:** 2023 (with ongoing work from the same team on newer projects)

**Applicability to Multi-Agent:**
Medium. The multimodal memory concept could be extended to shared multi-agent memory. The MLLM planner could generate coordination plans. However, the architecture is fundamentally single-agent and the incomplete release makes extension difficult.

---

### 8. MP5 (2023) — Qin et al., Multi-Modal Open-Ended Embodied System

**Paper:** "MP5: A Multi-modal Open-ended Embodied System in Minecraft via Active Perception" (CVPR 2024) [^57](https://arxiv.org/abs/2312.07472)

**Architecture Summary:**
MP5 is a five-module embodied system built on top of MLLMs:
1. **Parser:** Decomposes long-horizon tasks into sub-objectives
2. **Percipient:** A LoRA-enabled Multimodal LLM that answers visual questions about the environment
3. **Planner:** Schedules action sequences and refines sub-objectives based on current situation
4. **Performer:** Executes actions with frequent environment interaction
5. **Patroller:** Verifies plans/actions and provides feedback on potential improvements

**Key innovation:** Active perception — multi-round interaction between Percipient and Patroller actively perceives contextual information in response to queries from Planner and Performer.

**Observation/Action Space:**
- **Observation:** Visual observations (RGB images) for the Percipient, plus inventory and status
- **Action:** High-level actions executed through Mineflayer-like APIs
- **Key feature:** The active perception loop allows the agent to ask clarifying visual questions before acting

**What Worked:**
- Achieved **22% success rate on difficult process-dependent tasks** and **91% on context-dependent tasks** [^57](https://arxiv.org/abs/2312.07472)
- The active perception scheme proved critical for context-dependent tasks where the agent needs to perceive different information at different stages
- Demonstrated strong performance on open-ended tasks combining process and context challenges

**What Failed:**
- **GPT-4/GPT-3.5 dependency** limits accessibility [^64](https://arxiv.org/html/2312.07472v1)
- Limited to the Minecraft simulation platform — no generalization to other environments
- The five-module design creates significant latency — each action requires multiple LLM calls
- MP5 without situation-aware planning (w/o P.) achieves **0% success** on process-dependent tasks, showing the system is brittle when planning fails

**Code Availability:**
- **GitHub:** Code linked in the paper's awesome-LLM-game-agent-papers listing but the direct repo is not widely known
- **Status:** Code availability is limited. The paper mentions "code will be made available" but finding the actual repository is difficult.
- **Assessment:** Effectively paper-only for most researchers.

**Applicability to Multi-Agent:**
Medium. The modular design (Parser, Percipient, Planner, Performer, Patroller) could be distributed across agents. The active perception scheme could be extended to inter-agent observation. However, no multi-agent support exists in the current architecture.

---

### 9. STEVE-1 (2023) — Lifshitz et al., Text-to-Target Mining

**Paper:** "STEVE-1: A Generative Model for Text-to-Behavior in Minecraft" (NeurIPS 2023) [^92](https://github.com/Shalev-Lifshitz/STEVE-1)

**Architecture Summary:**
STEVE-1 is an instruction-tuned generative model for following text instructions in Minecraft:
1. Uses the **unCLIP** methodology from DALL-E 2 — decoupling instruction understanding from behavior generation
2. Two-stage training:
   - Stage 1: Adapts pretrained VPT model to follow MineCLIP latent space commands (self-supervised behavioral cloning + hindsight relabeling)
   - Stage 2: Trains a prior to predict MineCLIP latent codes from text instructions
3. Leverages pretrained VPT foundation model + MineCLIP for text-conditioned behavior

**Observation/Action Space:**
- **Observation:** Raw RGB pixels (64x64 in MineRL, adapted for MineDojo)
- **Action:** Low-level keyboard and mouse controls at 20Hz (native human interface)
- **Interface:** Text and visual goal instructions → MineCLIP embedding → VPT policy → keyboard/mouse actions

**What Worked:**
- First model to achieve robust text-to-behavior in Minecraft with **low-level controls and raw pixel inputs**
- Completed **12 of 13 tasks** in the early-game evaluation suite
- Cost only **$60 of compute** to train, leveraging pretrained models
- Demonstrated the power of combining pretrained models (VPT + MineCLIP) with text-conditioned generation techniques
- Successfully generalizes to visual goal specifications (not just text)

**What Failed:**
- **Strong behavioral biases from VPT pretraining** — VPT has strong priors for certain behaviors (digging, killing spiders) that dominate instruction following. Other papers note STEVE-1 "shows poor performance across all hunting tasks except hunt a spider" [^91](https://arxiv.org/html/2408.01942v1)
- **Prompt engineering is critical** — performance varies dramatically with prompt template. "Kill a {animal}" works while "hunt a {animal}" fails completely [^91](https://arxiv.org/html/2408.01942v1)
- Limited to short-horizon tasks — cannot handle multi-step crafting chains like ObtainDiamond
- Low-level motor control makes long-horizon composition extremely difficult

**Code Availability:**
- **GitHub:** https://github.com/Shalev-Lifshitz/STEVE-1
- **License:** MIT (assumed)
- **Status:** Well-documented repo with model weights, training scripts, and evaluation tools. However, limited recent activity.
- **Last meaningful activity:** 2023

**Applicability to Multi-Agent:**
Low-Medium. STEVE-1 is fundamentally a single-agent low-level controller. However, it serves as the goal-conditioned controller component in JARVIS-1, suggesting it can be integrated into multi-agent systems as a "motor cortex" while higher-level planners handle coordination.

---

### 10. VPT (2022) — Baker et al., Video PreTraining

**Paper:** "Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos" (NeurIPS 2022) [^96](https://arxiv.org/abs/2206.11795)

**Architecture Summary:**
VPT extends the internet-scale pretraining paradigm to sequential decision domains:
1. **Inverse Dynamics Model (IDM):** Trained on ~2,000 hours of labeled contractor gameplay to predict keyboard/mouse inputs from before/after frames
2. **Dataset Labeling:** The IDM labels ~70,000 hours of unlabeled internet Minecraft videos with pseudo-actions
3. **Foundation Model:** A 0.5B parameter behavioral cloning model trained on the pseudo-labeled dataset
4. **Fine-tuning:** The foundation model is fine-tuned with both imitation learning and reinforcement learning

**Observation/Action Space:**
- **Observation:** Raw RGB pixels (64x64)
- **Action:** Native human interface — mouse and keyboard at 20Hz, including drag-and-drop inventory management. This is the **full, unmodified human action space**, far more complex than simplified agent actions.
- **Policy:** Transformer architecture processing frame history

**What Worked:**
- **First AI to craft a diamond pickaxe** in Minecraft without action-space simplifications [^126](https://the-decoder.com/openai-ai-builds-diamond-axe-in-minecraft-why-it-matters/)
- Foundation model learned impressive zero-shot behaviors: tree chopping, crafting, swimming, hunting, village exploration, pillar jumping
- The semi-supervised approach (small labeled data unlocking massive unlabeled data) proved highly scalable
- Fine-tuned models achieved human-level performance on many tasks

**What Failed:**
- The foundation model's zero-shot performance is far below human proficiency (e.g., 0.19 crafting tables per episode vs 5.44 for human contractors) [^95](https://ar5iv.labs.arxiv.org/html/2206.11795)
- RL fine-tuning requires significant compute (720 GPUs for foundation model training)
- The IDM introduces labeling errors that propagate to the foundation model
- Training at 30 fps with mouse/keyboard actions is extremely compute-intensive

**Code Availability:**
- Code was released by OpenAI as part of the VPT publication
- **GitHub:** Not a single standalone repo; distributed across OpenAI's repositories
- **License:** MIT
- **Status:** The VPT models and training code were released but are now largely superseded by later work. OpenAI has not maintained the codebase.
- **Assessment:** Historical significance high, but not a practical starting point for new projects.

**Applicability to Multi-Agent:**
Low. VPT is a single-agent behavioral prior. While the pretrained model could theoretically serve as initialization for multi-agent policies, the architecture and training setup are purely single-agent.

---

### 11. OpenAI's Minecraft Project (2022) — The VPT/Diamond Story

**Context:**
This refers to OpenAI's broader Minecraft research program that culminated in the VPT paper and the associated blog posts. The work was done by the OpenAI team (Baker, Akkaya, Zhokov, Huizinga, Tang, Ecoffet, Houghton, Sampedro, Clune).

**Key Achievements:**
- Collected ~270,000 hours of Minecraft YouTube video, filtered to ~70,000 hours of clean gameplay [^126](https://the-decoder.com/openai-ai-builds-diamond-axe-in-minecraft-why-it-matters/)
- Trained the IDM on ~2,000 hours of contractor gameplay at a cost of ~$2,000
- Used the IDM to pseudo-label the 70k hours of video, unlocking massive training data
- Foundation model (0.5B parameters) trained on 720 V100 GPUs over ~9 days
- Fine-tuned with RL to achieve diamond pickaxe crafting — a task requiring 24,000+ consecutive actions [^127](https://singularityhub.com/2022/06/26/openais-new-ai-learned-to-play-minecraft-by-watching-70000-hours-of-youtube/)

**What This Work Established:**
- Proof that internet-scale video pretraining works for sequential decision domains
- The IDM approach for labeling unlabeled video is a key technique still used today
- Foundation models for behavior (behavioral priors) are feasible and effective
- The "VPT recipe" (IDM → pseudo-label → foundation model → fine-tune) has been adopted across domains

**Limitations:**
- Massive compute requirements (720 GPUs) put this out of reach for most researchers
- The diamond pickaxe success rate was low (~2.5% for foundation model, higher after RL fine-tuning)
- Zero-shot capabilities are primitive compared to LLM-based agents from 2023
- Not multi-agent capable

**Code Availability:**
- Released by OpenAI but not as a maintained package
- **Blog post:** https://openai.com/blog/vpt/
- **Paper:** https://arxiv.org/abs/2206.11795

**Applicability to Multi-Agent:**
Low directly. However, the VPT model could serve as a behavioral prior for multi-agent agents — each agent could be initialized with VPT weights before multi-agent training.

---

### 12. DeepMind XLand (2021-2023) — Open-Ended Multi-Agent Learning

**Papers:**
- "Open-Ended Learning Leads to Generally Capable Agents" (2021) [^165](https://deepmind.google/blog/generally-capable-agents-emerge-from-open-ended-play/)
- "Human-Timescale Adaptation in an Open-Ended Task Space" (2023) [^162](https://ar5iv.labs.arxiv.org/html/2301.07608)

**Architecture Summary:**
XLand is not Minecraft but a DeepMind-developed 3D multiplayer environment for training generally capable agents:
- **Procedural task generation:** Billions of tasks are generated programmatically across varied games, worlds, and player configurations
- **Multiplayer focus:** Agents train with and against other agents (co-players), creating a dynamic learning environment
- **3D first-person:** Agents observe RGB images and receive text goal descriptions
- **Auto-curriculum:** The training system dynamically adjusts task difficulty based on agent performance (PLR method)
- **Meta-learning:** Agents trained to adapt within a single episode (few-shot adaptation)

**Observation/Action Space:**
- **Observation:** RGB images + text goal description
- **Action:** 3D movement and interaction (varies by game)
- **Multi-agent:** Supports 1-4 player games with varied cooperation/competition structures

**What Worked:**
- Produced agents with **general heuristic behaviors** (experimentation, tool use) that transfer to completely unseen tasks
- Agents exhibited **human-timescale adaptation** — improving within a single episode through in-context learning [^162](https://ar5iv.labs.arxiv.org/html/2301.07608)
- The open-ended learning process (agents never stop learning) proved scalable and effective
- Multi-agent training produced sophisticated social behaviors (cooperation, competition, theory of mind)

**What Failed:**
- XLand is a custom environment, not Minecraft — direct transfer to Minecraft is non-trivial
- Massive compute requirements (TPU pods, weeks of training)
- The environment is less rich than Minecraft (no crafting tech tree, no block manipulation)
- Results are impressive but the environment's simplicity compared to Minecraft limits real-world applicability

**Code Availability:**
- **XLand-MiniGrid:** A simplified open-source reimplementation exists (https://github.com/dyllanwli/XLand-MiniGrid)
- **Original XLand:** Not publicly released by DeepMind
- **Status:** Open-source ecosystem is limited. XLand-MiniGrid is a research tool, not a production environment.

**Applicability to Multi-Agent:**
**VERY HIGH (conceptually).** XLand is the only surveyed project that was fundamentally designed for multi-agent scenarios. Key lessons for Minecraft village-building:
- **Open-ended auto-curriculum** is essential for multi-agent training — tasks should dynamically adapt
- **Fictitious self-play** (sampling co-players from historical agent versions) creates a natural learning progression
- **Multi-agent training requires both cooperative and competitive tasks** for robust behavior
- **Meta-learning for adaptation** within episodes is critical when agents must coordinate with unknown partners
- **Recommendation:** Study XLand's training methodology even though the environment is different. The PLR auto-curriculum and self-play mechanisms are directly applicable to multi-agent Minecraft.

---

### 13. Optimus-3 (2025) — Latest MoE Generalist Agent

**Paper:** "Optimus-3: Dual-Router Aligned Mixture-of-Experts Agent with Dual-Granularity Reasoning-Aware Policy Optimization" [^42](https://arxiv.org/html/2506.10357v2)

**Architecture Summary:**
Optimus-3 is the state-of-the-art generalist Minecraft agent, built on Qwen2.5-VL with a novel MoE architecture:
1. **Dual-Router Aligned MoE:**
   - **Task Router:** Assigns inputs to task-specific experts (planning, action, captioning, embodied QA, grounding, reflection)
   - **Layer Router:** Accelerates action inference by selectively skipping intermediate layers
   - **Shared Knowledge Expert:** Maintains common knowledge across all tasks
2. **Dual-Granularity Reasoning-Aware Policy Optimization (DGRPO):**
   - Fine-grained reward functions per task type
   - Dependency-Aware Synthesis Reward for planning (crafting dependency path as thinking reward)
   - Hallucination-Aware Consistency Reward for vision tasks

**Observation/Action Space:**
- **Observation:** Visual (RGB) + textual inventory/instruction input to the MLLM
- **Action:** Unified output tokens that can represent plans, actions, captions, answers, or reflections
- **Key feature:** End-to-end MLLM that handles all capabilities within one model, activated by task routing

**What Worked:**
- **Highest success rate across all 7 task groups** on long-horizon tasks, with 15% SR on Diamond Group [^5](https://arxiv.org/html/2506.10357v1)
- Achieves **35% success and 69% completion rate** on the Diamond Sword open-ended task — substantially above all baselines including JARVIS-1 and GPT-4o [^42](https://arxiv.org/html/2506.10357v2)
- MoE architecture effectively eliminates **task interference** seen in dense Qwen2.5-VL variants
- Task-level routing outperforms token-level routing on captioning, planning, and grounding
- Entire data collection pipeline cost only **$300 in API costs** with 4x L40 GPUs over 36 hours
- Self-contained: performs planning and action prediction without external tools or models

**What Failed:**
- Requires GPU with **at least 32GB VRAM** for server deployment — not accessible to all researchers [^125](https://github.com/JiuTian-VL/Optimus-3)
- The system's complexity (dual routers, MoE, DGRPO) makes it difficult to modify or extend
- Still fundamentally single-agent — multi-agent coordination is not addressed
- Relatively new — has not been as extensively validated as Voyager or JARVIS-1

**Code Availability:**
- **GitHub:** https://github.com/JiuTian-VL/Optimus-3
- **HuggingFace Models:** https://huggingface.co/MinecraftOptimus/Optimus-3
- **License:** Not explicitly stated (assumed MIT)
- **Status:** **Actively maintained.** The team (Harbin Institute of Technology + Peng Cheng Laboratory) regularly releases updates:
  - June 2025: Initial release
  - March 2026: Optimus-3-v2 and MineSys2 Benchmark
  - March 2026: OptimusM4 Dataset on Huggingface
- **GUI client available:** Yes, with OptimusGUI for interactive play

**Applicability to Multi-Agent:**
**HIGH.** Optimus-3 is the most technically advanced single-agent system and has several multi-agent-relevant features:
- The task router naturally extends to **multi-agent task allocation** — each agent's router could include coordination tasks
- The end-to-end MLLM design means **communication can be another output modality**
- The active development team and comprehensive codebase make it a better foundation than abandoned projects
- **However:** The MoE architecture is complex and modifying it for multi-agent requires deep expertise

---

### 14. Odyssey (2024) — Open-World Skills Benchmark

**Paper:** "Odyssey: Empowering Minecraft Agents with Open-World Skills" (ICLR 2025) [^124](https://arxiv.org/abs/2407.15325)

**Architecture Summary:**
Odyssey is a framework and benchmark focused on **diverse open-world skills** beyond the tech tree:
1. **Open-World Skill Library:** 40 primitive skills + 183 compositional skills covering combat, farming, cooking, animal husbandry, and more
2. **Fine-tuned LLaMA-3:** Trained on a 390k+ instruction-entry QA dataset derived from Minecraft Wiki
3. **Agent Capability Benchmark:** Three task types:
   - Long-term planning tasks (combat, crafting chains)
   - Dynamic-immediate planning tasks (reacting to environment changes)
   - Autonomous exploration tasks

**Observation/Action Space:**
- **Observation:** Uses Mineflayer-based state extraction (similar to Voyager)
- **Action:** JavaScript code calling Mineflayer APIs (Voyager-style code-as-policy)
- **Key difference:** Much richer skill library than Voyager, covering diverse gameplay beyond crafting

**What Worked:**
- Addresses the limitation of prior agents being overly focused on ObtainDiamond
- The 40 primitive + 183 compositional skills cover **combat, farming, cooking, animal husbandry, building, and exploration**
- The LLaMA-3 fine-tuning on Minecraft Wiki data provides strong domain knowledge without GPT-4 API costs
- Benchmark effectively evaluates different agent capabilities separately
- All datasets, model weights, and code are publicly available [^128](https://github.com/zju-vipa/odyssey)

**What Failed:**
- Performance on long-horizon tasks still lags behind GPT-4-based agents like Optimus-3
- The skill library, while extensive, may not cover all village-building scenarios
- Evaluation is primarily benchmark-based — less emphasis on open-ended exploration metrics

**Code Availability:**
- **GitHub:** https://github.com/zju-vipa/odyssey
- **License:** MIT (assumed, based on Voyager heritage)
- **Status:** Released in 2024. Based on Voyager framework. Moderate community activity.
- **Last meaningful activity:** 2024-2025

**Applicability to Multi-Agent:**
**HIGH.** Odyssey is the most directly relevant benchmark for village-building:
- The diverse skill library (especially farming, building, animal husbandry) maps directly to village-building tasks
- The task types (long-term planning, dynamic-immediate, exploration) correspond to village-building requirements
- The Voyager-based architecture is easily extensible to multi-agent
- The benchmark includes combat scenarios which are relevant for village defense

---

## Project Rankings for Multi-Agent Village Building

### Rank 1: **Voyager** — Best starting point
- **Why:** Modular architecture, MIT license, large community, skill library concept directly extensible to multi-agent, code-as-policy enables composable behaviors
- **Clone and extend:** The automatic curriculum → shared multi-agent curriculum, skill library → communal skill repository
- **Effort required:** Medium — need to add inter-agent communication protocol and task allocation

### Rank 2: **Optimus-3** — Best SOTA foundation
- **Why:** State-of-the-art performance, actively maintained, end-to-end MLLM, MoE architecture naturally handles multi-task allocation
- **Clone and extend:** Add communication output modality to the task router, extend DGRPO with multi-agent rewards
- **Effort required:** High — complex architecture, needs significant GPU resources

### Rank 3: **Odyssey** — Best skill coverage
- **Why:** Most diverse skill library (40 primitives + 183 compositional), covers farming/building/animal husbandry directly relevant to villages, benchmark framework for evaluation
- **Clone and extend:** Leverage Voyager-based architecture with richer skills, add multi-agent coordination
- **Effort required:** Medium — builds on Voyager, well-documented

### Rank 4: **Plan4MC** — Best for RL-based skill learning
- **Why:** Demonstration-free RL skill learning, skill graph for planning, Finding-skills are useful for exploration
- **Clone and extend:** Train specialized skills for village tasks, extend skill graph to include coordination primitives
- **Effort required:** High — RL training is compute-intensive

### Rank 5: **MineDojo** — Best environment framework
- **Why:** 3000+ tasks, flexible observation/action space, internet-scale knowledge integration
- **Use as:** Environment backbone for training and evaluation
- **Effort required:** Low for environment setup, but multi-agent requires modifications

### Rank 6: **XLand** (lessons only) — Best multi-agent training methodology
- **Why:** Auto-curriculum, fictitious self-play, meta-learning for adaptation
- **Apply lessons:** PLR auto-curriculum, co-player pool dynamics, task procedural generation
- **Effort required:** Conceptual — implement XLand's training mechanisms in Minecraft

### Rank 7: **JARVIS-1** — Best multimodal memory
- **Why:** Multimodal perception, memory-augmented planning, self-improvement
- **Caveat:** Incomplete code release makes extension difficult
- **Effort required:** High due to incomplete codebase

### Rank 8: **DEPS** — Best interactive planning
- **Why:** Describe-explain-plan feedback loop, learned Selector module
- **Use for:** Error recovery mechanism in multi-agent systems
- **Effort required:** Medium

---

## Open-Source vs Paper-Only Status

| Project | Code Available | License | Actively Maintained | Recommended to Clone |
|---------|--------------|---------|-------------------|---------------------|
| **Voyager** | Yes | MIT | Limited (community forks) | **YES — #1 Priority** |
| **Optimus-3** | Yes | Assumed MIT | **YES (2025-2026)** | **YES — #2 Priority** |
| **Odyssey** | Yes | MIT | Moderate | **YES — #3 Priority** |
| **Plan4MC** | Yes | Assumed MIT | No (2023) | Consider |
| **JARVIS-1** | Yes (incomplete) | Unspecified | No (moved to newer projects) | With caution |
| **DEPS/MC-Planner** | Yes | MIT | No (superseded) | For reference |
| **GITM** | Yes | Unspecified | No | For reference |
| **STEVE-1** | Yes | Assumed MIT | No | For reference |
| **MP5** | Limited | Unknown | No | Paper-only for most |
| **MineRL** | Yes | MIT | No (deprecated) | Historical only |
| **MineDojo** | Yes | MIT | Limited (use MC-Simulator fork) | For environment |
| **VPT/OpenAI** | Partially | MIT | No (historical) | Historical only |
| **XLand** | Partially (MiniGrid) | Unknown | No | Conceptual lessons only |

---

## Concrete Recommendations

### For Multi-Agent Village Building

1. **Start with Voyager's codebase** as the foundation. Its modular design (curriculum, skill library, iterative prompting) is the cleanest architecture for multi-agent extension. Add a communication protocol layer between agents and extend the skill library with village-specific skills. [^25^](https://github.com/MineDojo/Voyager)

2. **Integrate Optimus-3's MoE architecture** for the perception-planning-action backbone. The task-level routing in Optimus-3 is directly applicable to multi-agent task allocation. Study their DGRPO training methodology. [^125^](https://github.com/JiuTian-VL/Optimus-3)

3. **Adopt Odyssey's skill library** for diverse village tasks — farming, building, animal husbandry, cooking, and combat skills are all directly applicable. [^128^](https://github.com/zju-vipa/odyssey)

4. **Study XLand's training methodology** for multi-agent dynamics — specifically the PLR auto-curriculum and fictitious self-play mechanisms. These are environment-agnostic and directly applicable to Minecraft multi-agent training. [^165^](https://deepmind.google/blog/generally-capable-agents-emerge-from-open-ended-play/)

5. **Use MineDojo/MC-Simulator** as the environment backbone. The 3000+ tasks provide a starting point for defining village-related programmatic tasks. [^170^](https://github.com/CraftJarvis/MC-Simulator)

### Projects to Clone and Study (In Priority Order)

| Priority | Project | What to Study |
|----------|---------|---------------|
| 1 | **Voyager** | Skill library architecture, automatic curriculum, code-as-policy, iterative prompting |
| 2 | **Optimus-3** | MoE task routing, DGRPO training, end-to-end MLLM design |
| 3 | **Odyssey** | Open-world skill library (223 skills), LLaMA-3 fine-tuning on Minecraft knowledge |
| 4 | **Plan4MC** | RL skill training, skill graph generation, Finding-skill design |
| 5 | **XLand papers** | Auto-curriculum (PLR), fictitious self-play, meta-learning for adaptation |

### Technology Stack Recommendation

For a multi-agent village-building project, the recommended stack:

- **Environment:** MineDojo/MC-Simulator (CraftJarvis fork) for rich task definitions, or Mineflayer (for speed)
- **Perception:** Qwen2.5-VL or similar MLLM (following Optimus-3's approach)
- **Planning:** Code-as-policy (Voyager style) with LLM-generated JavaScript/Mineflayer code
- **Skills:** Extended Odyssey skill library with village-specific primitives
- **Memory:** Multimodal memory (JARVIS-1 style) with shared access across agents
- **Coordination:** Graph-based task allocation (inspired by Plan4MC skill graph + XLand self-play)
- **Training:** DGRPO-style RL for skill refinement (following Optimus-3)

---

## Open Questions

1. **Multi-agent coordination protocols:** How should LLM-based agents communicate and coordinate in Minecraft? Existing work (VillagerAgent, AgentVerse) is preliminary. What communication primitives (task delegation, resource sharing, conflict resolution) are needed?

2. **Scalability with agent count:** How do LLM-based agents scale to 5, 10, or 100 agents? Current systems are designed for single agents. The API costs and latency of N agents each calling GPT-4/Claude are prohibitive.

3. **Emergent social behaviors:** Can multi-agent Minecraft training produce emergent social structures (markets, hierarchies, specialization) similar to what XLand produced for simple games?

4. **Long-horizon village construction:** No existing agent can plan and execute multi-day construction projects requiring hundreds of sequential steps with parallel sub-tasks. This is an unsolved challenge.

5. **Resource allocation and economics:** Village building requires resource gathering, crafting, trading, and allocation decisions. How can LLM agents develop economic reasoning?

6. **Evaluation framework:** There is no standard benchmark for multi-agent village building. How should success be measured? (Built structures? Villager survival? Tech tree progress? Economic metrics?)

7. **Hybrid RL-LLM approaches:** Can the sample efficiency of Plan4MC's RL skills be combined with the planning flexibility of LLM agents? Current systems are mostly one or the other.

8. **Persistent worlds:** Most agents start fresh each episode. How do agents operate in persistent worlds where their constructions remain and other agents' actions matter?

9. **Model dependency:** Most LLM agents depend on GPT-4/Claude APIs. How can agents work with local models (Llama, Qwen) without massive performance degradation?

10. **Safety and alignment:** Multi-agent Minecraft with powerful LLMs raises safety concerns. How do we ensure agents don't develop harmful strategies or exploit game mechanics in unintended ways?

---

## References

All citations are inline hyperlinked to their sources throughout this document. Key repositories:

- Voyager: https://github.com/MineDojo/Voyager
- Optimus-3: https://github.com/JiuTian-VL/Optimus-3
- Odyssey: https://github.com/zju-vipa/odyssey
- Plan4MC: https://github.com/PKU-RL/Plan4MC
- JARVIS-1: https://github.com/CraftJarvis/JARVIS-1
- CraftJarvis (MC-Planner, MC-Controller, MC-Simulator): https://github.com/CraftJarvis
- GITM: https://github.com/OpenGVLab/GITM
- STEVE-1: https://github.com/Shalev-Lifshitz/STEVE-1
- MineDojo: https://github.com/MineDojo/MineDojo
- MineRL: https://github.com/minerllabs/minerl
