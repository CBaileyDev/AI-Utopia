## 1. Prior Art in Minecraft RL and LLM Agents

Building a multi-agent AI village in Minecraft requires understanding what has already been attempted, what succeeded, what failed, and — critically — which codebases can be cloned and studied. This chapter surveys fourteen major projects spanning 2019 to 2026, organized into three generations: Reinforcement Learning (RL) benchmarks and behavioral priors (2019–2022), Large Language Model (LLM) planning agents (2023), and the current state of the art (2024–2026). Each project is evaluated on architectural contribution, code availability, maintenance status, and relevance to multi-agent village construction. The chapter concludes with a ranked list of repositories to clone, justified by a scoring rubric that weights architectural relevance, license permissiveness, and active maintenance.

### 1.1 RL-Based Approaches (2019–2022)

The first wave of Minecraft AI research treated the game as a Reinforcement Learning (RL) benchmark. The core challenge was defined as sample-efficient learning in a high-dimensional, sparse-reward environment with a combinatorially large action space. Four projects defined this era.

#### 1.1.1 MineRL: Benchmark Framework and 60M+ Demonstrations

MineRL (Guss et al., IJCAI 2019) is a benchmark framework and dataset that catalyzed RL research in Minecraft [^23^]. Built on Microsoft's deprecated Project Malmo, it provides a Gym-compatible environment wrapper, 60 million state-action pairs from human demonstrations across nine hierarchical tasks, and a NeurIPS competition track for sample-efficient RL [^28^]. Observations are 64×64 RGB frames plus inventory state, compass angle, and crafting grid. The action space mimics human controls — discrete compound actions for movement, camera, attack/use, and inventory — with hundreds of valid combinations per timestep. Naive RL algorithms (Rainbow DQN) failed to outperform random policies on most tasks, and the ObtainDiamond objective remained unsolved, with best submissions achieving only modest success via imitation learning [^33^].

The MineRL codebase (MIT, `minerllabs/minerl`) entered maintenance mode after 2022; Project Malmo is deprecated. Its value is historical: it established the task vocabulary (Treechop, ObtainIronPickaxe, ObtainDiamond) and proved that pure RL in the human action space is intractable without strong priors. The 60M demonstrations could theoretically pre-train behavioral priors, but the engineering effort to rehabilitate the stack exceeds the value.

#### 1.1.2 VPT: Video PreTraining with Internet-Scale Behavioral Priors

Video PreTraining (VPT; Baker et al., NeurIPS 2022) learned behavioral priors from internet-scale video data through a three-stage pipeline: an Inverse Dynamics Model (IDM) trained on ~2,000 hours of labeled contractor gameplay predicts keyboard and mouse inputs from before/after frames; the IDM then labels ~70,000 hours of unlabeled YouTube Minecraft video with pseudo-actions; and a 0.5-billion-parameter Transformer is trained on this pseudo-labeled corpus via behavioral cloning, then fine-tuned with RL [^96^].

VPT was the first AI to craft a diamond pickaxe in Minecraft without action-space simplifications — over 24,000 consecutive correct actions [^126^]. The foundation model learned impressive zero-shot behaviors: tree chopping, crafting, swimming, hunting, village exploration [^127^]. However, zero-shot proficiency remained far below human contractors (0.19 crafting tables per episode versus 5.44), and RL fine-tuning required 720 V100 GPUs over ~nine days [^95^]. Code was released by OpenAI but is now superseded with no maintained repository.

VPT's lasting contributions are the IDM technique for labeling unlabeled video (still used in vision-language models today) and the proof that internet-scale behavioral priors are feasible for sequential decision domains. For a village project, pretrained VPT weights could theoretically initialize multi-agent policies, but the architecture is purely single-agent and the compute requirements place it out of reach.

#### 1.1.3 OpenAI Diamond Project: RL from Scratch

The broader OpenAI Minecraft research program that produced VPT represents the most compute-intensive approach to the game. The team collected approximately 270,000 hours of YouTube video, filtered to 70,000 hours of clean gameplay, and trained the IDM on 2,000 hours of contractor data [^126^]. Fine-tuned with RL, the model achieved diamond pickaxe crafting with a low but non-zero success rate (~2.5% foundation model, improving after fine-tuning). The lesson is unambiguous: RL *can* solve long-horizon Minecraft tasks, but the scale — 720 V100 GPUs for foundation model training — is prohibitive for a solo developer on consumer hardware. The VPT recipe (IDM → pseudo-label → foundation model → fine-tune) is a useful conceptual framework but not a practical starting point.

#### 1.1.4 DeepMind XLand: Open-Ended Multi-Agent Lessons

XLand is not a Minecraft project, but it is the most important multi-agent learning system of the RL era. Developed at DeepMind, XLand is a custom 3D multiplayer environment where agents train with and against each other on procedurally generated tasks, with difficulty dynamically adjusted via Prioritized Level Replay (PLR) and meta-learning for within-episode adaptation [^162^].

XLand produced agents with general heuristic behaviors — experimentation, tool use, cooperation, competition — that transferred to unseen tasks [^165^]. Critically, it demonstrated that multi-agent training produces sophisticated social behaviors including theory-of-mind-like reasoning [^162^]. Key lessons: open-ended auto-curriculum is essential for multi-agent training; fictitious self-play (sampling co-players from historical agent versions) creates natural learning progression; and meta-learning is critical when agents must coordinate with unknown partners. The original environment was not open-sourced; XLand-MiniGrid is a simplified reimplementation. Study the methodology and port PLR and self-play to Minecraft.

| Project | Year | Approach | Key Contribution | Best Result | Code Status | Maintenance |
|---------|------|----------|-----------------|-------------|-------------|-------------|
| MineRL | 2019 | Imitation + RL benchmark | 60M+ demonstrations; competition framework | Modest ObtainDiamond success [^33^] | MIT; `minerllabs/minerl` | Maintenance mode since 2022 |
| VPT | 2022 | Video pretraining | IDM for labeling unlabeled video; 0.5B param foundation model | Diamond pickaxe crafting [^96^] | Partial (OpenAI) | Superseded |
| OpenAI Diamond | 2022 | RL from scratch | Internet-scale video → behavioral prior | ~2.5% foundation SR [^126^] | Blog post only | N/A |
| XLand | 2021–2023 | Multi-agent open-ended learning | PLR auto-curriculum; fictitious self-play | General transferable heuristics [^162^] | Not released (MiniGrid only) | N/A |

The RL generation established three enduring truths for this project. First, the human-like action space is too large for sample-efficient RL without strong priors — which motivated the shift to LLM-based planning. Second, internet-scale behavioral priors work but require compute resources beyond a solo developer's reach. Third, and most importantly, XLand proved that multi-agent training in an open-ended 3D environment can produce emergent social behaviors — the core objective of a Minecraft AI village.

### 1.2 LLM-Based Agents (2023)

The year 2023 marked a paradigm shift. Rather than training policies from pixels and sparse rewards, researchers began using Large Language Models (LLMs) as planners and reasoning engines, with structured low-level controllers for execution. This approach — treating Minecraft as a text-and-code environment rather than a vision-and-controls environment — produced dramatically better results on long-horizon tasks. Six projects defined this generation.

#### 1.2.1 Voyager: GPT-4 Lifelong Learning with Code-as-Policy

Voyager (Wang et al., NeurIPS 2023 Workshop) is the landmark LLM-powered lifelong learning agent for Minecraft and the single most influential codebase in the field [^27^]. Its architecture comprises three core components. First, an **Automatic Curriculum** uses GPT-4 to generate progressively harder tasks, maximizing exploration based on the agent's current state — inventory, location, completed and failed tasks. Second, a **Skill Library** is an ever-growing vector database of executable JavaScript code (Mineflayer API calls), indexed by GPT-3.5-generated descriptions. Each skill is a reusable function such as `craftStoneSword()` or `combatZombieWithSword()`. Third, an **Iterative Prompting Mechanism** implements a feedback loop where GPT-4 generates code, the code executes in Minecraft, environment feedback and execution errors are collected, and GPT-4 refines the code. A self-verification module confirms task completion before committing a skill to the library [^27^].

Voyager's observation space is entirely symbolic: inventory state, agent stats (health, hunger), nearby entities, block types, biome information, and chat messages — all extracted through Mineflayer's API rather than from raw pixels. Its action space is the defining innovation: JavaScript code that calls Mineflayer APIs. This **code-as-policy** approach treats temporally extended actions as programs rather than low-level motor commands, making complex behaviors composable and interpretable.

The results are striking. Voyager achieved 3.3 times more unique items collected, 2.3 times longer distances traveled, and up to 15.3 times faster tech tree milestone unlocking compared to prior state of the art [^27^]. The skill library effectively eliminated catastrophic forgetting — skills learned in one world transferred to new worlds. The iterative prompting mechanism (incorporating environment feedback, execution errors, and self-verification) dramatically improved code correctness [^27^].

However, Voyager has critical limitations. The system depends heavily on GPT-4: GPT-3.5 ablations showed dramatic performance drops, and weaker models are effectively non-viable [^31^]. It has no visual perception, relying entirely on symbolic state — it is "blind" to visual details. It is single-agent only with no mechanism for multi-agent coordination. The skill library can grow unwieldy over time; without pruning, retrieval quality degrades.

The codebase (`MineDojo/Voyager`, MIT license, 6,900 stars) was archived in July 2023 and is no longer actively maintained [^404^]. Community forks exist — notably Co-voyager with 18 commits ahead of main [^171^] — but the original repository is frozen. Despite this, Voyager's architecture remains the canonical pattern for LLM-driven skill acquisition. Its modular design (curriculum, skill library, iterative prompting) maps cleanly to multi-agent extensions: the skill library can be shared across agents, the curriculum can generate multi-agent tasks, and the prompting mechanism can incorporate inter-agent communication [^27^].

#### 1.2.2 GITM: Structured Action Primitives and LLM Planning

Ghost in the Minecraft (GITM; Zhu et al., 2023) uses a hierarchical LLM architecture: a Decomposer breaks high-level goals into sub-goals using Minecraft Wiki knowledge; a Planner generates sequences of structured actions; and an Interface translates actions into keyboard and mouse operations [^32^]. The observation space combines LiDAR rays (five-degree intervals), voxel observations within a ten-unit radius, inventory, and life status — no RGB images. The action space comprises nine structured primitives (equip, explore, approach, mine/attack, dig down, go up, build, craft/smelt, apply). Notably, breaking speed and strength were set to 100, giving the agent superhuman mining capabilities [^32^].

GITM achieved 67.5% success on ObtainDiamond — a 47.5% improvement over prior methods — and was the first agent to procure all items in the Overworld technology tree, running on a single 32-core CPU with no GPU [^32^]. The limitations are severe for village simulation: LiDAR-only observations miss visual information entirely, superhuman abilities distort the task difficulty, the agent relies heavily on Wiki knowledge with no learning from experience, and hardcoded BFS/DFS exploration is inefficient for large terrain.

Code is at `OpenGVLab/GITM` (640 stars, June 2023) with no clear license and limited maintenance [^224^]. Study the hierarchical decomposition pattern for multi-agent task allocation, but the superhuman abilities make it unsuitable for realistic village simulations.

#### 1.2.3 DEPS: Interactive Planning with Learned Feasibility

DEPS (Describe-Explain-Plan-Select; Wang et al., NeurIPS 2023) introduces an interactive planning framework with four components: a Descriptor summarizes the current situation on failure; an Explainer uses an LLM to self-explain the failure; a Planner regenerates the plan with error context; and a Selector — a learnable module, not just prompting — ranks candidate sub-goals by estimated completion steps for feasibility [^59^]. DEPS was the first zero-shot multi-task agent to accomplish over 70 Minecraft tasks, with the GPT-4 planner achieving ~90% success on MT1 tasks versus ~50% for vanilla GPT-4 [^58^]. The system generalized to non-Minecraft domains including ALFWorld [^59^].

The practical downsides are significant: high planning latency from multiple LLM calls per replan; dependency on the cancelled Codex API requiring model migration [^166^]; and code split across three repos (`CraftJarvis/MC-Planner`, `CraftJarvis/MC-Controller`, `CraftJarvis/MC-Simulator`) with the last meaningful update in March 2023. The CraftJarvis team has moved to newer projects [^169^]. The interactive planning loop and Selector's learned feasibility ranking are worth studying for multi-agent error recovery and task allocation, but significant refactoring would be required.

#### 1.2.4 Plan4MC: Skill Graphs Pre-Generated by LLMs

Plan4MC (Yuan et al., NeurIPS 2023 FMDM Workshop) is a demonstration-free RL agent that combines skill learning with LLM-assisted planning [^99^]. It defines three types of fine-grained basic skills trained with RL: Finding-skills (exploration policies that locate items), Manipulation-skills (policies for interacting with items and mobs), and Crafting-skills (hardcoded crafting recipes). Before task execution, an LLM pre-generates a directed graph of skill dependencies; a Depth-First Search-based algorithm then walks this graph to produce executable skill plans [^99^].

Plan4MC is the most sample-efficient demonstration-free RL method in the literature, solving 40 diverse tasks with only 7 million environment steps. The Finding-skill innovation dramatically improved success rates — providing better initialization for Manipulation-skills (0.40 conditional success with Finding versus 0.25 without) [^72^]. The skill graph plus search approach proved more reliable than direct LLM planning for long-horizon tasks [^99^].

The limitations are architectural. Finding-skills are not goal-aware during exploration. The skill graph depends on LLM domain knowledge — if the LLM lacks knowledge, the graph is incorrect. Only 40 tasks were evaluated, far fewer than LLM-based agents. Performance degrades on tasks requiring more than 10 sequential skills [^72^].

Code is available at `PKU-RL/Plan4MC` (200 stars, MIT license, last updated March 2024) with pre-trained models included. The skill graph concept is highly applicable to multi-agent coordination: multiple agents can each specialize in different skills, the graph can be extended to include communication and coordination primitives, and skill search can allocate sub-tasks to different agents [^99^]. For a village project, the Finding-skills are particularly relevant — exploration policies that locate wood, stone, ore, and food sources are foundational capabilities.

#### 1.2.5 JARVIS-1: Memory-Augmented Multimodal LLM

JARVIS-1 (Wang et al., T-PAMI 2024) is a multimodal agent built on pre-trained Multimodal Large Language Models (MLLMs) [^63^]. Its architecture combines four components: Multimodal Perception processes RGB visual observations together with textual instructions through an MLLM; Planning generates language plans from visual and text input; Goal-Conditioned Controllers dispatch plans to STEVE-1/VPT-based controllers for low-level execution at 20Hz; and a Multimodal Memory system combines pre-trained Minecraft knowledge, actual gameplay experiences, and multimodal state-action sequences [^63^].

JARVIS-1 achieved nearly perfect performance on over 200 tasks from the Minecraft Universe Benchmark, and a 12.5% success rate on ObtainDiamondPickaxe — five times improvement over previous records [^62^]. Critically, it demonstrated self-improvement in lifelong learning experiments: performance increased with more gameplay experience, and the multimodal memory proved essential for this improvement [^63^].

However, the released code is incomplete. The multimodal descriptor and retrieval components were not fully released; the GitHub README has listed these as "Coming Soon" since 2023 [^95^]. The system is complex with many moving parts, making replication difficult. It requires significant computational resources (GPU for the vision model plus API costs for the LLM). The architecture is fundamentally single-agent.

The codebase (`CraftJarvis/JARVIS-1`, 395 stars, unspecified license) has seen the team move to newer projects (OmniJARVIS, Optimus series). The multimodal memory concept — combining pre-trained knowledge, actual gameplay experiences, and state-action sequences — could be extended to a shared multi-agent memory system. But the incomplete release makes extension difficult [^95^].

#### 1.2.6 STEVE-1, MP5, and Other 2023 Agents

Two additional 2023 agents merit brief mention for methodological contributions with limited direct applicability.

**STEVE-1** (Lifshitz et al., NeurIPS 2023) is an instruction-tuned generative model using the unCLIP methodology from DALL-E 2: a prior predicts MineCLIP latent codes from text, and a VPT-based policy executes them [^92^]. It completed 12 of 13 early-game tasks and cost only $60 to train. However, it exhibits severe behavioral biases from VPT pretraining — "poor performance across all hunting tasks except hunt a spider" — and prompt engineering is critical: "Kill a {animal}" works while "hunt a {animal}" fails completely [^91^]. Code is at `Shalev-Lifshitz/STEVE-1` (MIT) but stagnant. Its value is as a low-level goal-conditioned controller (it serves this role within JARVIS-1).

**MP5** (Qin et al., CVPR 2024) is a five-module system (Parser, Percipient, Planner, Performer, Patroller) with active perception — multi-round visual questioning before acting [^57^]. It achieved 22% success on process-dependent tasks and 91% on context-dependent tasks, but MP5 without its planning module achieves 0% on process-dependent tasks, showing extreme brittleness [^64^]. Code availability is effectively paper-only. The modular design pattern could conceptually be distributed across agents, but no practical codebase exists to clone.

### 1.3 State of the Art (2024–2026)

The current generation of agents combines lessons from both the RL and LLM eras, using Mixture-of-Experts (MoE) architectures, open-world skill libraries, and increasingly sophisticated benchmarks. Two projects define this frontier.

#### 1.3.1 Optimus-3 (2025): MoE Generalist with Task-Level Routing

Optimus-3 is the state-of-the-art generalist Minecraft agent, built on the Qwen2.5-VL vision-language model with a novel MoE architecture [^42^]. Its key innovation is a Dual-Router Aligned MoE: a Task Router assigns inputs to task-specific experts (planning, action, captioning, embodied question-answering, grounding, reflection); a Layer Router accelerates action inference by selectively skipping intermediate layers; and a Shared Knowledge Expert maintains common knowledge across all tasks. Training uses Dual-Granularity Reasoning-Aware Policy Optimization (DGRPO), which provides fine-grained reward functions per task type — including a Dependency-Aware Synthesis Reward for planning that uses the crafting dependency path as a thinking reward, and a Hallucination-Aware Consistency Reward for vision tasks [^42^].

Optimus-3 achieves the highest success rate across all seven task groups on long-horizon tasks, with 15% success rate on the Diamond Group. On the Diamond Sword open-ended task, it achieves 35% success and 69% completion rate — substantially above all baselines including JARVIS-1 and GPT-4o [^5^]. The MoE architecture effectively eliminates task interference seen in dense Qwen2.5-VL variants, and task-level routing outperforms token-level routing on captioning, planning, and grounding tasks [^42^].

Notably, the entire data collection pipeline cost only $300 in API costs with four L40 GPUs over 36 hours. The system is self-contained, performing planning and action prediction without external tools or models [^42^].

The limitations are practical. Deployment requires a GPU with at least 32GB VRAM — accessible but not trivial [^125^]. The system's complexity (dual routers, MoE, DGRPO) makes modification or extension difficult. It remains fundamentally single-agent, with no multi-agent coordination mechanism. As a relatively new system, it has not been as extensively validated as Voyager or JARVIS-1 [^125^].

The codebase (`JiuTian-VL/Optimus-3`, HuggingFace models available) is actively maintained by the Harbin Institute of Technology and Peng Cheng Laboratory team, with updates through March 2026 including Optimus-3-v2, MineSys2 Benchmark, and the OptimusM4 Dataset [^125^]. A GUI client (OptimusGUI) is available for interactive play. This is one of only two actively maintained major agent codebases in the field, alongside the CraftJarvis ecosystem.

#### 1.3.2 Odyssey: Open-World Skills Benchmark with Multi-Agent Extensions

Odyssey (ZJU-VIPA, ICLR 2025) is a framework and benchmark focused on diverse open-world skills beyond the technology tree [^124^]. It provides 40 primitive skills and 183 compositional skills covering combat, farming, cooking, animal husbandry, building, and exploration — far broader than prior agents that focused narrowly on ObtainDiamond. Odyssey fine-tunes LLaMA-3 on a 390,000+ instruction-entry question-answering dataset derived from Minecraft Wiki, providing strong domain knowledge without GPT-4 API costs [^124^]. The benchmark evaluates three task types: long-term planning, dynamic-immediate planning (reacting to environment changes), and autonomous exploration.

Odyssey uses Mineflayer-based state extraction (similar to Voyager) and JavaScript code calling Mineflayer APIs (Voyager-style code-as-policy). Its key advance over Voyager is the much richer skill library covering diverse gameplay beyond crafting. All datasets, model weights, and code are publicly available at `zju-vipa/odyssey` (384 stars, MIT license) [^128^]. In February 2025, Odyssey added multi-agent support with parallelized planning-acting, a centralized memory system, and a DAG-based skill library [^430^].

Performance on long-horizon tasks still lags behind GPT-4-based agents like Optimus-3, and the skill library, while extensive, may not cover all village-building scenarios. Evaluation is primarily benchmark-based with less emphasis on open-ended exploration metrics [^124^].

For a village-building project, Odyssey is the most directly relevant benchmark. Its farming, building, and animal husbandry skills map directly to village tasks. The Voyager-based architecture is easily extensible to multi-agent, and the benchmark includes combat scenarios relevant for village defense [^128^].

#### 1.3.3 Multi-Agent Gap Analysis: Why Nearly All Prior Art Is Single-Agent

A striking pattern emerges from this survey: despite 14 major projects, genuine multi-agent coordination in Minecraft remains almost entirely unexplored. MineRL, VPT, Voyager, GITM, DEPS, Plan4MC, JARVIS-1, MP5, STEVE-1, Optimus-3, and Odyssey are all fundamentally single-agent systems. Multi-agent work is nascent and mostly from 2024–2026.

The few multi-agent projects are worth noting. **MineLand** (cocacola-lab/MineLand, 111 stars, MIT license) supports up to 48 agents with limited multimodal senses and a Gym-style API [^432^]. **VillagerAgent** (cnsdqd-dyb/VillagerAgent, 92 stars) provides graph-based task coordination with a TaskManager, DataManager, and GlobalController — the closest academic precedent to a multi-agent village system [^183^]. **mindcraft** (mindcraft-bots/mindcraft, 5,300 stars, MIT license) is a Node.js-based multi-agent framework using LLMs plus Mineflayer that directly enables multi-agent collaboration with profile-based agent configuration [^212^]. **Project Sid** by Altera AI demonstrated 1,000 agents with emergent culture, religion, and governance, but the code is not open source [^424^].

This gap represents both the project's primary risk and its primary opportunity. The risk is that there is no established architecture for multi-agent Minecraft AI — the system must be built from first principles, combining lessons from single-agent LLM planners (Voyager, GITM), multi-agent training methodologies (XLand), and multi-agent simulators (MineLand). The opportunity is that a working 4-role cooperative village system would be a genuinely novel contribution at the intersection of three under-explored areas: multi-agent RL in open-world games, heterogeneous agent roles in Minecraft, and LLM-plus-RL hybrid architectures for embodied agents [^188^].

### 1.4 Clone-and-Study Recommendations

This section ranks the repositories most relevant to a solo developer building a multi-agent AI village, using a scoring rubric that weights four factors: (1) architectural relevance to the village use case (25%), (2) license permissiveness (20%), (3) maintenance status and recency (25%), and (4) code completeness and documentation (30%). Each factor is scored 1–10, producing a weighted total out of 10.

#### 1.4.1 Ranked Repositories with Justification

| Rank | Repository | Stars | Last Commit | License | Arch. Relevance | License Score | Maintenance | Completeness | **Total** | Verdict |
|------|-----------|-------|-------------|---------|----------------|---------------|-------------|--------------|-----------|---------|
| 1 | `mindcraft-bots/mindcraft` [^212^] | 5.3k | May 2026 | MIT | 9/10 | 10/10 | 10/10 | 9/10 | **9.4** | Clone first |
| 2 | `JiuTian-VL/Optimus-3` [^125^] | 200+ | Mar 2026 | Assumed MIT | 9/10 | 8/10 | 9/10 | 8/10 | **8.6** | Study architecture |
| 3 | `zju-vipa/odyssey` [^128^] | 384 | Oct 2025 | MIT | 8/10 | 10/10 | 7/10 | 8/10 | **8.1** | Clone for skills |
| 4 | `MineDojo/Voyager` [^404^] | 6.9k | Jul 2023 | MIT | 8/10 | 10/10 | 3/10 | 9/10 | **7.1** | Study architecture only |
| 5 | `cocacola-lab/MineLand` [^432^] | 111 | Jan 2025 | MIT | 7/10 | 10/10 | 6/10 | 7/10 | **7.2** | Reference for multi-agent |
| 6 | `PKU-RL/Plan4MC` [^99^] | 200 | Mar 2024 | MIT | 6/10 | 10/10 | 5/10 | 7/10 | **6.8** | Reference for RL skills |
| 7 | `cnsdqd-dyb/VillagerAgent` [^183^] | 92 | Mar 2026 | N/A | 8/10 | 4/10 | 7/10 | 5/10 | **6.0** | Study for task graphs |
| 8 | `CraftJarvis/JARVIS-1` [^95^] | 395 | Apr 2024 | N/A | 5/10 | 4/10 | 3/10 | 4/10 | **4.0** | Read code, don't clone |
| 9 | `OpenGVLab/GITM` [^224^] | 640 | Jun 2023 | N/A | 4/10 | 4/10 | 3/10 | 4/10 | **3.6** | Read paper only |
| 10 | `CraftJarvis/MC-Planner` [^166^] | 50+ | Mar 2023 | MIT | 4/10 | 10/10 | 2/10 | 5/10 | **4.8** | Historical reference |

**Rank 1: mindcraft.** This is the single most relevant repository. It is multi-agent by design, actively maintained (69 contributors, commits through May 2026), profile-based (each agent has configurable personality and goals via JSON), LLM-agnostic (supports OpenAI, Google, local models), and ships with Docker support. The MineCollab benchmark provides a multi-agent collaboration evaluation framework [^212^]. mindcraft is built on Mineflayer, so it shares architectural DNA with Voyager and Odyssey. The JSON profile system maps naturally to village roles (farmer, miner, builder, defender). Fork this repo first.

**Rank 2: Optimus-3.** The state-of-the-art single-agent system. Its MoE task router is conceptually extensible to multi-agent task allocation — each agent's router could include coordination tasks. The DGRPO training methodology and the end-to-end MLLM design (where communication can be another output modality) are directly relevant. The active maintenance and comprehensive codebase make it a better foundation than abandoned projects [^125^]. The dependency on a 32GB VRAM GPU is a constraint but not a blocker given the project's RTX 4080 hardware.

**Rank 3: Odyssey.** The most diverse skill library (40 primitives plus 183 compositional skills) covering farming, building, animal husbandry, cooking, and combat — all directly applicable to village tasks. The Voyager-based architecture is familiar and well-documented, and the February 2025 multi-agent extension adds centralized memory and parallelized planning-acting [^128^]. Clone this for the skill library and multi-agent memory architecture.

**Rank 4: Voyager.** Despite being archived, Voyager's architecture is the canonical pattern for LLM-driven skill acquisition. Study its skill library data structures (vector database of executable code with embedding-based retrieval), automatic curriculum design, and the code generation-execution-verification loop. Do not fork this for active development — use Odyssey or mindcraft instead [^404^].

**Rank 5: MineLand.** The best academic multi-agent simulator for Minecraft. Study its three-module architecture (Python bot plus Java environment plus JavaScript bridge), its limited-sense agent design (agents with partial information must communicate), and its Docker-based deployment. This is a reference architecture, not a codebase to fork directly [^432^].

**Ranks 6–10: Reference-only.** Plan4MC for RL skill training patterns and skill graph design. VillagerAgent for graph-based task decomposition. JARVIS-1 for multimodal memory concepts (but incomplete code makes cloning unrewarding). GITM and MC-Planner for historical understanding of the planning evolution.

#### 1.4.2 What to Extract from Each Codebase

| Repository | Architectural Patterns | Skill Library Design | Observation / Action Formats | Multi-Agent Lessons |
|-----------|----------------------|---------------------|----------------------------|-------------------|
| `mindcraft` | Profile-driven agent config; LLM-agnostic provider abstraction; inter-agent chat protocol [^212^] | JSON-constrained action schemas; profile-templated system prompts | Mineflayer state (position, inventory, nearby blocks/entities); JavaScript code actions | Multi-agent orchestration via chat; role specialization via profiles; shared world state |
| `Optimus-3` | MoE dual-router (task + layer); DGRPO training with dependency-aware rewards; end-to-end MLLM [^42^] | Task-specific expert modules; shared knowledge expert for cross-task transfer | Visual RGB + textual inventory/instruction; unified output tokens (plans, actions, reflections) | Task router extensible to multi-task allocation; communication as output modality |
| `Odyssey` | Parallelized planning-acting; interruptible execution; centralized multi-agent memory [^430^] | 40 primitives + 183 compositional skills as DAG; LLaMA-3 fine-tuned for domain knowledge | Mineflayer state extraction; JavaScript code-as-policy | Centralized memory with shared skill DAG; interruptible execution for real-time coordination |
| `Voyager` | Automatic curriculum generation; iterative prompting with error feedback; self-verification [^27^] | Vector DB of executable JS functions; GPT-3.5-generated descriptions for retrieval | Symbolic state (inventory, stats, entities, biome); JS code calling Mineflayer APIs | Skill library naturally shared across agents; curriculum extensible to multi-agent tasks |
| `MineLand` | Three-module architecture (Python/JS/Java); Gym-style ParallelEnv API; limited senses [^429^] | Alex agent framework with multitasking-based coordination | Limited vision, auditory, environmental senses (partial information); parameterized skill commands | 48-agent scaling; partial information forces communication; Docker deployment pattern |
| `Plan4MC` | Skill graph pre-generated by LLM; DFS-based skill search; demonstration-free RL training [^99^] | Three skill types (Finding, Manipulation, Crafting); directed dependency graph | RGB (160×256) via MineCLIP encoder; compressed discrete action space (12×3) | Skill graph extensible to coordination primitives; Finding-skills for resource location |

The extraction strategy should follow the development timeline of the project itself. In Phase 1, clone mindcraft and deploy a single gatherer agent with a basic profile — this validates the entire pipeline (Minecraft server → bot API → LLM provider → action execution) within hours. In Phase 2, integrate Optimus-3's MoE architecture for the perception-planning backbone, using the task router pattern even if the full MoE is too complex to port directly. In Phase 3, adopt Odyssey's skill library wholesale — the 223 skills (40 primitives plus 183 compositional) cover farming, building, animal husbandry, cooking, and combat, providing the behavioral primitives that village agents need. Throughout all phases, study Voyager's architecture (especially the skill library data model and automatic curriculum) and MineLand's multi-agent deployment patterns, adapting them rather than cloning them directly.

One final consideration: the CraftJarvis ecosystem (`CraftJarvis/MineStudio`, `CraftJarvis/OpenHA`, `CraftJarvis/ROCKET-1`) should be monitored closely [^595^]. This team has produced more actively maintained, high-quality Minecraft AI code than any other research group, and their work is likely to define the state of the art through 2026. MineStudio in particular is emerging as a unified development platform that combines simulator, trajectory data management, pre-trained model gallery, and training pipelines — it may become the de-facto standard environment for Minecraft AI research, superseding both MineRL and MineDojo [^595^].
