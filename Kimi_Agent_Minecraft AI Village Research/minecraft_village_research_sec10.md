## 10. Open-Source Projects to Study or Fork (~2500 words, 1 master table)

### 10.1 Top-Tier Repositories

#### 10.1.1 mindcraft: 5.3k stars, actively maintained multi-agent LLM framework, directly extensible

The `mindcraft` repository is the single most relevant codebase for building a multi-agent AI village in Minecraft. It is a Node.js-based framework that combines LLM-driven reasoning with the Mineflayer bot API, and it is multi-agent by design — multiple bots can coexist in a shared world, communicate via in-game chat, and collaborate on tasks [^212^]. As of May 2026, the project has 5,300 stars, 793 forks, 69 contributors, and a commit velocity that places it among the most actively maintained projects in the entire Minecraft AI ecosystem. The `develop` branch shows commits as recent as May 4, 2026, and the project releases versions regularly (v0.1.4 shipped March 20, 2026) [^212^].

What makes `mindcraft` uniquely suitable is its profile system. Each agent is configured via a JSON profile that specifies personality traits, goals, LLM backend (OpenAI, Google, Replicate, or local models), and behavioral parameters. This maps directly onto the village concept — a farmer profile, a miner profile, a builder profile, and a defender profile can be defined and instantiated independently, yet they share a world and can communicate [^212^]. The framework also supports code generation and execution sandboxing, allowing agents to write and run JavaScript code to solve novel problems. The MineCollab benchmark, published by the mindcraft team, provides standardized multi-agent collaboration tasks that can serve as evaluation protocols [^602^].

**Forkability: HIGH.** The MIT license permits unrestricted use. The architecture is modular — profiles, tasks, and services are separated — and Docker support with docker-compose simplifies deployment. A solo developer can start with the default profiles and incrementally add village-specific behaviors.

#### 10.1.2 MineLand: 48-agent academic simulator, Gym API, Docker support

MineLand is the most sophisticated academic multi-agent simulator for Minecraft. It supports up to 48 simultaneous agents with limited multimodal senses (visual field of view, auditory range, environmental awareness) and physical needs (food, health, resources) that force natural collaboration [^432^]. The simulator was presented at ICML 2024 and is built on a three-module architecture: a Python Bot Module that hosts agent logic, a Java Environment Module built on Fabric that runs the Minecraft server, and a JavaScript Bridge Module using Mineflayer that connects the two [^429^].

The MineLand simulator exposes a Gym-style API, making it compatible with standard RL frameworks, and provides Docker images for reproducible deployment [^432^]. The Alex agent framework, included in the repository, implements multitasking-based coordination inspired by cognitive science. With 111 stars, 24 forks, and 5 contributors, MineLand is smaller than `mindcraft` in community size, but its academic rigor and scale make it the best platform for research-grade multi-agent experiments. The last commit on the main `mineland` directory was January 8, 2025, with documentation updates as recent as September 30, 2025 [^432^].

**Forkability: HIGH.** MIT license. The simulator is the product — agents can be customized without modifying the core environment. The three-module architecture (Python+Java+JS) is clean and extensible.

#### 10.1.3 Baritone + mineflayer: essential building blocks for any bot architecture

Two infrastructure repositories are essential building blocks for any Minecraft bot project, regardless of the higher-level agent framework chosen.

**Baritone** (`cabaletta/baritone`) is the gold-standard pathfinding and automation library for Minecraft. With 8,900 stars, 1,900 forks, 71 contributors, and commits through May 18, 2026, it is one of the most actively maintained Minecraft automation projects [^150^]. Baritone implements A* pathfinding with chunk caching, cost calculation, and fallback strategies. It supports both Fabric and Forge mod loaders and provides full automation capabilities: mining, building, farming, following, and exploring. The LGPL-3.0 license requires derivative works to be open source, which is compatible with this project's goals. The API is heavily documented with Javadocs, and the settings system allows fine-grained behavior configuration [^150^].

**mineflayer** (`PrismarineJS/mineflayer`) is the foundational JavaScript library for creating Minecraft bots. With 7,000 stars, 1,300 forks, 231 contributors, and releases through May 3, 2026 (v4.37.1), it is the protocol layer that most non-RL agent frameworks build upon [^600^]. mineflayer supports Minecraft versions 1.8 through 1.21.11, provides a plugin system for extensibility, and includes Python examples and Google Colab notebooks. The library handles physics simulation, crafting, inventory management, digging, building, and chat interaction. MIT license [^600^].

**Forkability: HIGH for both.** Baritone can be included as a library dependency; mineflayer is a direct dependency for JS-based agents. Both are designed to be consumed as libraries rather than forked, but their source code is invaluable for understanding Minecraft protocol internals.

#### 10.1.4 CraftJarvis MineStudio: unified dev platform, PyPI package, active research group

MineStudio from the CraftJarvis team is emerging as the unified development platform for Minecraft AI research. It combines a customizable simulator (based on MineRL), a trajectory data system for efficient storage and retrieval, a model gallery with pre-trained checkpoints (VPT, GROOT, STEVE-1, ROCKET-1, ROCKET-2), and both offline and online training pipelines built on PyTorch Lightning and Ray [^595^]. The package is available on PyPI (`pip install MineStudio`) and includes Docker support.

The CraftJarvis team is the most prolific research group in the Minecraft AI space in 2025–2026, with publications at CVPR, ICML, and NeurIPS. Their active maintenance record — the `minestudio` directory shows commits through August 11, 2025, and README updates through October 10, 2025 — makes MineStudio the modern replacement for MineDojo [^595^]. The 377 stars and 32 forks underestimate its importance; this is infrastructure that the research community is actively adopting.

**Forkability: HIGH.** MIT license, designed as an extensible platform. The PyPI package means it can be treated as a dependency, while the source repository provides templates for adding new models and training pipelines.

### 10.2 Reference-Only but Essential

#### 10.2.1 Voyager: study skill library architecture, don't depend on (archived)

Voyager is the most influential single-agent LLM framework in the Minecraft AI field, with 6,900 stars and 673 forks [^404^]. Its three-component architecture — automatic curriculum, skill library with vector embeddings, and iterative prompting with execution feedback — has been adopted or extended by nearly every subsequent LLM agent project. The skill library pattern, where executable JavaScript code is stored with embedding-based retrieval, is the canonical design for LLM-driven skill acquisition [^404^].

However, Voyager was archived in July 2023. The last commit on the `voyager` directory was July 23, 2023. Only 12 contributors ever worked on the project, and the 35 total commits indicate a relatively thin codebase [^404^]. The dependency on specific versions of prismarine-block and chromadb creates fragility. A developer should clone Voyager, study its architecture thoroughly — especially the `skill_library/` data structures and the `voyager/` prompting loop — and then reimplement the patterns within a more modern framework like `mindcraft` or Odyssey [^404^].

**Forkability: REFERENCE ONLY.** MIT license, but archived and unmaintained. The Co-voyager community fork exists but has limited activity [^171^].

#### 10.2.2 GITM: hierarchical planner design patterns for LLM subgoal decomposition

Ghost in the Minecraft (GITM) demonstrated that purely LLM-based planning with no GPU training could achieve a 67.5% success rate on ObtainDiamond, improving on OpenAI's VPT by 47.5 percentage points [^224^]. Its hierarchical decomposition — goal to sub-goal to action to operation — with text-based knowledge from the Minecraft Wiki and a structured memory system, is the cleanest example of LLM subgoal decomposition in the field [^32^].

The repository has 640 stars and 24 forks, but only 14 commits and a single contributor. The last commit was June 5, 2023. No explicit open-source license is specified [^224^]. The code is minimal — primarily a README with the paper PDF and some figures. The primary value is architectural: the four-level hierarchy and the use of BFS/DFS exploration strategies demonstrate how an LLM can decompose complex Minecraft tasks without any RL training. This pattern can be adapted for village-level task planning.

**Forkability: REFERENCE ONLY.** No explicit license, minimal code. Study the hierarchy design and adapt it.

#### 10.2.3 MineDojo: task formalization and MineCLIP reward model design

MineDojo is the foundational RL environment framework for Minecraft AI research. With 2,200 stars, 194 forks, and 8 contributors, it provides 3,000+ programmatic tasks, a Gym-style API, and MineCLIP — a contrastive vision-language model trained on Minecraft YouTube videos that provides language-conditioned rewards [^604^]. The internet-scale knowledge base integration (730K YouTube videos, 7K Wiki pages, 340K Reddit posts) was a novel contribution that influenced subsequent work.

However, MineDojo's last meaningful commit was April 26, 2023, with README updates in August 2023. The project is effectively superseded by MineStudio [^604^]. The primary value of studying MineDojo lies in its task formalization — how it defines programmatic goals using game state predicates — and the MineCLIP reward model design, which demonstrates how vision-language models can provide dense reward signals without hand-crafted reward engineering. Both concepts are directly applicable to designing reward functions for village-building tasks.

**Forkability: REFERENCE ONLY.** MIT license, but maintenance mode since 2023. Use MineStudio instead for new development.

#### 10.2.4 Odyssey: parallelized planning-acting DAG architecture

Odyssey is a Voyager-based framework that extends the single-agent paradigm with multi-agent support and a comprehensive open-world skill library. With 384 stars, 24 forks, and 5 contributors, it provides 40 primitive skills and 183 compositional skills covering combat, farming, cooking, animal husbandry, building, and exploration — far beyond the tech-tree focus of earlier agents [^128^]. The Multi-Agent module, added in April 2025, introduces parallelized planning-acting with interruptible execution and a DAG-based skill library [^430^].

The repository also includes an MC-Crawler for Minecraft Wiki data collection and a MineMA LLM fine-tuning pipeline, both valuable for building domain-specific knowledge bases [^128^]. The last commit on the main Odyssey directory was October 22, 2025, indicating active development. Accepted at IJCAI 2025.

**Forkability: MEDIUM-HIGH.** MIT license. The Voyager heritage means the architecture is familiar, and the multi-agent additions are directly relevant. The skill library is the most reusable component.

### 10.3 Emerging and Watch-List

#### 10.3.1 VillagerAgent: graph-based multi-agent task coordination (ACL 2024)

VillagerAgent is the closest academic precedent to the village-building concept. With 92 stars, 16 forks, and 4 contributors, it provides graph-based task decomposition with a TaskManager for planning, a DataManager for state tracking, and a GlobalController for orchestration [^593^]. The framework includes VillagerBench, a benchmark with construction, cooking, and farming tasks, and supports human-agent and agent-agent real-time chat. PPO-based RL training for LLM API ranking was added in December 2024.

The repository name is `VillagerAgent-Minecraft-multiagent-framework` (the original `cnsdqd-dyb/VillagerAgent` URL redirects). Last commit was March 8, 2026, indicating ongoing maintenance [^593^]. No explicit open-source license is displayed. Presented at ACL 2024 Findings.

**Forkability: MEDIUM.** Python-based but somewhat coupled to its specific architecture. The graph-based task decomposition pattern is the most valuable takeaway.

#### 10.3.2 HeMAC benchmark environment for heterogeneous MARL evaluation

The Heterogeneous Multi-Agent Challenge (HeMAC) is a standardized benchmark environment built on PettingZoo for evaluating heterogeneous multi-agent reinforcement learning algorithms [^619^]. With 15 stars, 5 forks, and 2 contributors, it is a young project (initial release August 2025, last commit December 17, 2025) from Thales Group research. HeMAC features three agent types — Quadcopters, Observers, and Provisioners — with distinct observation spaces, action spaces, and capabilities, organized into three challenges of increasing heterogeneity.

While HeMAC is not Minecraft-specific, its design principles are directly applicable. The finding that IPPO outperforms MAPPO in highly heterogeneous scenarios [^1^] validates the algorithm recommendation in Chapter 7. The PettingZoo-based API means any MARL algorithm developed for HeMAC can be adapted to a Minecraft environment with minimal changes.

**Forkability: MEDIUM.** Standard Thales open-source license. The PettingZoo AEC API implementation pattern is reusable for Minecraft multi-agent environments.

#### 10.3.3 UnionClef: Py4J bridge implementation pattern

UnionClef is a monorepo that merges altoclef, Baritone (as "shredder"), and Tungsten into a single Fabric-compatible codebase for Minecraft 1.21 [^180^]. Its most valuable contribution for this project is the production Py4J two-way bridge that enables Java-to-Python communication with active maintenance through March 31, 2026. The bridge supports multi-instance launching, live screenshot capture, and a rich contextual method base for agents [^180^].

With only 7 stars, 1 fork, and 4 contributors, UnionClef is a hidden gem. The Py4J integration at `adris.altoclef.Py4JEntryPoint` provides the exact pattern needed for a Fabric mod to communicate with a Python-based MARL training loop. The project's GPL-3.0 license is compatible with this project's open-source goals [^180^].

**Forkability: HIGH.** GPL-3.0 license. The Py4J bridge implementation is the reference pattern for the Java-Python bridge layer. Study the `Py4JEntryPoint` class and the `scripts/` directory for the Python-side interface.

### Master Repository Table

The following table consolidates all 19 repositories, ranked by relevance to the multi-agent AI village project. "Relevance" prioritizes active maintenance, multi-agent support, and direct applicability to village-building scenarios. Stars and last commit dates are as of the research date (July 2026).

| # | Repository | Stars | Last Commit | License | Language | Tier | What to Learn | Forkability |
|---|-----------|-------|-------------|---------|----------|------|---------------|-------------|
| 1 | [mindcraft-bots/mindcraft](https://github.com/mindcraft-bots/mindcraft) | 5.3k | May 2026 | MIT | JavaScript | Top-Tier | Multi-agent orchestration, profile-driven behavior, LLM-to-game-action pipeline, inter-agent communication, code sandboxing [^212^] | **HIGH** — directly extensible |
| 2 | [cabaletta/baritone](https://github.com/cabaletta/baritone) | 8.9k | May 2026 | LGPL-3.0 | Java | Top-Tier | A* pathfinding, goal abstraction, movement cost calc, mod architecture [^150^] | **HIGH** — library dependency |
| 3 | [PrismarineJS/mineflayer](https://github.com/PrismarineJS/mineflayer) | 7.0k | May 2026 | MIT | JavaScript | Top-Tier | Full bot API, event-driven architecture, plugin patterns [^600^] | **HIGH** — library dependency |
| 4 | [cocacola-lab/MineLand](https://github.com/cocacola-lab/MineLand) | 111 | Jan 2025 | MIT | Python/JS | Top-Tier | 48-agent simulation, limited-sense design, Py-JS-Java bridge [^432^] | **HIGH** — simulator is the product |
| 5 | [CraftJarvis/MineStudio](https://github.com/CraftJarvis/MineStudio) | 377 | Aug 2025 | MIT | Python/Java | Top-Tier | Unified dev pipeline, trajectory data, model gallery, RL training [^595^] | **HIGH** — PyPI package |
| 6 | [3ndetz/unionclef](https://github.com/3ndetz/unionclef) | 7 | Mar 2026 | GPL-3.0 | Java | Top-Tier | Py4J two-way bridge, Java-Python interface pattern [^180^] | **HIGH** — bridge pattern |
| 7 | [MineDojo/Voyager](https://github.com/MineDojo/Voyager) | 6.9k | Jul 2023 | MIT | Python/JS | Reference | Skill library architecture, auto curriculum, code-as-policy [^404^] | **REF** — archived |
| 8 | [OpenGVLab/GITM](https://github.com/OpenGVLab/GITM) | 640 | Jun 2023 | N/A | Python | Reference | Hierarchical LLM planning, subgoal decomposition [^224^] | **REF** — minimal code |
| 9 | [MineDojo/MineDojo](https://github.com/MineDojo/MineDojo) | 2.2k | Aug 2023 | MIT | Java/Python | Reference | Task formalization, MineCLIP reward model, Gym API [^604^] | **REF** — superseded |
| 10 | [zju-vipa/odyssey](https://github.com/zju-vipa/odyssey) | 384 | Oct 2025 | MIT | Python/JS | Reference | Open-world skill lib (223 skills), multi-agent DAG, LLM fine-tuning [^128^] | **M-HIGH** — Voyager-based |
| 11 | [CraftJarvis/JARVIS-1](https://github.com/CraftJarvis/JARVIS-1) | 395 | Apr 2024 | N/A | Java/Python | Reference | Multimodal memory, visual perception, planning [^95^] | **MED** — incomplete release |
| 12 | [cnsdqd-dyb/VillagerAgent](https://github.com/cnsdqd-dyb/VillagerAgent-Minecraft-multiagent-framework) | 92 | Mar 2026 | N/A | Python | Emerging | Graph-based task coordination, centralized controller, inter-agent chat [^593^] | **MED** — coupled architecture |
| 13 | [CraftJarvis/OpenHA](https://github.com/CraftJarvis/OpenHA) | 33 | May 2026 | MIT | Python | Emerging | Hierarchical agents, CrossAgent inference, tool integration [^598^] | **HIGH** — very active |
| 14 | [ThalesGroup/hemac](https://github.com/ThalesGroup/hemac) | 15 | Dec 2025 | Thales OSS | Python | Emerging | HeMARL benchmarking, PettingZoo AEC API, heterogeneity evaluation [^619^] | **MED** — not Minecraft-specific |
| 15 | [CraftJarvis/ROCKET-1](https://github.com/CraftJarvis/ROCKET-1) | 46 | Feb 2025 | N/A | Java/Python | Emerging | Visual-temporal prompting, SAM-2 segmentation [^594^] | **MED** — vision-focused |
| 16 | [PKU-RL/Plan4MC](https://github.com/PKU-RL/Plan4MC) | 200 | Mar 2024 | MIT | Python | Emerging | RL skill training, skill graph generation, LLM planning [^407^] | **MED** — single-agent focus |
| 17 | [IranQin/MP5](https://github.com/IranQin/MP5) | 107 | Jun 2024 | N/A | Python | Emerging | 5-module agent design, active perception, modularity [^423^] | **MED** — modular pattern |
| 18 | [minerllabs/minerl](https://github.com/minerllabs/minerl) | 948 | Jul 2024 | MIT | Java/Python | Reference | Gym env, 60M frames human demos, RL foundation [^413^] | **REF** — use MineStudio |
| 19 | [lizaijing/Awesome-Minecraft-Agent](https://github.com/lizaijing/Awesome-Minecraft-Agent) | 67 | May 2026 | N/A | N/A | Reference | Comprehensive paper list, code links, field tracking [^624^] | **REF** — meta-resource |

The table reveals a clear pattern: the most actively maintained repositories (Tier 1) all show commits within the last six months and carry permissive licenses (MIT or LGPL). The six top-tier repositories collectively represent 22,066 stars and span the full technology stack — from low-level pathfinding (Baritone) to protocol-level bot control (mineflayer) to multi-agent LLM orchestration (mindcraft) to simulation infrastructure (MineLand, MineStudio) to the critical Java-Python bridge (UnionClef). Together, they provide a complete foundation without requiring any proprietary components.

The reference-only tier contains the most intellectually influential work but suffers from the archival problem endemic to academic research code. Voyager (6.9k stars, archived July 2023), GITM (June 2023), and MineDojo (August 2023) were all groundbreaking but none has seen meaningful maintenance in over two years [^404^] [^224^] [^604^]. The pattern is clear: academic code accompanies a paper, achieves its citation goal, and then decays. A production project must not depend on these repositories — it should study their architectures and reimplement the patterns in actively maintained frameworks.

The emerging tier represents the cutting edge. VillagerAgent's graph-based task coordination and OpenHA's hierarchical agent models both posted commits in March–May 2026, indicating active research [^593^] [^598^]. HeMAC, while not Minecraft-specific, is the only standardized benchmark for heterogeneous multi-agent RL and provides validated evidence that IPPO outperforms MAPPO in high-heterogeneity scenarios [^619^] — a finding that directly informed the algorithm recommendations in Chapter 7. These repositories belong on a watch list; they may produce breakthroughs that reshape the field within the next 12 months.

### Technology Stack Mapping

The top-tier repositories map cleanly onto the recommended architecture from preceding chapters:

| Architecture Layer | Recommended Repository | Rationale |
|-------------------|----------------------|-----------|
| Game Client | Minecraft 1.21+ Fabric | Best mod compatibility, server-side only |
| Low-Level Bot Control | Baritone (Java) + mineflayer (JS) | Mature, well-documented, active maintenance |
| Java-Python Bridge | UnionClef Py4J pattern | Production two-way bridge, active Mar 2026 |
| Agent Framework | mindcraft or MineLand | Multi-agent, LLM-driven, profile-based |
| Skill Management | Voyager pattern (reimplemented) | Composable, retrievable, growing library |
| Task Planning | VillagerAgent graph or GITM hierarchy | Proven for multi-step, multi-agent tasks |
| Memory | Odyssey centralized DB + JARVIS-1 multimodal | Shared village state, visual + symbolic |
| Simulator (testing) | MineLand | Headless, 48-agent, Docker, fast |
| Model Training | MineStudio | Unified pipeline, pre-trained models, Ray |
| MARL Algorithms | HARL library (HAPPO/HATRPO) | Theoretical guarantees for heterogeneity |

This stack leverages the `mindcraft` framework as the primary agent orchestration layer, with Baritone and mineflayer handling low-level execution, UnionClef's Py4J pattern bridging Java and Python, and MineStudio providing the training and model infrastructure. The reference-tier repositories inform the design of the skill library, task planner, and memory system without creating maintenance dependencies. The emerging-tier repositories are tracked for future integration as the project matures.
