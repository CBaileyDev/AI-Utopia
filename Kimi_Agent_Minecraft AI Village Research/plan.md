# Multi-Agent Minecraft AI Village — Deep Research Plan

## Objective
Produce a comprehensive, cited research report covering 10 clusters of investigation to inform the architecture and implementation of a multi-agent RL-driven Minecraft village system.

## Stage 1: Parallel Deep Research (All Clusters Simultaneous)
Load: `deep-research-swarm` skill
Dispatch: 10 parallel research clusters, each with 2-3 subagents for breadth+depth.

### Cluster Assignments
- **Cluster 1**: Prior art in Minecraft RL and LLM agents (MineRL, MineDojo, Voyager, GITM, DEPS, Plan4MC, JARVIS-1, MP5, STEVE-1, VPT, OpenAI, DeepMind XLand)
- **Cluster 2**: Fabric mod ecosystem for bot/AI integration (Baritone, Altoclef, MineFlayer, Carpet Mod, MCP-Reborn, etc.)
- **Cluster 3**: Observation and action space design (pixels vs symbolic vs hybrid, action abstractions)
- **Cluster 4**: Multi-agent RL frameworks (PettingZoo, MARLlib, EPyMARL, Ray RLlib, etc.)
- **Cluster 5**: LLM-as-planner architectures (Voyager, GITM, DEPS, JARVIS-1, MP5 patterns)
- **Cluster 6**: Reward shaping for gatherer role (resource collection reward functions)
- **Cluster 7**: Multi-agent failure modes and mitigations (credit assignment, non-stationarity, etc.)
- **Cluster 8**: Server performance and accelerated training (tick warp, headless, parallel instances)
- **Cluster 9**: Geyser/Floodgate Bedrock compatibility assessment
- **Cluster 10**: Existing open-source projects to study or fork (repo discovery and evaluation)

### Per-Cluster Output Format
- Executive summary (3-5 bullets)
- Key findings with citations/links
- Concrete recommendations for the project
- Open questions needing human decision

### Final Synthesis (after all clusters return)
- Top 5 highest-priority decisions
- Showstoppers/major risks
- Revised milestone plan
- 10 most important papers/repos/docs to read first, ranked

## Stage 2: Report Assembly
Load: `report-writing` skill
Synthesize all cluster outputs into final structured report in Markdown, then convert to .docx.

## Stage 3: Artifact Production
Load: `docx` skill
Convert final .md to .docx for delivery.

## Deliverables
- `/mnt/agents/output/minecraft_ai_village_research.md` — full report
- `/mnt/agents/output/minecraft_ai_village_research.docx` — formatted document
