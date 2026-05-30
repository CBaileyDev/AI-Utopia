## 8. Server Performance and Accelerated Training

The training throughput of a multi-agent reinforcement learning (RL) pipeline is bounded at its lowest level by how quickly the Minecraft environment can generate experience data. This chapter evaluates the achievable tick rates, parallel instance counts, and resource allocation strategies on the target hardware—a Ryzen 9800X3D (8 cores / 16 threads, 96 MB 3D V-Cache), 64 GB DDR5-6000, and an RTX 4080 16 GB. The central question is practical: can this workstation run 4–8 parallel Minecraft server instances fast enough to keep the GPU saturated during policy updates? The evidence points to a firm yes for 4–6 instances, with headroom for tick acceleration on select servers, provided the correct mod stack and JVM tuning are applied.

### 8.1 Acceleration Techniques

#### 8.1.1 Carpet Mod Tick Sprint and Warp

Carpet Mod, developed by Gnembon, provides the `/tick` command family for precise server tick control. For RL training, the two most relevant commands are `/tick rate <rate>` (which sets a fixed TPS ceiling) and `/tick sprint <ticks>` (formerly `/tick warp`, which processes ticks as fast as CPU allows). Vanilla Minecraft (1.20.3+) now includes a native `/tick` command, but Carpet Mod's implementation offers broader control, including `/tick freeze`, `/tick unfreeze`, and `/tick step <n>` for deterministic tick-by-tick execution [^463^]. These commands make Carpet Mod essential for RL pipelines where reproducible evaluation and accelerated training rollouts are both required.

The theoretical maximum of `/tick sprint` was demonstrated at over 1,000 TPS in Gnembon's original 2018 video [^582^]. In practice, the achievable ceiling depends on single-thread CPU performance, world complexity, and entity count. The Ryzen 9800X3D's Cinebench 2024 single-core score of 133 places it among the fastest consumer CPUs for Minecraft server workloads, and its 96 MB L3 cache provides exceptional bandwidth for the random-access memory patterns characteristic of entity ticking and block-state lookups [^519^]. For a village-sized loaded area (~100–200 chunks, ~50 entities), the following TPS ranges are realistic:

| Scenario | Achievable TPS | Stability |
|----------|---------------|-----------|
| Idle / light activity | 300–800 | High |
| Active villager AI + redstone | 100–300 | Moderate |
| Heavy entity collisions, pathfinding | 50–150 | Low–Moderate |

These figures carry medium confidence because no direct empirical benchmark exists for precisely village-sized worlds on the 9800X3D [^463^] [^519^]. Stability concerns at high tick rates are significant: effect timers desync from visual display [^579^], redstone contraptions behave differently under acceleration, and random-tick operations (crop growth, fire spread) scale with tick rate, potentially causing cascading block updates that stall the server watchdog. Pre-generating the world with the Chunky mod is mandatory before using tick sprint [^513^]. The practical recommendation is to reserve tick sprint for 1–2 dedicated fast-rollout instances while keeping the majority of training environments at a stable 20 TPS.

#### 8.1.2 Headless Fabric Server with `--nogui`

The `--nogui` flag disables the Swing-based server control window, saving ~100–200 MB RAM and minor CPU cycles [^540^]. For fully headless environments, set `java.awt.headless=true` or use an `xvfb-run` wrapper. This is the default mode for all RL training servers: the bot framework (Mineflayer, UnionClef, or Scarpet) handles all communication without human interaction.

Fabric Loader is the preferred mod platform. It has minimal overhead versus vanilla—approximately 1.2 GB base RAM versus 1.8 GB for Forge—and starts 50–60% faster than equivalent Forge setups [^590^]. Fabric API adds mod hooks without significant bloat, and Carpet Mod's server-side functionality operates without client-side components.

#### 8.1.3 Optimization Mod Stack

A carefully chosen mod stack is the difference between a server that stutters at 20 TPS and one that sustains accelerated rates with CPU cycles to spare. The following table summarizes the core optimization mods for a Fabric-based RL training server, with their measured performance contributions.

| Mod | Optimization Target | Performance Gain | RL Relevance |
|-----|--------------------|------------------|--------------|
| Lithium | Game logic: mob AI, collisions, chunk ticking, block entity processing | 30–50% faster tick times [^518^] | **Essential.** Reduces MSPT (milliseconds per tick) directly, enabling higher sustained TPS or lower CPU usage at 20 TPS. |
| FerriteCore | Block state and model object deduplication in RAM | 40–50% RAM reduction [^518^] | **Essential.** At 2–3 GB baseline per instance, this saves 1–1.5 GB—enabling more parallel instances within the 64 GB budget. |
| Krypton | Network protocol stack: packet compression, TCP tuning | Up to 40% less network CPU [^518^] | **Recommended.** Reduces overhead from bot communication (Mineflayer/UnionClef) at high observation frequencies. |
| C2ME | Parallelized chunk loading and generation | 70% faster chunk I/O [^513^] | **Recommended for warp instances.** Critical when tick sprint triggers rapid chunk loading; labeled experimental, so test before production use. |
| Alternate Current | Redstone wire update order and propagation | Up to 95% faster redstone [^518^] | **Conditional.** Only if the village contains redstone farms or automated mechanisms; otherwise unnecessary. |
| LazyDFU | DataFixerUpper deferral (startup only) | 20–30 seconds saved per launch [^518^] | **Quality of life.** Valuable during development when servers restart frequently. |

Lithium and FerriteCore together are non-negotiable: Lithium reduces per-tick CPU burden by optimizing entity collision detection, mob AI goal selection, and block entity ticking, while FerriteCore deduplicates block state objects to cut RAM usage. Together they transform a 3–4 GB, CPU-heavy vanilla server into a 2–3 GB, CPU-light instance—effectively doubling parallel capacity on fixed hardware. Krypton reduces network CPU from bot bridge traffic at high observation frequencies. C2ME should be deployed cautiously: its experimental status means it may interact unpredictably with Carpet Mod's tick acceleration; test on a single instance before fleet-wide deployment.

### 8.2 Parallel Instances

#### 8.2.1 Resource Requirements per Instance

The resource footprint of each Minecraft server instance depends on world size, entity count, and tick rate. The following table provides a granular breakdown for a Fabric server running Carpet Mod, Lithium, and FerriteCore, configured with `simulation-distance=4` and `view-distance=4` (the recommended settings for RL training, yielding 81 actively ticked chunks per player) [^544^] [^547^].

| Configuration | RAM | CPU (at 20 TPS) | MSPT Target |
|--------------|-----|-----------------|-------------|
| Base server (empty world) | 1.5–2 GB | 5–10% of one core | <10 ms |
| Village-sized loaded area (~100–200 chunks, ~50 entities) | 2–3 GB | 15–25% of one core | <25 ms |
| Active village + redstone farms (full villager AI, golems) | 3–4 GB | 30–50% of one core | <35 ms |
| With tick sprint (200+ TPS) | 3–5 GB | 60–100% of one core | N/A (core-saturated) |

Sources: [^462^] [^516^] [^467^]

At the "village-sized loaded area" tier—which represents the expected steady state for a single training environment—each instance consumes approximately 2–3 GB of RAM and 15–25% of a single CPU core when running at the standard 20 TPS. The 15–25% figure is critical for capacity planning: it means one physical core can comfortably host one instance with headroom, but attempting to run two instances on a single core will cause tick-time inflation and TPS drops below 20. The 9800X3D has 8 physical cores (16 threads), but Minecraft's main tick loop is fundamentally single-threaded, so logical threads do not provide proportional benefit for tick processing. Plan on 1 core per instance.

The RAM figures assume Aikar's tuned JVM flags, which optimize G1GC for Minecraft's short-lived object patterns [^532^] [^542^]. Critical parameters: `-Xms` equal to `-Xmx` (preventing heap resizing), `G1HeapRegionSize=8M`, `MaxGCPauseMillis=200`, and `MaxTenuringThreshold=1`. The full flag set is well-documented in the PaperMC and Aikar's guides; omit `-XX:+AlwaysPreTouch` when running inside Docker containers with memory limits, as it conflicts with cgroup-aware heap sizing [^532^].

#### 8.2.2 Realistic Concurrent Count on Target Hardware

With 8 physical cores and 64 GB RAM, the straightforward capacity calculation is:

- **CPU-bound:** 8 cores × 1 instance/core = 8 instances maximum
- **RAM-bound:** 64 GB / 3 GB per instance = ~21 instances maximum

The CPU is the limiting factor. However, practical considerations reduce the ceiling from 8 to 4–6 instances. First, the ML training process (PyTorch/Ray) and observation preprocessing require dedicated cores—budget 2 cores for the learner and data pipeline. Second, the operating system and background services consume resources. Third, tick sprint on even one instance will saturate its core, making that core unavailable for a second instance. The cross-verification analysis confirms this finding: "4–6 parallel instances at 20 TPS is realistic on 9800X3D" [^462^].

A recommended layout for the target hardware is:

| Instance | Port | Purpose | RAM | CPU Affinity |
|----------|------|---------|-----|--------------|
| 1 | 25565 | Training env A (Gatherer) | 3 GB | Core 0 |
| 2 | 25566 | Training env B (Builder) | 3 GB | Core 1 |
| 3 | 25567 | Training env C (Farmer) | 3 GB | Core 2 |
| 4 | 25568 | Training env D (Defender) | 3 GB | Core 3 |
| 5 | 25569 | Evaluation env (20 TPS, deterministic) | 3 GB | Core 4 |
| 6 | 25570 | Fast-rollout env (tick sprint) | 4 GB | Core 5 |
| — | — | ML training (Ray RLlib + PyTorch) | 16–24 GB | Cores 6–7 |
| — | — | OS, JVM overhead, buffers | 8–12 GB | Distributed |

This layout consumes 18–20 GB for Minecraft servers, 16–24 GB for ML training, and 8–16 GB for system overhead—totaling 42–60 GB of the 64 GB available, leaving 4–22 GB of headroom. The CPU assigns one dedicated core per instance, with two cores reserved for the learner process. If training uses the RTX 4080 heavily (typical for policy gradient methods with batch sizes above 1,024), the CPU's role during the update phase is primarily data movement and gradient aggregation, which two cores can handle comfortably.

An alternative hybrid configuration—2 instances with tick sprint (200–400 TPS) plus 2 normal instances—peaks at ~480 steps/sec but with significantly reduced stability. The observation bridge (Py4J or WebSocket) may fail to keep up with accelerated tick rates, making agent observation frequency the bottleneck rather than server TPS. Cross-dimensional analysis confirms: parallel normal-speed instances beat tick-warped instances for RL training stability [^462^].

#### 8.2.3 Docker Configuration for Reproducible Parallel Deployments

Docker provides process isolation and reproducible configuration across the parallel instance fleet. The `itzg/minecraft-server` image is the community standard and comes pre-configured with Java optimization [^575^]. Each container requires isolated data volumes and unique port mappings. The `--privileged` flag may eliminate lag spikes caused by Docker's security overhead when writing to world data directories [^512^].

A `docker-compose.yml` excerpt for four training instances:

```yaml
services:
  mc-env-a:
    image: itzg/minecraft-server
    ports:
      - "25565:25565"
    volumes:
      - ./world-a:/data
    environment:
      - TYPE=FABRIC
      - VERSION=1.21
      - FABRIC_LOADER_VERSION=0.16.9
      - MEMORY=3G
      - JVM_OPTS=-XX:+UseG1GC -XX:MaxGCPauseMillis=200 ...
      - ENABLE_RCON=true
      - RCON_PORT=25575
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 3.5G
```

Each additional instance increments the external port (`25566:25565`, `25567:25565`, etc.) and uses a separate world volume. The `cpus: '1.0'` limit enforces the one-core-per-instance rule at the cgroup level. For instances using tick sprint, increase the memory limit to 4.5 GB to accommodate the transient allocation spikes during accelerated ticking.

### 8.3 Training Throughput Ceiling

#### 8.3.1 Single Fast Server vs. Multiple Normal Servers

The architectural choice between one tick-sprinted server and multiple 20 TPS servers has implications beyond raw step count. The following comparison isolates the trade-offs:

| Approach | Peak Steps/sec | Stability | Observation Extraction | Setup Complexity |
|----------|---------------|-----------|----------------------|------------------|
| 1 server + tick sprint (300 TPS) | 300 | Low–Moderate | May drop frames at high speeds | Low |
| 4 servers × 20 TPS | 80 | High | Reliable, one obs per tick | Moderate |
| 6 servers × 20 TPS | 120 | High | Reliable, one obs per tick | Moderate |
| Hybrid: 2× warp + 2× normal | 440 (peak) | Moderate | Mixed reliability | High |

At first glance, the hybrid approach appears optimal with 440 peak steps/sec. However, this figure assumes the observation bridge can extract state at 200 TPS on the warped instances—an assumption that has not been validated empirically. Mineflayer and UnionClef both operate over network protocols with inherent latency; a full state extraction (entity positions, inventories, block changes) at 200 Hz is substantially harder than at 20 Hz. If observation extraction becomes the bottleneck, the effective step rate of warped instances drops to whatever the bridge can sustain, negating the TPS advantage.

The conservative and recommended approach is to start with 4 parallel instances at 20 TPS. This configuration provides 80 steps/sec of stable, deterministic experience with reliable observation extraction. Once the full pipeline (environment → observation → policy → action) is validated at this throughput, introduce tick sprint on a fifth instance and measure whether the observation bridge can keep pace. Do not optimize the server throughput until the training pipeline is proven to be CPU-bound on environment stepping rather than GPU-bound on policy updates.

#### 8.3.2 GPU Training While Running Servers: Resource Contention Analysis

The RTX 4080 serves dual roles: neural network training during policy updates (80–100% GPU utilization) and inference during rollout collection (10–30% utilization). The critical risk is CPU-GPU contention—if the CPU cannot feed data fast enough, GPU utilization drops [^554^]. The mitigation strategy has three components: (1) pin server instances to specific cores (cores 0–5) using `taskset` or Docker's `cpuset-cpus`, leaving cores 6–7 for ML training; (2) use asynchronous rollout collection via Ray RLlib's EnvRunner actors, which collect experience while the Learner performs GPU-based policy updates, so phases overlap rather than alternate [^605^]; (3) offload observation preprocessing to GPU where possible, converting raw state vectors into batched CUDA tensors to eliminate CPU-GPU transfer overhead.

The RTX 4080's 16 GB VRAM is sufficient for typical multi-agent RL models. A CTDE architecture with 4 agent policies, value networks, and a 1-million-step replay buffer consumes 6–10 GB VRAM, leaving 6–10 GB for batch processing and CUDA overhead. Large vision transformers for pixel-based observations would strain this budget; the symbolic observation approach (Chapter 3) avoids the constraint entirely.

#### 8.3.3 Ray Distributed Rollout Collection Architecture

Ray RLlib provides the distributed infrastructure to scale experience collection across the parallel Minecraft instances. The architecture maps naturally onto the multi-instance server layout: each Minecraft server instance becomes a Ray EnvRunner actor, with the number of actors controlled through `config.env_runners(num_env_runners=...)` [^605^]. Each EnvRunner hosts one or more environment copies (vectorization via `num_envs_per_env_runner`), batches policy inference across them, and streams experience tuples back to the central Learner actor.

For the village setup, the recommended Ray configuration sets `num_env_runners=4` (one per server instance), `num_envs_per_env_runner=1` (multi-agent environments are not yet vectorizable in RLlib), `num_cpus_per_env_runner=1` for dedicated core affinity, and `sample_timeout_s=60` to accommodate slow Minecraft environments [^605^]. The `STRICT_PACK` placement strategy keeps all actors on one node, with Ray's shared-memory object store passing experience batches without serialization overhead [^608^] [^610^].

EnvRunner fault tolerance is a practical advantage: Ray automatically restarts failed workers, preventing a single server watchdog timeout from killing the training run [^606^]. This is valuable during curriculum learning (Chapter 6), where unstable multi-agent configurations may crash servers more frequently than single-agent baselines.

The throughput ceiling with 4 EnvRunners at 20 TPS is ~80 environment steps per second. A typical PPO configuration collecting 4,096 steps per batch completes one gradient update every ~51 seconds; for a small network (2–4 layers, 256 hidden units), the GPU update takes 2–5 seconds, yielding a 4–10% GPU duty cycle. Scaling to 6 EnvRunners raises the step rate to 120/sec and GPU utilization to 6–15%. Higher utilization requires larger batch sizes, vectorized multi-agent environments (a known RLlib limitation the Ray team is actively addressing), or offline data supplementation via Ray Data [^605^].
