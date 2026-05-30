## 2. Fabric Mod Ecosystem for Bot/AI Integration

The preceding chapter mapped the academic landscape of Minecraft AI agents; this chapter shifts focus to the concrete modding infrastructure available for Fabric, the lightweight mod loader that has become the de facto standard for Minecraft Java Edition server-side development. For a solo developer aiming to field Python-controlled agents in a Fabric world, the critical question is not *which algorithm* to train but *which mods expose pathfinding, world state, or RPC interfaces* that a Python process can consume. This chapter surveys the active mod ecosystem, evaluates each candidate against the criteria of programmability, maintenance status, and Python accessibility, and concludes with a ranked list of five codebases worth reading before writing any custom code.

---

### 2.1 Pathfinding and Bot Frameworks

At the lowest level of agent control, three projects dominate the landscape: Baritone for embedded pathfinding in Java, UnionClef as a high-level task bot with a Python bridge, and MineFlayer as a protocol-level JavaScript alternative. Understanding their trade-offs is essential because the choice of pathfinding infrastructure constrains every upstream architectural decision.

#### 2.1.1 Baritone: Actively Maintained through 1.21.8, A* Pathfinding, LGPL-3.0

Baritone is the de facto standard Minecraft pathfinding library, originally developed by leijurv for the Impact client and now distributed as a standalone library with 8,900 GitHub stars. [^150^] It is actively maintained, with release v1.15.0 (August 2025) supporting Minecraft 1.21.6 through 1.21.8, and with continuous community contributions across Fabric, Forge, and NeoForge variants. [^200^]

Baritone exposes a clean Java API through the `baritone.api` package, providing A* pathfinding with chunk caching (reportedly 30 times faster than the original MineBot implementation), configurable goal types (`GoalXZ`, `GoalBlock`, `GoalNear`, `GoalGetToBlock`), and roughly one hundred tunable settings governing whether the bot may break blocks, place blocks, sprint, parkour, or fly with elytra. [^150^] The API is straightforward to invoke programmatically:

```java
BaritoneAPI.getSettings().allowSprint.value = true;
BaritoneAPI.getSettings().primaryTimeoutMS.value = 2000L;
BaritoneAPI.getProvider().getPrimaryBaritone().getCustomGoalProcess()
    .setGoalAndPath(new GoalXZ(10000, 20000));
```

The project is licensed under **LGPL-3.0**, which permits use as a library in custom utility clients but requires that derivative works of Baritone itself be released under a compatible license. [^50^] For a research project, this is generally unproblematic; for a closed-source commercial deployment, consult legal counsel.

**The Python problem:** Baritone offers no direct Python binding. It is Java-only, and the three viable integration paths all require additional engineering: (1) embedding a Py4J gateway inside a Fabric client mod and calling Baritone methods through Java reflection, as UnionClef demonstrates; (2) issuing chat commands (e.g., `#goto 1000 500`) from a Python process, which is brittle for programmatic use because command parsing lacks type safety and error propagation; or (3) wrapping BaritoneAPI calls behind a custom gRPC or WebSocket server running inside the mod. All three are feasible, but none are turnkey.

#### 2.1.2 Altoclef/UnionClef: Active Fork with Built-in Py4J Two-Way Bridge

Altoclef was a high-level task bot built on Baritone that achieved the first fully autonomous Minecraft completion in May 2021. [^47^] The original repository by gaucho-matrero was archived in June 2024, but a critical active fork emerged: **UnionClef**, maintained by 3ndetz as a monorepo unifying Altoclef, a Baritone "shredder" fork, and the Tungsten pathfinder. [^180^]

UnionClef is the single most important codebase for this project. As of April 2026 it has accumulated 420+ commits across four contributors, with the latest release v0.21.1 targeting Minecraft 1.21.1 on Fabric. [^180^] Beyond Baritone's pathfinding, UnionClef exposes a task system (`@get`, `@craft`, `@mine`, `@gamer`, `@deposit`, `@stash`), automatic crafting via the Recipe Book, PvP support (ender pearl clutches, arrow dodging, mace combat), and multiplayer features including auto-login and anti-cheat bypass rotation. [^160^]

The decisive feature is the **built-in Py4J two-way gateway**. [^160^] [^180^] The gateway port is configurable via `@set pythonGatewayPort <port>`, and multi-instance launching is supported so that each bot receives its own Py4J endpoint. From Python, the developer can send commands to the Java side, receive live data (position, health, inventory, nearby entities and blocks), and even request live screenshots. From Java, callbacks into Python enable bidirectional decision-making. The entry point class `adris.altoclef.Py4JEntryPoint` provides a well-defined API surface for this exchange. [^180^]

**Caveat:** The author has cautioned that the codebase was "mostly for prototyping" when begun, and the Java quality reflects a learning trajectory. However, 420+ commits and active CI/CD indicate substantial maturation. The GPL-3.0 license (incorporating Baritone's LGPL-3.0, Altoclef's MIT, and Tungsten's CC0-1.0) requires that distributed derivatives also be open-source. [^180^]

#### 2.1.3 MineFlayer: Protocol-Level JavaScript Alternative, 7k Stars, MIT

MineFlayer is a high-level JavaScript API for creating Minecraft bots that operates at the protocol level via `node-minecraft-protocol`, requiring no client modding. [^49^] It is the foundational library for virtually all modern Minecraft AI research: Voyager [^197^], mindcraft [^212^], VillagerAgent [^133^], and Odyssey [^128^] all build upon it.

With 7,000 GitHub stars, 1,200+ forks, and support for Minecraft 1.8 through 1.21.11, MineFlayer is very actively maintained. [^49^] [^199^] Its plugin ecosystem includes `mineflayer-pathfinder` for A* navigation, `prismarine-viewer` for 3D world rendering, `mineflayer-pvp` for combat, and `mineflayer-auto-eat` for survival automation. Python developers can consume MineFlayer through the `javascript` package (JSPyBridge), with examples available in `examples/python/` and on Google Colab. [^49^] [^204^]

**Trade-off assessment:** Choose MineFlayer when rapid prototyping, protocol-level control, or multi-server proxy architectures are required. Choose a Fabric mod approach (Baritone/UnionClef) when the project needs full client-side world rendering, anti-cheat bypasses, complex building, visual feedback (screenshots for vision models), or Baritone's superior pathfinding performance. For a multi-agent village simulation running on a local workstation, the Fabric mod path provides richer world state access and more reliable movement execution, at the cost of requiring one JVM instance per bot.

---

### 2.2 Server-Side Automation Tools

Once agents are connected to a Minecraft server, controlling the server's temporal and computational behavior becomes critical for reinforcement learning (RL) training pipelines. Carpet Mod and a suite of optimization mods provide this capability.

#### 2.2.1 Carpet Mod: Tick Warp, Scarpet Scripting, and RL Training Infrastructure

Carpet Mod, authored by gnembon (a core Fabric team contributor), is the standard server-side mod for technical Minecraft. It is actively maintained for versions 1.14.x through 1.21.11, with a broad ecosystem including carpet-extra and a public Scarpet app store. [^111^] [^112^]

For RL training, the `/tick` command family is the most important feature:

- `/tick rate <rate>` sets a fixed TPS (ticks per second) rate with nanosecond precision, effectively allowing the game to run as fast as the CPU permits. [^581^]
- `/tick sprint <ticks>` (formerly `/tick warp`) runs the game at maximum CPU utilization for a specified number of ticks. [^463^]
- `/tick freeze` pauses all game logic, and `/tick step <n>` processes exactly `n` ticks while frozen, enabling deterministic evaluation. [^463^]

On a Ryzen 9800X3D workstation with a village-sized loaded area (roughly 100–200 chunks, ~50 entities), Carpet Mod's sprint command can sustain 200–500 TPS under light load or 50–150 TPS when villager AI and redstone contraptions are active. [^463^] [^581^] However, parallel instances at normal 20 TPS each provide more stable rollouts and better core utilization than a single tick-warped server; on an 8-core CPU, running 4–6 Fabric servers at 20 TPS is the empirically superior strategy for RL training. [^462^]

Carpet Mod also bundles **Scarpet**, a full in-game programming language inspired by Clojure, for server-side automation. [^43^] Scarpet can handle events (`__on_player_joins`, `__on_tick`, `__on_entity_dies`), manipulate blocks and inventories, register custom commands with typed arguments, and integrate with scoreboards. Scripts placed in `world/scripts/` autoload on startup. Scarpet is **not** Python, but it can execute server commands that trigger external processes or write data to files that a Python process polls. For a Python-centric architecture, Carpet Mod should be paired with a separate bridge mod rather than used as the primary integration layer.

#### 2.2.2 Lithium, FerriteCore, Krypton: Optimization Mods for Parallel Instances

When running multiple server instances for parallel RL rollouts, vanilla Minecraft's resource consumption becomes prohibitive. Three optimization mods address this:

| Mod | Target | Performance Gain |
|---|---|---|
| Lithium | Game logic (mob AI, collisions, chunk ticking) | 30–50% faster tick times [^518^] |
| FerriteCore | RAM usage via optimized block-state storage | 40–50% less memory [^518^] |
| Krypton | Networking stack CPU consumption | Up to 40% less network CPU [^518^] |

Combined, Lithium + FerriteCore + Krypton allow a Fabric server with a village-sized loaded area to operate at 2–3 GB RAM and 15–25% of a single CPU core at 20 TPS, down from 3–4 GB and 30–50% on vanilla. [^518^] [^462^] For a 64 GB workstation, this translates to 4–6 parallel instances comfortably, leaving 40+ GB for the Python training pipeline. Additional mods worth considering are **Alternate Current** (up to 95% faster redstone) [^518^], **C2ME** (parallel chunk loading, experimental) [^513^], and **LazyDFU** (faster server startup) [^518^]. Fabric Loader itself has minimal overhead compared to Forge — approximately 1.2 GB base RAM versus 1.8 GB at idle, with 50–60% faster startup times. [^590^]

---

### 2.3 Modding Toolchains and APIs

Understanding the available toolchains for deobfuscation, mapping, and networking is necessary background for any developer who will write a custom bridge mod.

#### 2.3.1 MCP-Reborn: Research-Only Deobfuscation Toolchain

MCP-Reborn is a Mod Coder Pack that deobfuscates and decompiles Minecraft's source code into a Gradle-based project with `setup` and `runClient` tasks. [^114^] An auto-updated fork, `dubfib/mcp-reimagined`, tracks versions 1.13 through 1.21.11 via GitHub Actions. [^214^]

The critical limitation is Mojang's copyright: **"You CANNOT publish any code generated by this tool."** [^114^] MCP-Reborn produces a standalone client JAR with no mod compatibility, and it carries 350+ open issues documenting build failures on newer versions. [^211^] JDK requirements also escalate aggressively: Minecraft 1.17 requires JDK 16, 1.18+ requires JDK 17, and 1.21+ requires JDK 21. [^114^]

**Recommendation:** Skip MCP-Reborn for a production bot project. Use Fabric API + Yarn mappings instead. MCP-Reborn is useful only for reading obfuscated vanilla code to understand internal Minecraft behavior when writing a custom mod.

#### 2.3.2 Fabric API Networking Module: CustomPayload API for Bridge Mod Development

Fabric API provides the `CustomPayload` networking module, which is the standard transport layer for any mod that needs to send data between client and server. [^145^] [^185^] Since Fabric API 0.77.0 (Minecraft 1.20.5+), the packet API uses record-based type-safe definitions with automatic codec serialization:

```java
public record BotCommandPayload(String command, int botId) implements CustomPayload {
    public static final Id<BotCommandPayload> ID =
        new Id<>(Identifier.of("aivillage", "bot_command"));
    public static final PacketCodec<RegistryByteBuf, BotCommandPayload> CODEC =
        PacketCodec.tuple(
            PacketCodecs.STRING, BotCommandPayload::command,
            PacketCodecs.INTEGER, BotCommandPayload::botId,
            BotCommandPayload::new);

    @Override public Id<? extends CustomPayload> getId() { return ID; }
}
```

Registration on the server and client uses `PayloadTypeRegistry.playS2C()` and `ClientPlayNetworking.registerGlobalReceiver()`, respectively, with thread-safe callbacks that execute on the main server or client thread by default. [^145^] [^149^] Bidirectional communication is fully supported, and `PlayerLookup` helpers simplify sending packets to all players tracking a given chunk or entity.

For a custom bridge mod, the Fabric API networking module serves as the **transport layer** beneath a higher-level protocol such as Py4J or WebSocket. The mod would define custom payload types for commands (e.g., `BotCommandPayload`, `WorldStatePayload`), handle incoming packets from a Python gateway, execute actions in Minecraft, and return results via outgoing packets. This is the correct architectural boundary: Fabric networking for intra-Minecraft communication, Py4J or WebSocket for Minecraft-to-Python communication.

---

### 2.4 Python Bridge Landscape

The absence of a production-ready, version-current Python bridge mod is the single largest gap in the Fabric ecosystem. This section documents the landscape, explains why gRPC is not the pragmatic choice, and establishes Py4J as the recommended bridge technology.

#### 2.4.1 Why No Production gRPC Mod Exists for Current Fabric

The `fabric-grpc-api` project on Modrinth provides a shade-packed gRPC-Java library as a Fabric mod dependency, licensed under Apache-2.0. [^51^] However, it targets **Minecraft 1.20.1 only** and has not been updated in approximately three years. For Minecraft 1.21.x, which introduced breaking changes in the networking API (the `CustomPayload` record-based system described in Section 2.3.2), this mod is non-functional without significant migration work.

No alternative gRPC bridge mod was found in the ecosystem survey. [^51^] [^101^] Building a production gRPC server inside a Fabric mod would require: (1) shading gRPC-Java and its Netty dependency; (2) adapting to 1.21.x's `CustomPayload` API for any packets that must traverse the Minecraft client-server boundary; and (3) managing protobuf definitions for the bot API surface. This is weeks of engineering for a solo developer, with no reference implementation to copy from.

#### 2.4.2 Py4J as the Pragmatic Choice: UnionClef's Implementation as Reference Architecture

Py4J is a lightweight Java library that enables Python programs to dynamically access Java objects over a local socket. It is the same technology UnionClef uses for its two-way bridge, and it has been production-tested with multi-instance support. [^160^] [^180^]

The architecture is straightforward. On the Java side, a `GatewayServer` exposes an entry-point object whose public methods become callable from Python:

```java
GatewayServer server = new GatewayServer(new MinecraftEntryPoint());
server.start();
```

On the Python side, `py4j.java_gateway.JavaGateway` connects to the server and provides direct access to the entry-point object:

```python
from py4j.java_gateway import JavaGateway
gateway = JavaGateway()
mc = gateway.entry_point
mc.sendChatMessage("Hello from Python!")
```

This is strictly more ergonomic than gRPC protobuf message passing for a single-machine deployment: Python calls Java methods directly, with type marshaling handled automatically. gRPC's advantages — language-agnostic service definitions, HTTP/2 multiplexing, service mesh integration — do not apply when the Python orchestrator and the Fabric client mod are running on the same workstation with local sockets. [^180^]

UnionClef's `Py4JEntryPoint` class (package `adris.altoclef`) should be studied as the reference implementation. It demonstrates how to: initialize the gateway on client connection; expose position, health, inventory, and entity data; send AltoClef commands from Python; and handle multi-instance port allocation. [^160^]

#### 2.4.3 Custom WebSocket Bridge as Alternative: When and Why

A WebSocket bridge is worth considering under two conditions. First, if the Python orchestrator and the Minecraft client must run on separate machines (e.g., a cloud GPU training node controlling a local Minecraft farm), WebSocket's TCP-based transport works across networks whereas Py4J's local socket does not. Second, if the observation pipeline requires streaming large volumes of world-state data to multiple consumers simultaneously, WebSocket's pub-sub semantics are a better fit than Py4J's one-to-one gateway model.

The `httpInfoServer-mod` by dadencukillia provides a minimal reference architecture: a Fabric client mod with `WebSocketDoor.java` (WebSocket client), `InfoCollector.java` (JSON data generation), and `HttpInfoServer.java` (entry point). [^101^] The mod connects via WebSocket to an external HTTP API server, sending game data outbound and receiving `collectData` commands inbound. However, this is a single-author project with limited testing, and it should be treated as a design reference rather than a dependency. [^101^]

| Bridge Technology | Effort | Performance | Maturity | Multi-Machine | Recommended When |
|---|---|---|---|---|---|
| **Py4J (custom)** | Low | High (local socket) | Proven (UnionClef) | No | Single-machine deployment |
| **WebSocket + JSON** | Medium | Medium | Reference available | Yes | Remote orchestrator or pub-sub |
| **gRPC (custom)** | High | High | No 1.21 mod exists | Yes | Existing protobuf infrastructure |
| **Fabric API packets** | Medium | High | Native | No | Server-side companion mod only |

For the typical solo developer running 4–6 Fabric client instances on a single workstation, **Py4J is the clear recommendation**. The effort is low, the performance is high (local Unix sockets have negligible overhead), and UnionClef provides a working reference implementation that can be adapted or forked.

---

### 2.5 Recommended Architecture and Top 5 Mods to Study

#### 2.5.1 Architecture Decision: Client-Side Mod vs. Server Plugin vs. Protocol-Level Bot

Three architectural patterns are available for connecting Python AI agents to Minecraft:

**Client-side Fabric mod.** A Fabric mod (based on UnionClef/Baritone) runs inside a full Minecraft client, communicates with a Python orchestrator via Py4J, and connects to a server as a normal player. This gives full access to chunk data, entity positions, inventory GUI state, Baritone pathfinding, and screenshot capture. It is the recommended pattern for a single-machine RL training setup because it provides the richest observation space and the most reliable action execution. [^180^] [^150^]

**Server plugin / mod.** A server-side-only Fabric mod can expose world state and accept commands, but it lacks access to client-side rendering, detailed block interaction states, and Baritone's pathfinding engine. Server-side automation is better handled by Scarpet (Carpet Mod) or by a protocol-level bot. This pattern is appropriate if Bedrock compatibility (via Geyser) is a hard requirement, since Geyser emulates a vanilla Java client and cannot tolerate client-side mod modifications.

**Protocol-level bot (MineFlayer).** A JavaScript bot connects directly to the Minecraft server protocol without running a full client. This is the lightest-weight option and the best fit for multi-server proxy architectures, but it lacks the deep world rendering and pathfinding quality of Baritone. [^49^]

The recommended architecture for this project is **client-side Fabric mod + Python orchestrator**, with one mod instance per bot (each on a distinct Py4J port), all connecting to a local Fabric server running Carpet Mod, Lithium, and FerriteCore. This pattern is used by the most advanced active projects in the space, including UnionClef, and it directly enables the multi-agent village simulation described in subsequent chapters.

#### 2.5.2 Top 5 Mods to Read Source Code of Before Writing Your Own

The following table ranks the five most instructive codebases for a developer preparing to write a custom Fabric bridge mod. The ranking balances relevance to the project's goals, code quality, and the educational value of each codebase's design patterns.

| Rank | Repository | License | Lines of Code (est.) | What to Study | Why It Matters |
|---|---|---|---|---|---|
| 1 | **UnionClef** (`3ndetz/unionclef`) [^180^] | GPL-3.0 | ~45k | `Py4JEntryPoint.java`, task system, command handling, Baritone wrapping | Single most relevant codebase — demonstrates the complete Py4J bridge pattern, multi-instance support, and how to build a high-level task system on Baritone. [^160^] |
| 2 | **Baritone** (`cabaletta/baritone`) [^150^] | LGPL-3.0 | ~80k | `baritone.api` package, `IBaritone` interface, `CustomGoalProcess`, settings system | Gold-standard pathfinding API design. The `IBaritone` abstraction and goal-based movement system are exemplary mod architecture. [^200^] |
| 3 | **httpInfoServer-mod** (`dadencukillia/httpInfoServer-mod`) [^101^] | Unknown | ~2k | `WebSocketDoor.java`, `InfoCollector.java`, `HttpInfoServer.java` | Minimal WebSocket bridge pattern — the shortest path to understanding how to stream Minecraft state to an external HTTP/WebSocket server. [^101^] |
| 4 | **Carpet Mod** (`gnembon/fabric-carpet`) [^111^] | Open-source | ~120k | Scarpet event registration, `/tick` command implementation, command hook system | Demonstrates how to register server-side events, manipulate game ticks, and build a scripting language inside a Fabric mod. [^43^] |
| 5 | **PythonMC** (`modrinth.com/mod/pythonmcmod`) [^155^] | Unknown | ~3k | Python subprocess management, `pythonmc_api` module, event hook system | Shows how to embed a Python runtime inside a server-side Fabric mod, with `on_server_started`, `on_player_join`, and `on_tick` callbacks. [^155^] |

**Analytical interpretation of the ranking.** UnionClef holds the top position because it is the only active project that combines all three elements this project requires: Fabric mod integration, Baritone pathfinding, and a working Python bridge. Reading `Py4JEntryPoint.java` and the `scripts/` folder in sequence reveals the complete data flow from Python command issuance to Java execution to result callback. [^160^] Baritone ranks second not because it is harder to use, but because its API is already well-documented; the primary reason to read the source is to understand the `CustomGoalProcess` lifecycle and the caching strategy for chunk-aware pathfinding, both of which are essential when tuning agent movement for RL reward shaping. The `httpInfoServer-mod` at rank three punches above its weight because its tiny codebase (approximately 2,000 lines) isolates the WebSocket bridge pattern without the distracting complexity of a full task framework — it is the fastest way to validate whether a WebSocket transport meets latency requirements. Carpet Mod at rank four is essential for understanding server-side tick control and Scarpet's event model, which becomes relevant when the training pipeline needs to freeze or step the world deterministically. PythonMC at rank five rounds out the list by demonstrating an alternative embedding strategy: rather than exposing Java to Python (Py4J's model), it runs Python scripts from within the mod via a subprocess, which is less performant but architecturally simpler and worth understanding as a fallback option.

The combined reading list represents roughly 250,000 lines of Java code. A focused read — restricting attention to the specific classes identified in column 4 — can be completed in approximately 3–4 days and will save weeks of trial-and-error when writing a custom bridge mod. The recommended sequence is: start with UnionClef's `scripts/` folder to understand the Python side of the bridge, then read Baritone's `api/` package to understand the underlying pathfinding primitives, then study `httpInfoServer-mod` if WebSocket is a candidate transport, and finally read Carpet Mod's `/tick` command implementation to understand server-side temporal control.

