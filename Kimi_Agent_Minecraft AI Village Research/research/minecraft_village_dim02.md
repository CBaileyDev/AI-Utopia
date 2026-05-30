# Fabric Mod Ecosystem for Bot/AI Integration - Research Report

**Date:** 2025-06-30
**Scope:** Minecraft Java Edition, Fabric Loader, AI/Bot integration, programmatic control
**Minecraft Versions:** 1.20.1 - 1.21.8 (current stable)

---

## Executive Summary

- **UnionClef** (fork of Altoclef) is the most advanced existing solution for Python-integrated Minecraft botting on Fabric — it ships with a built-in Py4J two-way bridge, multiplayer/anticheat support, and active development as of April 2026. [^180^]
- **Baritone** remains the gold standard for pathfinding, with active maintenance through MC 1.21.8, a stable `baritone.api` Java API, and LGPL-3.0 licensing that permits library use in custom clients. [^150^] [^200^]
- **MineFlayer** (JavaScript/Node.js, MIT license) is the best protocol-level option if you want to avoid modding entirely — it supports 1.8-1.21.11, has Python bindings via `javascript`/`JSPyBridge`, and powers research projects like Voyager and mindcraft. [^49^] [^199^]
- **Carpet Mod + Scarpet** provides a powerful server-side scripting environment for automation, but it is NOT a Python bridge — it uses its own Scarpet language. For Python integration, combine it with a separate bridge mod. [^111^] [^43^]
- **No production-ready gRPC mod exists for current Minecraft versions** — `fabric-grpc-api` on Modrinth targets only 1.20.1 and has not been updated in ~3 years. For a multi-agent system, a custom WebSocket or Py4J bridge is the pragmatic path. [^51^]
- **Recommended architecture for a solo developer:** Client-side Fabric mod (using UnionClef/Baritone) + Py4J bridge to a Python orchestrator process. This gives full world state access, pathfinding, inventory control, and clean multi-agent coordination without protocol-level reimplementation.

---

## 1. Baritone - Pathfinding Bot

### Overview
Baritone is the de facto standard Minecraft pathfinding bot library. Originally developed by leijurv for the Impact client, it is now a standalone library used across dozens of projects. [^150^]

### Maintenance Status: **ACTIVELY MAINTAINED** 
- Last release: **v1.15.0** (August 26, 2025) supporting MC 1.21.6, 1.21.7, 1.21.8 [^200^]
- Fabric, Forge, and NeoForge all supported
- Previous release v1.14.0 (May 13, 2025) for 1.21.5; v1.13.1 for 1.21.4
- Continuous community contributions; 6,500+ GitHub stars across ecosystem

### Version Compatibility
| Minecraft | Baritone Version | Status |
|-----------|-----------------|--------|
| 1.21.8 | v1.15.0 | Current |
| 1.21.6-1.21.7 | v1.15.0 | Current |
| 1.21.5 | v1.14.0 | Supported |
| 1.21.4 | v1.13.1 | Supported |
| 1.21.3 | v1.11.1 | Supported |
| 1.21.1 | v1.11.2 | Supported |
| 1.20.4 | v1.10.4 | Legacy |
| 1.18.2 | v1.8.6 | Legacy |

### What It Exposes
- **Pathfinding:** A* pathfinding with chunk caching, 30x faster than original MineBot
- **World state:** Block scanning, chunk awareness, cached chunk rendering
- **Movement:** Sprint, parkour, elytra flying, water pathing (limited)
- **Block interaction:** Mining (`#mine`), placing, building (`#build`)
- **Goals:** `GoalXZ`, `GoalBlock`, `GoalNear`, `GoalGetToBlock`, custom goals
- **Settings:** ~100 configurable settings (`allowBreak`, `allowPlace`, `allowSprint`, `allowParkour`, `legitMine`, etc.)
- **API Access:** All stable API lives in `baritone.api` package only [^150^]

### API Example (Java)
```java
BaritoneAPI.getSettings().allowSprint.value = true;
BaritoneAPI.getSettings().primaryTimeoutMS.value = 2000L;
BaritoneAPI.getProvider().getPrimaryBaritone().getCustomGoalProcess()
    .setGoalAndPath(new GoalXZ(10000, 20000));
```

### Python Bridge?
**No direct Python API.** Baritone is Java-only. However, it can be controlled via:
1. **Py4J** (as UnionClef demonstrates — calling Baritone through a Java gateway)
2. **Chat commands** (e.g., `#goto 1000 500`) — but this is brittle for programmatic use
3. **gRPC/WebSocket bridge** (would require a custom Fabric mod wrapping BaritoneAPI calls)

### License
**LGPL-3.0** — Can be used as a library in custom utility clients. Cannot publish modified Baritone code under a more restrictive license. [^50^]

---

## 2. Altoclef / UnionClef - High-Level Task Bot

### Overview
Altoclef is a high-level task bot built on Baritone that can accomplish complex Minecraft tasks autonomously — it was the first bot to beat Minecraft fully autonomously (May 24, 2021). The original repo by gaucho-matrero was archived in June 2024. [^47^]

### The Active Fork: UnionClef (by 3ndetz)
**This is the most important finding for Python-integrated bot development.**

UnionClef is a monorepo unifying Altoclef + Baritone ("shredder" fork) + Tungsten pathfinder, with **built-in Py4J Python integration**. [^180^]

### Maintenance Status: **VERY ACTIVELY MAINTAINED**
- Latest release: **v0.21.1** for MC 1.21.1 (April 3, 2026) [^180^]
- Last commit: March 31, 2026
- 420+ commits, 4 contributors
- Monorepo with `scripts/` folder for Python scripting via Py4J

### Version Compatibility
- **Minecraft 1.21.1** (Fabric only) — primary target
- Multi-version support structure planned

### What It Exposes (Beyond Baritone)
- **Task system:** `@get`, `@craft`, `@mine`, `@gamer`, `@deposit`, `@stash` commands
- **Inventory management:** Automatic crafting via Recipe Book, chest storage
- **Survival gameplay:** Full Minecraft walkthrough (beat the game autonomously)
- **PvP:** Ender pearl clutches, arrow dodging, shield usage, mace combat
- **Minigames:** SkyWars, BedWars, SkyPvP, MurderMystery (various completeness)
- **Building:** `@grave`, `@sign`, schematic support (planned)
- **Multiplayer:** Auto-login, auto-register, anti-cheat bypass rotation
- **Python Integration:** Py4J two-way gateway with configurable port [^180^]

### Python Bridge (Py4J) — Critical Detail
UnionClef implements a **two-way Py4J interface:** [^160^] [^180^]

```
Port configurable via: @set pythonGatewayPort <port>
Multi-instance launching supported
```

**From Python:**
- Send commands to Java side (AltoClef commands, chat messages)
- Receive live data from Java side (position, health, inventory, nearby entities/blocks)
- Live screenshot support
- Rich contextual data via `adris.altoclef.Py4JEntryPoint` class

**From Java:**
- Invoke Python callbacks for decision-making
- Pass game state to Python process
- Bridge supports multi-instance (multiple bots, different ports)

This is the **closest existing solution to a production Python bridge for Fabric botting.**

### License
**GPL-3.0** — Incorporates code from Baritone (LGPL-3.0), Altoclef (MIT), Tungsten (CC0-1.0). [^180^]

### Key Caveat
The author explicitly warns the code is "mostly for prototyping" and their Java experience was limited when starting. However, the project has matured significantly with 420+ commits and active CI/CD. [^160^]

---

## 3. MineFlayer - Protocol-Level JavaScript Bot

### Overview
MineFlayer is a high-level JavaScript API for creating Minecraft bots. It operates at the protocol level (via node-minecraft-protocol) — no client modding required. It is the foundation for nearly all modern Minecraft AI research (Voyager, mindcraft, VillagerAgent, etc.). [^49^]

### Maintenance Status: **VERY ACTIVELY MAINTAINED**
- 6,500+ GitHub stars, 1,200+ forks
- Supports Minecraft 1.8 through **1.21.11** (continuously updated) [^49^]
- Python support via `javascript` package (JSPyBridge) or Google Colab examples
- Last package.json update: October 2025 [^199^]
- Active test suite updated August 2025

### What It Exposes
- **Entity tracking:** Full entity knowledge and tracking
- **Block queries:** Millisecond block lookups in the world
- **Physics & movement:** All bounding boxes, realistic physics simulation
- **Combat:** Attacking, using vehicles, PvP/PvE
- **Inventory management:** Full inventory, crafting, chests, dispensers, enchantment tables
- **World interaction:** Digging, building, activating blocks, using items
- **Chat:** Send/receive chat, parse chat messages
- **Plugins:** pathfinder, prismarine-viewer, statemachine, PVP, AutoEat, CollectBlock, etc. [^49^]

### Python Usage
MineFlayer can be used from Python via the `javascript` package:

```python
from javascript import require, On, once

mineflayer = require('mineflayer')
bot = mineflayer.createBot({
    'host': 'localhost',
    'username': 'PythonBot',
    'auth': 'offline'
})

@On(bot, 'chat')
def on_chat(this, username, message):
    if username == bot.username:
        return
    bot.chat(f"Echo: {message}")
```

Python examples are in `examples/python/` on GitHub and on Google Colab. [^49^] [^204^]

### License
**MIT** — Extremely permissive. Ideal for research and commercial use.

### When to Choose MineFlayer vs. Fabric Mod
- **Choose MineFlayer** when: You need quick prototyping, protocol-level control is sufficient, you're doing AI research, you need to avoid client modding entirely, or you're building a proxy/multi-server system.
- **Choose Fabric mod** when: You need full client-side world rendering, anti-cheat bypasses, complex building, visual feedback/screenshots, or Baritone-level pathfinding.

---

## 4. Carpet Mod / Fabric Carpet

### Overview
Carpet Mod is a server-side Fabric mod that gives full control over vanilla Minecraft's technical aspects. It is the standard tool for technical Minecraft servers and is particularly powerful when combined with its built-in scripting language, Scarpet. [^111^]

### Maintenance Status: **ACTIVELY MAINTAINED**
- Author: gnembon (core Fabric team contributor)
- Supports Minecraft 1.14.x through **1.21.11** (continuous releases) [^112^]
- Latest release tracked on GitHub releases page [^115^]
- Ecosystem includes carpet-extra, scarpet app store, and dozens of community extensions

### What It Exposes
- **Tick warp:** `/tick warp` — accelerate time for farm testing
- **Performance monitoring:** `/log` for live mobcap, TPS, entity tracking
- **Mechanics changes:** Movable block entities, customizable spawning rules
- **Hopper counters:** Item flow analysis
- **Scarpet scripting:** Full in-game programming language (see section 8)
- **Server-side only:** No client installation required for most features

### Version Compatibility
Extremely broad — versions available for 1.14.4, 1.15.x, 1.16.x, 1.17.x, 1.18.x, 1.19.x, 1.20.x, 1.21.x (up to 1.21.11). [^112^]

### Python Bridge?
**No.** Carpet Mod does not provide a Python bridge. It uses Scarpet, its own scripting language. However:
- Scarpet apps can load/unload dynamically via `/script` commands
- A Python process could theoretically communicate via RCON or custom plugin messages
- For Python integration, pair Carpet Mod with a separate bridge mod (see sections 9-10)

### License
Not explicitly stated in README, but Scarpet apps and extensions are open-source. Carpet mod itself follows standard Fabric community practices.

---

## 5. MCP-Reborn / MCPFabric

### Overview
MCP-Reborn is a Mod Coder Pack for Minecraft that deobfuscates and decompiles Minecraft's source code for research and client modding. It is NOT a mod — it is a development toolchain. [^114^]

### Maintenance Status: **MAINTAINED BUT COMMUNITY-DRIVEN**
- Original: Hexeption/MCP-Reborn on GitHub
- Auto-updated fork: dubfib/mcp-reimagined (GitHub Actions automated updates) [^214^]
- Supports versions 1.13 through **1.21.11** (README says 1.21.4, but latest PRs go to 1.21.11) [^114^]
- Many open issues (350+) suggest it can be fragile for newer versions

### What It Provides
- Decompiled, deobfuscated Minecraft source code in `src/` folder
- Gradle-based build system with `setup` and `runClient` tasks
- Supports modifying core game code and building custom JARs
- Based on MCPConfig and ForgeGradle

### Critical Limitations
- **"You CANNOT publish any code generated by this tool"** — Mojang's copyright applies
- No mod compatibility — this is for standalone client research only
- JDK requirements escalate: 1.17< needs JDK 16, 1.18> needs JDK 17, 1.21> needs JDK 21 [^114^]
- Many open issues about build failures on newer versions [^211^]

### Python Bridge?
**No.** This is a research/development toolchain, not a mod. Not suitable for a production multi-agent system directly, but useful for understanding Minecraft internals when writing your own mod.

### Recommendation
**Skip MCP-Reborn** for a production bot project. Use Fabric API + Yarn mappings instead. MCP-Reborn is only useful if you need to understand obfuscated vanilla code for advanced modding.

---

## 6. Python Bridge Mods for Fabric

### The State of Python Bridges (2025)
**There is no mature, widely-adopted Python bridge mod for Fabric.** However, several options exist at different maturity levels:

#### Option A: UnionClef Py4J Bridge (MOST MATURE)
- **Built into:** UnionClef mod (see section 2)
- **Technology:** Py4J (Python-Java gateway over socket)
- **Direction:** Two-way — Python can call Java methods, Java can call Python callbacks
- **Port:** Configurable via `@set pythonGatewayPort <port>`
- **Maturity:** Production-tested with multi-instance support
- **Use case:** Best if you're already using UnionClef/AltoClef task system [^180^] [^160^]

#### Option B: PythonMC
- **Type:** Server-side Fabric mod
- **Minecraft:** 1.21.1
- **Approach:** Runs Python 3.8+ scripts from `config/pythonmc/scripts/`
- **API:** `pythonmc_api` module with `mc.log()`, `mc.tell()`, `mc.broadcast()`, command execution
- **Events:** `on_server_started`, `on_player_join`, `on_tick`, etc.
- **Reload:** `/pythonmc reload` for hot-reloading scripts
- **License:** Unknown (modrinth page) [^155^]

#### Option C: fabricpy (Code Generator)
- **Type:** Python library (`pip install fabricpy`)
- **Approach:** Write mod logic in Python → generates full Fabric mod project with Java source
- **Features:** Items, blocks, tools, food, recipes, loot tables, creative tabs
- **NOT a runtime bridge** — it's a build-time code generator
- **Use case:** If you want to write Fabric mods in Python syntax but compile to Java [^148^]

#### Option D: Py4J Custom Bridge (RECOMMENDED FOR CUSTOM ARCHITECTURE)
Py4J is the lightweight Java library that enables Python programs to dynamically access Java objects. It is the same technology UnionClef uses. For a custom multi-agent system:

```java
// In your Fabric mod
GatewayServer server = new GatewayServer(new MinecraftEntryPoint());
server.start();
```

```python
# In Python
from py4j.java_gateway import JavaGateway
gateway = JavaGateway()
mc = gateway.entry_point
mc.sendChatMessage("Hello from Python!")
```

### Recommendation
For a **production multi-agent system**, use **Option D (custom Py4J bridge)** or **fork UnionClef's Py4J setup.** This gives you full control over the API surface while leveraging proven technology.

---

## 7. Minecraft-Console-Client (MCC)

### Overview
MCC is a lightweight, cross-platform TUI client for Minecraft Java Edition written in C#. It enables connecting to servers, sending commands, and running automation scripts without the full Minecraft client. [^103^]

### Maintenance Status: **ACTIVE BUT LAGGING ON VERSIONS**
- MCCTeam organization on GitHub
- MIT License [^102^]
- Supports versions 1.4.6 through **1.20.4** officially [^113^]
- **Limited 1.21 support** — community reports it works but is limited [^209^]
- Issue #2738 (June 2024) requesting 1.21 support is still open [^210^]

### What It Exposes
- **Chat automation:** Send/receive messages, chat bots
- **Scripting:** C# scripts with full ChatBot API
- **Auto-reconnect, auto-respawn, auto-AFK**
- **Scripting API:** `MCC.PerformInternalCommand()`, `MCC.SendText()`, inventory handling
- **Cross-platform:** Linux, macOS, Windows, even Android

### Python Bridge?
**No.** MCC is C#-based and does not have a Python API. The scripting language is C#.

### Recommendation
**Not recommended for a Fabric-based multi-agent system.** MCC is useful for chat bots and simple automation, but it lacks the world-interaction depth of Baritone/MineFlayer and the Python ecosystem you need. Consider it only if you need ultra-lightweight chat-only agents.

---

## 8. Scarpet (Carpet Script)

### Overview
Scarpet is a powerful in-game programming language built into Carpet Mod. It enables writing `.sc` script files that can interact with the Minecraft world, define commands, respond to events, and automate server tasks. [^43^]

### What It Can Do
- **Event handling:** `__on_player_joins`, `__on_tick`, `__on_entity_dies`, etc.
- **World manipulation:** Place/break blocks, query entities, read NBT data
- **Custom commands:** Register `/` commands with typed arguments
- **Inventory management:** Item manipulation, crafting, container queries
- **Scoreboard integration:** Full scoreboard read/write
- **Autoloading:** Scripts in `world/scripts/` auto-load on startup
- **App scoping:** Per-player or global scope

### Example Scarpet Script
```scarpet
// Save as world/scripts/hello.sc
__config() -> {'scope' -> 'global'};

__on_player_joins(player) -> (
    print(player, 'Welcome! This is a Scarpet script.');
);

count_blocks(pos, radius) -> (
    scan(pos, radius, block(_x, _y, _z))
);
```

### Python Bridge?
**No.** Scarpet is its own language (inspired by Clojure). It is NOT Python. However, Scarpet scripts can:
- Execute server commands that trigger external processes
- Write data to files that a Python process polls
- Be paired with a WebSocket/gRPC bridge mod for external communication

### Documentation
Full API docs at: `github.com/gnembon/fabric-carpet/blob/master/docs/scarpet/api/Overview.md` [^43^]
Public Scarpet apps: `github.com/gnembon/scarpet` [^202^]

---

## 9. Fabric API Networking Module

### Overview
Fabric API provides a modern networking module for custom packet communication between client and server. This is the foundation for building any RPC bridge mod. [^145^] [^185^]

### Current API (1.20.5+ / Networking API v1)
Fabric API 0.77.0+ introduced a packet object-based API that is thread-safe and type-safe: [^149^] [^145^]

**Server → Client:**
```java
// Define payload
public record LightningPayload(BlockPos pos) implements CustomPayload {
    public static final Id<LightningPayload> ID = 
        new Id<>(Identifier.of("mymod", "lightning"));
    public static final PacketCodec<RegistryByteBuf, LightningPayload> CODEC = 
        PacketCodec.tuple(BlockPos.PACKET_CODEC, LightningPayload::pos, LightningPayload::new);
    
    @Override public Id<? extends CustomPayload> getId() { return ID; }
}

// Register
PayloadTypeRegistry.playS2C().register(LightningPayload.ID, LightningPayload.CODEC);

// Send
ServerPlayNetworking.send(player, new LightningPayload(player.getBlockPos()));
```

**Client → Server:**
```java
ClientPlayNetworking.registerGlobalReceiver(LightningPayload.ID, (payload, context) -> {
    context.client().execute(() -> {
        // Handle on client thread
    });
});
```

### Key Capabilities
- **CustomPayload records** — type-safe packet definitions
- **Automatic codec serialization** — no manual `PacketByteBuf` writing
- **Thread-safe callbacks** — run on main server/client thread by default
- **PlayerLookup helpers** — send to all players tracking an entity/chunk
- **Bidirectional** — C2S and S2C both fully supported [^145^]

### When to Use This
You would use Fabric API networking as the **transport layer** for a custom bridge mod. Your mod would:
1. Define custom payload types for commands (e.g., `BotCommandPayload`, `WorldStatePayload`)
2. Handle incoming packets from a Python WebSocket/gRPC gateway
3. Execute actions in Minecraft and send results back via outgoing packets

---

## 10. gRPC / WebSocket Mods for Fabric

### The Landscape (2025)
The ecosystem for RPC bridge mods is **immature** — only a handful of projects exist, most targeting older Minecraft versions.

#### A. fabric-grpc-api (Modrinth)
- **What:** Shade-packed gRPC-Java library as a Fabric mod dependency
- **Minecraft:** 1.20.1 only
- **Last updated:** ~3 years ago (stale)
- **License:** Apache-2.0
- **Use:** Add as Gradle dependency to implement gRPC server/client in your mod [^51^]
- **Verdict:** **STALE** — would need updating for 1.21.x

#### B. httpInfoServer-mod (GitHub: dadencukillia)
- **What:** Fabric mod that connects via WebSocket to an external HTTP API server
- **Direction:** Client mod sends game data to external server; server sends `collectData` commands
- **Components:** `HttpInfoServer.java` (entry), `WebSocketDoor.java` (WebSocket client), `InfoCollector.java` (JSON data generation)
- **License:** Unknown
- **Verdict:** Interesting reference architecture for a WebSocket bridge, but single-author and not widely tested [^101^]

#### C. MC Remote Control (Modrinth: Pn9wqc8u)
- **What:** Discord-based remote control for Minecraft
- **Minecraft:** 1.21.4, 1.21.10
- **Approach:** Fabric client mod hosts a Discord bot; control via Discord `/` commands
- **Features:** In-game bot control through Discord messages
- **License:** Unknown
- **Verdict:** Different architecture (Discord bridge, not Python), but shows the pattern of external control [^182^]

#### D. ws2tcp / MISS (WebSocket Protocol Forwarding)
- **ws2tcp:** Fabric client mod to connect to MC servers via WebSocket proxy (1.21.4) [^106^]
- **MISS (LiterMC):** Mod that forwards Minecraft connections over WebSocket (1.19+) [^118^]
- **Verdict:** These are protocol proxy tools, not API bridges. Useful for cloud hosting but not for bot control.

### Recommendation
**For a production multi-agent system, build a custom bridge.** Options ranked:

| Approach | Effort | Performance | Maturity |
|----------|--------|-------------|----------|
| **Py4J (custom)** | Low | High (local socket) | Proven (UnionClef) |
| **WebSocket + JSON** | Medium | Medium | Reference available |
| **gRPC (custom)** | High | High | No current 1.21 mod |
| **Fabric API packets only** | Medium | High | Native but Minecraft-specific |

---

## Key Research Findings

### Is there an existing Fabric mod that provides a Python bridge?
**Yes, but with caveats:**
- **UnionClef** has the most mature Py4J Python bridge, integrated into a full task bot framework. [^180^]
- **PythonMC** provides server-side Python scripting but is limited in API surface. [^155^]
- **fabricpy** generates Java code from Python but is not a runtime bridge. [^148^]

**Best path:** Fork UnionClef's Py4J integration or build your own Py4J bridge using Fabric API networking.

### Which 3-5 mods should the user read the source code of before writing their own?

1. **UnionClef** (`github.com/3ndetz/unionclef`) — Study `Py4JEntryPoint.java`, task system, command handling, and how it wraps Baritone. This is the single most relevant codebase. [^180^]

2. **Baritone** (`github.com/cabaletta/baritone`) — Study `baritone.api` package, `IBaritone` interface, `CustomGoalProcess`, and settings system. The API design is clean and well-documented. [^150^]

3. **httpInfoServer-mod** (`github.com/dadencukillia/httpInfoServer-mod`) — Study `WebSocketDoor.java` and `InfoCollector.java` for a minimal WebSocket bridge pattern. [^101^]

4. **Carpet Mod** (`github.com/gnembon/fabric-carpet`) — Study Scarpet event system and how it registers commands/hooks for the server-side scripting approach. [^111^]

5. **PythonMC** (`modrinth.com/mod/pythonmcmod`) — Study how it manages a Python subprocess and exposes the API for server-side scripting. [^155^]

### What's the recommended architecture?

For a **solo developer building a production-quality multi-agent system:**

**Architecture: Client-Side Fabric Mod + Python Orchestrator**

```
+------------------+        Py4J/WS        +-------------------------+
|  Python Agent    |<--------------------->|  Fabric Client Mod      |
|  Orchestrator    |    (localhost socket) |  (UnionClef-based)      |
|                  |                       |  - Baritone pathfinding |
|  - LLM reasoning |                       |  - Inventory/crafting   |
|  - Task planning |                       |  - World state access   |
|  - Multi-agent   |                       |  - Custom bridge        |
|    coordination  |                       +-------------------------+
+------------------+                                   |
       |                                                |
       v                                                v
+------------------+                          +------------------+
|  Minecraft Server| (if multiplayer)         |  Minecraft Client|
+------------------+                          +------------------+
```

**Why client-side mod, not server plugin or proxy?**
- **Full world state access:** Client has chunk data, entity positions, inventory GUI state
- **Baritone integration:** The best pathfinding only works client-side
- **Visual feedback:** Can capture screenshots for vision models (UnionClef supports this)
- **Anti-cheat compatibility:** Can use human-like movement (UnionClef's WindMouse rotations)
- **No server modifications needed:** Bots connect as normal clients

**Why not MineFlayer (proxy approach)?**
- MineFlayer is excellent for research/quick prototyping but lacks the deep world rendering and pathfinding of Baritone
- For multi-agent with complex task dependencies, the client mod approach gives richer state

### How should the gRPC/websocket bridge integrate with Fabric?

**Answer: As a client-side Fabric mod.**

Specifically:
1. Create a Fabric client mod that depends on `fabric-api` and `baritone` (or `unionclef`)
2. On `ClientPlayConnectionEvents.JOIN`, start a Py4J GatewayServer or WebSocket server
3. Expose a clean Java API class (e.g., `BotController`) that Python can call
4. Use `ClientPlayNetworking` if you need custom packets to a server-side companion mod
5. Run one mod instance per bot, each on a different Py4J/WebSocket port

**The bridge mod should NOT be a server plugin** because server-side code lacks access to client rendering, detailed block interaction states, and Baritone's pathfinding engine.

---

## Additional Notable Projects

### VillagerAgent (ACL 2024)
A graph-based multi-agent framework built on MineFlayer for complex task dependencies in Minecraft. Features a DAG task decomposer, agent controller, and state manager. Uses GPT-4 for task planning. MIT licensed. [^133^] [^183^]

### Voyager (ICLR 2024, MineDojo)
The landmark LLM-powered embodied agent using MineFlayer + GPT-4 code generation. Uses an automatic curriculum, skill library with vector DB, and iterative prompting. MIT licensed. [^197^] [^199^]

### mindcraft
A simpler but more practical LLM+MineFlayer agent framework. Supports multiple LLM backends (OpenAI, Google, local models), profile-based bot personalities, and multi-agent coordination. Active development. [^212^]

### Odyssey
Zhejiang University's open-world Minecraft agent with skill library, fine-tuned LLaMA models, and MineFlayer-based environment interaction. [^128^]

---

## Concrete Recommendations

### For Immediate Start (Week 1)
1. **Clone UnionClef** and get it running with MC 1.21.1 + Fabric [^180^]
2. **Study the `scripts/` folder** and Py4J interface to understand the Python bridge
3. **Run the basic Altoclef tasks** (`@get diamond_pickaxe`, `@gamer`) to understand capabilities

### For Custom Development (Month 1)
1. **Fork UnionClef** or create a new Fabric mod depending on Baritone API
2. **Implement a Py4J GatewayServer** in your mod's client initializer
3. **Design your Python API surface:** `get_world_state()`, `path_to(x,y,z)`, `craft(item)`, `mine(block)`
4. **Write the Python orchestrator** with asyncio for concurrent multi-agent management

### For Production Multi-Agent System
1. **One mod instance per bot**, each with its own Py4J port (e.g., 25333, 25334, ...)
2. **Central Python orchestrator** using `asyncio` + `py4j` for parallel agent control
3. **LLM integration** via function calling (OpenAI/Anthropic APIs) mapping to your bot API
4. **Shared state database** (Redis or SQLite) for agent coordination and memory
5. **Monitoring dashboard** using MineFlayer's `prismarine-viewer` or custom WebSocket feed

### Tech Stack Recommendation
| Layer | Technology | Reason |
|-------|-----------|--------|
| Minecraft client | Fabric 1.21.1 + UnionClef | Best Python bridge + pathfinding |
| Bridge | Py4J (custom gateway) | Proven, fast, two-way |
| Agent logic | Python 3.11+ + asyncio | Multi-agent concurrency |
| LLM integration | OpenAI/Anthropic APIs | Function calling for tool use |
| State/memory | Redis + SQLite | Fast shared state |
| Monitoring | Custom WebSocket feed | Real-time agent status |

---

## Open Questions

1. **Version targeting:** Should you target 1.21.1 (UnionClef's version) or 1.21.4+ (Baritone's latest)? UnionClef is pinned to 1.21.1; updating it would require Gradle work.

2. **Anti-cheat compatibility:** If targeting multiplayer servers, UnionClef's WindMouse rotations help, but each server has different anti-cheat configurations. Testing matrix will be large.

3. **Scalability:** Py4J uses local sockets — how many bot instances can run on one machine before contention? Plan for 1 JVM + 1 Py4J gateway per bot.

4. **World state serialization:** What format should world state be sent to Python in? JSON is easy but slow for large chunk data. Consider binary formats (MessagePack, Protobuf) over WebSocket.

5. **Error handling:** Py4J connections can drop. What's the reconnect strategy? UnionClef's implementation may not have robust reconnection logic.

6. **Legal considerations:** Running bots on third-party servers may violate Terms of Service. Design for your own server or single-player worlds first.

---

## Source Index

| [^43^] | `github.com/gnembon/fabric-carpet/blob/master/docs/scarpet/api/Overview.md` |
| [^47^] | `github.com/gaucho-matrero/altoclef` (archived) |
| [^49^] | `github.com/prismarinejs/mineflayer` |
| [^50^] | `curseforge.com/minecraft/mc-mods/baritone-bot` |
| [^51^] | `modrinth.com/mod/fabric-grpc-api` |
| [^98^] | `arxiv.org/html/2412.05255v1` (TeamCraft benchmark) |
| [^101^] | `github.com/dadencukillia/httpInfoServer-mod` |
| [^103^] | `github.com/MCCTeam/Minecraft-Console-Client` |
| [^106^] | `modrinth.com/project/cq5M3XFb` (ws2tcp) |
| [^111^] | `github.com/gnembon/fabric-carpet` |
| [^112^] | `9minecraft.net/carpet-mod/` |
| [^114^] | `github.com/Hexeption/MCP-Reborn` |
| [^118^] | `github.com/LiterMC/MISS` |
| [^133^] | `arxiv.org/abs/2406.05720` (VillagerAgent) |
| [^145^] | `docs.fabricmc.net/develop/networking` |
| [^148^] | `pypi.org/project/fabricpy/` |
| [^149^] | `gist.github.com/apple502j/9c6b9e5e8dec37cbf6f3916472a79d57` |
| [^150^] | `github.com/cabaletta/baritone` |
| [^155^] | `modrinth.com/mod/pythonmcmod` |
| [^159^] | `medium.com/@hieutrantrung.it/the-ai-agent-framework-landscape-in-2025` |
| [^160^] | `github.com/3ndetz/autoclef` (archived, moved to UnionClef) |
| [^162^] | `cybrancee.com/blog/minecraft-client-side-and-server-side-explained/` |
| [^180^] | `github.com/3ndetz/unionclef` |
| [^183^] | `github.com/cnsdqd-dyb/VillagerAgent` |
| [^185^] | `github.com/FabricMC/fabric-api` |
| [^197^] | `openreview.net/forum?id=ehfRiF0R3a` (Voyager) |
| [^199^] | `github.com/MineDojo/Voyager` |
| [^200^] | `github.com/cabaletta/baritone/releases` |
| [^207^] | `mccteam.github.io/guide/installation.html` |
| [^209^] | `manacube.com/threads/minecraft-console-clients.63270/` |
| [^210^] | `github.com/MCCTeam/Minecraft-Console-Client/issues/2738` |
| [^211^] | `issues.ecosyste.ms/hosts/GitHub/repositories/Hexeption%2FMCP-Reborn/issues` |
| [^212^] | `github.com/mindcraft-bots/mindcraft` |
| [^214^] | `github.com/dubfib/mcp-reimagined` |
