# Cluster 9: Geyser/Floodgate for Bedrock Support on Modded Fabric Servers

## Research Report: Bedrock Client Support for AI-Agent Fabric Servers

**Date**: 2026-05-19
**Researcher**: Deep Research Agent
**Sources**: 15+ independent web searches across GeyserMC docs, GitHub issues, Fabric community forums, Reddit, SpigotMC, and Modrinth.

---

## Executive Summary

- **Feasibility Verdict: CONDITIONAL YES, with significant architectural constraints.** Bedrock support via Geyser-Fabric is technically possible for a server running exclusively server-side Fabric mods, but it is NOT a drop-in solution. Every mod must be audited for client-side requirements. [^389^]

- **The hard rule**: Geyser emulates a vanilla Java client. If a vanilla Java client cannot join your Fabric server, Geyser cannot work. [^389^] This means all mods must be strictly server-side (like Lithium, Carpet, server management mods). Any mod requiring client-side installation is an immediate blocker for Bedrock players.

- **Floodgate-Fabric now exists**, solving the authentication problem that plagued Fabric servers for years. As of 2024-2025, both Geyser-Fabric and Floodgate-Fabric are actively maintained and available on Modrinth. [^432^] [^436^]

- **Custom entities/items for AI agents**: Partially supported via Geyser's Custom Item API v2 and Custom Blocks API, but requires significant manual mapping work and Bedrock resource pack creation. Geyser does NOT auto-convert Java mods to Bedrock. [^397^] [^443^]

- **Your gRPC/websocket bridge mod**: This is the biggest unknown. Custom Fabric networking channels using plugin messages may pass through Geyser untranslated, but complex custom packet structures could be dropped. There is no documented compatibility layer for this use case. [^384^] [^515^]

- **Hydraulic (the future)**: GeyserMC is developing "Hydraulic," a companion mod for modded server support. It is in very early pre-alpha, crashes with complex mods, and is explicitly not for production. Do not plan around it. [^433^] [^430^]

---

## 1. How GeyserMC Works (Protocol Translation Layer)

GeyserMC is a protocol translation proxy that sits between a Bedrock client and a Java server. It emulates a Minecraft Java Edition client to the server, translating all packets bidirectionally. [^390^] [^389^]

**Key architectural facts:**
- **Geyser-Fabric** runs as a mod inside the Fabric server's `mods` folder. Config is located at `/config/Geyser-Fabric/config.yml`. [^383^]
- It requires **Fabric API** to be installed. [^383^]
- Geyser-Fabric has **direct world access** optimizations, giving it lower memory usage and greater translation accuracy than standalone Geyser. [^389^]
- The server must support the **latest vanilla Minecraft Java version** (currently 1.21.9/26.1). Geyser emulates a 26.1 client. [^450^]
- **Critical**: Geyser-Fabric and Geyser-NeoForge only support the latest Minecraft Java version. They cannot run on older versions. [^450^]

**What this means for your AI village:**
- Geyser will translate ALL vanilla protocol packets (movement, block placement, chat, inventory, entity spawning, etc.) between Bedrock and Java formats.
- Any feature that deviates from vanilla behavior (custom blocks, custom entities, custom packets) requires explicit mapping/translation support.

---

## 2. Floodgate on Fabric (Authentication & Account Handling)

Floodgate is the companion authentication system that allows Bedrock players to join without Java accounts.

**Current status (2024-2025):**
- **Floodgate-Fabric EXISTS and is actively maintained.** It had been a major gap for years (GitHub issue #71 from 2020 requested it), but it is now available. [^432^] [^436^]
- The mod is published on Modrinth as a Fabric server-side mod. [^432^]
- Floodgate handles Xbox Live authentication for Bedrock players, assigning them Java-compatible UUIDs.
- Floodgate can cause issues with plugins/mods that modify the login process or check UUIDs. [^389^]

**Skin handling:**
- Bedrock player skins require additional setup. Geyser has a built-in custom skulls system that can pre-register skins as custom blocks. [^438^]
- Without Floodgate or additional skin plugins, Bedrock skins won't show for Java players and vice versa.

---

## 3. Fabric Mod Compatibility Matrix

The following matrix rates Fabric mod compatibility with Geyser-Fabric:

| Mod Category | Example Mods | Geyser Support | Notes |
|---|---|---|---|
| **Performance mods** | Lithium, Phosphor, Starlight, FerriteCore | **YES - Fully Compatible** | Pure server-side optimizations with no client impact. [^434^] [^438^] |
| **Carpet Mod** | Carpet, Carpet Extra, Carpet TIS | **PARTIAL - Known Issues** | Carpet's fake player bots (`/player` command) crash or error with Geyser-Fabric. Interaction with fake players broken for Bedrock clients. [^394^] [^515^] |
| **Utility/QoL (server-side)** | Ledger, LuckPerms, WorldEdit, Styled Chat | **YES - Fully Compatible** | Server-side only, no client requirements. [^434^] |
| **Structure mods (server-side)** | Mo' Structures, Repurposed Structures | **YES - Compatible** | Server-side world generation, vanilla client compatible. [^438^] |
| **Mods using Polymer** | Universal Shops, Polymer ports | **PARTIAL - Block breaking issues** | Polymer custom blocks (represented by player heads) cause NPEs in Geyser when Bedrock players try to break them. [^384^] |
| **Mods using PolyMc** | Various ported mods | **PARTIAL - Varies by mod** | PolyMc makes mods work with vanilla clients, but playability varies. [^462^] [^481^] |
| **Mods requiring client install** | Create, Origins, most content mods | **NO - INCOMPATIBLE** | Any mod needing client-side mods breaks Bedrock support entirely. [^389^] |
| **Custom networking mods** | Your gRPC/websocket bridge | **UNKNOWN - LIKELY PROBLEMATIC** | Custom plugin channels may pass through, but complex custom payloads are untested and likely dropped. |

**Key source**: Geyser's FAQ explicitly states: "the short answer: if a vanilla client can join the server, then so can Geyser." [^389^]

---

## 4. Specific Incompatibilities to Investigate

### 4.1 Fabric Carpet + Geyser: PROBLEMATIC

**Status**: Carpet Mod had Geyser compatibility issues that were partially fixed in v1.4.57 (Jan 2022), which "fixed carpet compatibility with adventure platform libraries, geyser, floodgate etc." [^387^]

**However**, there are ongoing issues:
- **GitHub issue #2251**: When using Carpet mod to spawn a bot via `/player steve spawn`, Geyser-Fabric throws "An unexpected error occurred trying to execute that command." [^394^]
- **Fake player interaction**: Bedrock players cannot interact with fake players (bot entities) created by Carpet or Leaves `/bot create`. Right-click interactions fail completely, and Bedrock players may not see the entities correctly. [^515^]
- **Impact for AI village**: If your AI agents use Carpet's fake player system for visualization, Bedrock players will NOT be able to interact with them properly. Consider using vanilla entity-based visualization (armor stands, named mobs) instead.

### 4.2 Custom gRPC/Websocket Bridge + Geyser: UNKNOWN RISK

**Status**: This is the most critical unknown for your architecture.

**Analysis**:
- Geyser translates standard Minecraft protocol packets. Custom plugin channels (used by Fabric mods for custom networking) may or may not pass through.
- Geyser has been observed dropping or failing to translate packets it doesn't understand, particularly custom payloads related to blocks and entities. [^384^] [^515^]
- If your bridge mod uses Minecraft's standard `CustomPayloadS2CPacket`/`CustomPayloadC2SPacket` channels, these might pass through untranslated (since Geyser is protocol-agnostic for unknown packets).
- However, if your bridge uses custom entity spawn packets, custom block state packets, or modifies the login handshake, Geyser will likely break them.
- **Recommendation**: Test this explicitly. Implement your bridge using standard Minecraft custom payload channels, not custom packet types. Ensure it works with a vanilla Java client first.

### 4.3 Custom Entities for Agent Visualization: PARTIALLY SUPPORTED

**Status**: Geyser supports custom entity visualization through several mechanisms:

- **Custom Items API v2**: Supports non-vanilla (modded) items from Fabric/NeoForge servers via the `NonVanillaCustomItemDefinition` API. Requires manual registration and a Bedrock resource pack. [^397^]
- **Custom Blocks API**: Supports custom block mappings via `GeyserDefineCustomBlocksEvent`. [^443^]
- **Custom Skulls**: Can be pre-registered to show custom textures. [^438^]
- **Extensions**: Platform-agnostic "plugins" for Geyser that can register custom items, blocks, and listen to events. [^400^]

**Limitations**:
- Geyser does NOT auto-convert Java mod content to Bedrock. You must manually create mappings and Bedrock resource packs. [^397^] [^398^]
- Custom entities (e.g., a unique AI agent entity type) require Geyser extension development with custom Bedrock entity definitions.
- There are known issues with entity visibility: Bedrock players sometimes cannot see entities at all after updates (GitHub issue #5952). [^448^]
- NPC skins and custom heads have had rendering regressions in recent builds. [^522^]

**Practical approach for AI village**: Use vanilla entity types (armor stands, named villagers, allays) with custom names and resource packs rather than custom entity types. This is far more reliable.

### 4.4 Inventory/Crafting Interactions: KNOWN GAPS

**Status**: Geyser has well-documented inventory translation limitations:

- **Cannot distinguish left/right clicks in inventories**: This is an unfixable Bedrock limitation. [^393^]
- **Custom anvil recipes**: Do not work (unfixable). [^393^]
- **Custom smithing table ingredients/patterns**: Do not work. [^393^]
- **Custom GUI interactions**: GUIs that rely on teleporting the player (some custom inventory implementations) break because Bedrock requires physical chest blocks to open inventories. [^454^]
- **Custom furnace cook times**: Do not work. [^393^]

**Impact**: If your AI village has custom crafting systems or complex inventory UIs, Bedrock players will have degraded or broken experiences.

---

## 5. Bot/AI Mod Interactions

### 5.1 Baritone: NOT APPLICABLE (Client-Side Only)

**Critical finding**: Baritone is a **client-side only** pathfinding mod. It runs on the player's Minecraft client, not on the server. [^439^] [^440^]

- Baritone bots connect to servers as regular Java players. They do NOT require server-side installation.
- A Baritone-based bot connecting through Geyser would be pointless: Baritone runs on Java, and Geyser connects Bedrock clients.
- For your AI village, you need **server-side** AI agents. Options:
  - **Mineflayer**: Node.js-based bot framework that connects as a regular player. Fully compatible with Geyser (since it connects as a Java client). [^451^] [^502^]
  - **Carpet fake players**: Server-side bots, but have Geyser compatibility issues (see 4.1).
  - **Custom Fabric mod**: Your own agent implementation using the server-side API.

### 5.2 AI Agents Coexisting with Bedrock Players: YES, with caveats

**Finding**: There is no inherent incompatibility between AI agents and Geyser-connected Bedrock players, as long as the agents are implemented server-side.

- If agents are implemented as **vanilla entities** (armor stands, custom-named mobs), Bedrock players will see them (with possible texture limitations).
- If agents are implemented as **fake players** (Carpet-style), Bedrock players will have interaction issues.
- If agents use **custom packets** for their UI/state, those packets must be compatible with vanilla Java clients (and thus Geyser).

**Mineflayer specifically**: Mineflayer bots connect as standard Java clients and are fully compatible with Geyser. [^502^] A Mineflayer bot would not even "know" that some players are on Bedrock.

---

## 6. Performance Implications

### 6.1 Geyser Overhead

Geyser adds measurable overhead to a Minecraft server:

- **Per-player memory**: Each Bedrock player consumes additional memory for Geyser's session cache, translation buffers, and Bedrock protocol state. Reports show 10-15% RAM increase per Bedrock player compared to Java players. [^485^]
- **Memory leaks**: GitHub issue #2504 documents excessive memory usage with high Bedrock player counts, requiring regular restarts. [^489^]
- **Scoreboard packet lag**: Geyser's scoreboard translation can cause "serious lag" under high scoreboard packet rates. The config option `scoreboard-packet-threshold` mitigates this by limiting updates. [^455^] [^464^]
- **CPU overhead**: Protocol translation is CPU-intensive, especially for chunk translation and entity management.

### 6.2 Impact on RL Training (Tick Rate)

**Relevant concerns:**
- Geyser-Fabric's direct connection mode (`use-direct-connection: true`) minimizes latency by avoiding a separate TCP connection. This should be enabled. [^455^]
- The `use-adapters` option enables direct server methods for block state retrieval. Disabling it has a performance impact. [^465^]
- Bedrock players on the server during RL training runs will add CPU load. For a small number of observer Bedrock players (<5), the impact should be minimal on a well-provisioned server.
- **Critical**: If your RL training uses scoreboards, team displays, or frequent tab list updates, the `scoreboard-packet-threshold` config will be essential. [^455^]

### 6.3 Optimization Recommendations

```yaml
# Geyser config.yml optimizations for RL training environments
bedrock:
  port: 19132
  address: 0.0.0.0

remote:
  address: 127.0.0.1  # Direct connection for Fabric
  port: 25565
  auth-type: floodgate

# Enable direct connection for lowest latency (Fabric only)
use-direct-connection: true

# Limit scoreboard updates to prevent lag
scoreboard-packet-threshold: 20

# Use adapters for better performance
use-adapters: true
```

---

## 7. The "Hydraulic" Project: Future of Modded Support

**Status**: The Geyser team is developing **Hydraulic**, a companion mod specifically designed for modded Minecraft servers. [^433^] [^430^]

**Current state (2025-2026):**
- Explicitly stated: "This project is still in very early development and should not be used on production setups!" [^433^]
- No binaries are distributed; must be compiled from source.
- Only supports simple mods (e.g., Advanced Netherite works; Farmer's Delight crashes with `StringIndexOutOfBoundsException` during texture conversion). [^495^]
- Requires a special Custom Item API V2 branch of Geyser.
- Known crashes on Fabric 1.21.8 startup. [^429^]

**Verdict**: Hydraulic is a promising long-term project but is NOT a viable solution for the foreseeable future. Do not architect your server around it.

---

## 8. Concrete Recommendations

### If you want Bedrock support, follow these architectural constraints NOW:

1. **Server-side mods ONLY**: Every mod you install must work with a vanilla Java client. Use the Fabric server-side mods list [^434^] [^438^] as a reference. Test by joining with an unmodded Java client.

2. **Avoid client-required mods**: No Create, no Origins, no custom client UI mods. These are absolute blockers.

3. **Use vanilla entity types for AI agents**: Instead of custom entity types, use armor stands, named villagers, allays, or other vanilla entities with resource packs. This ensures Bedrock compatibility.

4. **Test your gRPC/websocket bridge with a vanilla client**: Before committing to Geyser, verify your custom networking works when connected with an unmodded Java client. If it doesn't work vanilla, it won't work with Geyser.

5. **Avoid Carpet fake players for Bedrock-visible agents**: The `/player` bot system has known Geyser incompatibilities. Use alternative agent visualization.

6. **Use Mineflayer for external AI bots**: If you need bot players that connect as actual players, Mineflayer is the proven, Geyser-compatible solution. [^451^]

7. **Monitor performance**: Set `scoreboard-packet-threshold` in Geyser config. Plan for ~10-15% additional RAM per Bedrock player. [^485^]

8. **Stay on latest Minecraft version**: Geyser-Fabric only supports the latest Java version. You cannot lag behind on updates. [^450^]

9. **Create Bedrock resource packs**: Any custom items/blocks your AI village uses will need manual Bedrock resource pack creation and Geyser API registration. [^397^] [^398^]

10. **Keep Geyser and Floodgate updated**: Both are actively developed. Updates often fix critical compatibility issues.

---

## 9. Feasibility Verdict

### SHOULD the user plan for Bedrock support?

**Answer: YES, but with the understanding that it constrains your mod selection and agent architecture significantly.**

Bedrock support is feasible for a "vanilla-like" AI village experience where:
- AI agents are visible as vanilla entities (armor stands, named mobs)
- The server runs only server-side mods
- Custom networking uses standard Minecraft custom payload channels
- You're willing to create Bedrock resource packs for any custom visual content
- You accept inventory/GUI limitations for Bedrock players

### If YES, what architectural constraints now?
- Server-side mods only (verify with vanilla Java client test)
- Vanilla entity types for agent visualization
- Standard custom payload networking (no custom packet types)
- Regular Minecraft version updates (Geyser-Fabric is version-locked)
- Additional RAM provisioning (~10-15% per Bedrock player)
- Manual Bedrock resource pack creation for custom content

### If NO, what's the blocker?
- Any required client-side mod (Create, Origins, etc.) = instant blocker
- Custom entity types without Geyser extension development = blocker
- Complex custom networking that doesn't work with vanilla clients = likely blocker
- Agent visualization relying on Carpet fake players = partial blocker
- Custom inventory/crafting systems = degraded experience for Bedrock players

---

## 10. Compatibility Matrix Summary

| Component | Status | Notes |
|---|---|---|
| Geyser-Fabric | ✅ Available | Latest MC version only [^383^] |
| Floodgate-Fabric | ✅ Available | As of 2024 [^432^] |
| Lithium/Phosphor | ✅ Compatible | Pure server-side [^434^] |
| Carpet Mod | ⚠️ Partial | Fake player issues with Geyser [^394^] |
| Baritone | N/A (Client-side) | Not a server mod [^439^] |
| Mineflayer bots | ✅ Compatible | Connects as Java client [^451^] |
| Custom items/blocks | ⚠️ Manual work | Requires Geyser API + resource packs [^397^] |
| Custom entities | ⚠️ Complex | Requires Geyser extensions [^400^] |
| Custom networking | ❓ Unknown | Test with vanilla client first |
| Hydraulic mod | ❌ Not ready | Pre-alpha, not for production [^433^] |
| Inventory/GUI | ⚠️ Limited | Left/right click unfixable [^393^] |
| Scoreboard heavy use | ⚠️ Config needed | Use packet threshold [^455^] |

---

## 11. Open Questions

1. **Does your gRPC/websocket bridge use standard Minecraft custom payload channels?** If yes, it likely works. If it uses custom packet types or modifies the protocol handshake, it will break.

2. **How do you plan to visualize AI agents?** If using vanilla entities with resource packs, Bedrock support is straightforward. If using custom entity types or Carpet fake players, there will be problems.

3. **Will Bedrock players need to interact with agent UIs?** If agents have custom inventories/GUIs, expect limitations (no left/right click distinction, custom GUI breakage).

4. **What's your expected Bedrock player count?** For 1-5 observer Bedrock players, performance impact is manageable. For 20+ concurrent Bedrock players, significant optimization work is needed.

5. **Are you willing to create and maintain Bedrock resource packs?** This is required for any custom visual content.

6. **Can your agent system tolerate vanilla entity limitations?** Armor stands and named mobs are reliable but limited. If you need complex agent animations or behaviors, Geyser extension development is required.

---

## Sources

[^381^] MineStrator GeyserMC Tutorial (2025): https://minestrator.com/en/blog/article/install-geysermc-floodgate-java-bedrock-crossplay-2025

[^383^] Geyser-Fabric Wiki: http://geysermc.ru/wiki/other/geyser-fabric

[^384^] GitHub Issue #4555 - Polymer custom blocks error: https://github.com/GeyserMC/Geyser/issues/4555

[^387^] Carpet Mod v1.4.57 changelog (Geyser fix): https://www.curseforge.com/minecraft/mc-mods/carpet/files/3613865

[^389^] GeyserMC FAQ (official): https://geysermc.org/wiki/geyser/faq/

[^390^] GeyserMC Homepage: https://geysermc.org/

[^393^] Geyser Current Limitations: https://geysermc.org/wiki/geyser/current-limitations/

[^394^] GitHub Issue #2251 - Carpet bots error: https://github.com/GeyserMC/Geyser/issues/2251

[^397^] Geyser Custom Items documentation: https://geysermc.org/wiki/geyser/custom-items/

[^398^] Geyser Resource Packs documentation: https://geysermc.org/wiki/geyser/packs/

[^400^] Geyser Extensions documentation: https://geysermc.org/wiki/geyser/extensions/

[^429^] Hydraulic GitHub Issue #50 - Startup crash: https://github.com/GeyserMC/Hydraulic/issues/50

[^430^] Hydraulic Wiki: https://geysermc.org/wiki/other/hydraulic/

[^432^] Floodgate-Fabric Modrinth: https://modrinth.com/mod/floodgate/version/YlFk39jA

[^433^] Hydraulic GitHub Repository: https://github.com/GeyserMC/Hydraulic

[^434^] Fabric Server-Side Mods List: https://github.com/supsm/fabric-serverside-mods

[^436^] Floodgate-Modded Commits (active development): https://github.com/GeyserMC/Floodgate-Fabric/commits

[^438^] Fabric Established Server-Side Mods: https://wiki.fabricmc.net/community:serverside_mods

[^439^] Baritone Simply (client-side): https://www.curseforge.com/minecraft/mc-mods/baritone-simply

[^440^] Baritone commands client-side: https://www.spigotmc.org/threads/executing-baritone-commands-from-a-plugin.533036/

[^443^] Geyser API documentation: https://geysermc.org/wiki/geyser/api/

[^448^] GitHub Issue #5952 - Invisible entities: https://github.com/GeyserMC/Geyser/issues/5952

[^450^] Geyser Supported Versions: https://geysermc.org/wiki/geyser/supported-versions/

[^451^] Mineflayer documentation: https://mineflayer.com/what-is-mineflayer-used-for/

[^454^] GitHub Issue #1441 - Custom inventories: https://github.com/GeyserMC/Geyser/issues/1441

[^455^] Geyser Config documentation: http://geysermc.ru/wiki/geyser/understanding-the-config

[^462^] PolyMc documentation: https://wiki.fabricmc.net/community:serverside_mods

[^464^] GeyserMC Config PDF (scoreboard threshold): https://www.scribd.com/document/872571485/Config-yml

[^481^] PolyMc project page: https://theepicblock.github.io/PolyMc/

[^485^] GitHub Issue #3614 - High memory usage: https://github.com/GeyserMC/Geyser/issues/3614

[^489^] GitHub Issue #2504 - Memory leak: https://github.com/GeyserMC/Geyser/issues/2504

[^495^] Hydraulic Issue #65 - Farmer's Delight crash: https://github.com/GeyserMC/Hydraulic/issues/65

[^502^] Mineflayer Geyser compatibility discussion: https://github.com/PrismarineJS/mineflayer/discussions/3540

[^515^] GitHub Issue #5634 - Fake player interaction broken: https://github.com/GeyserMC/Geyser/issues/5634

[^522^] GitHub Issue #6026 - NPC skins not loading: https://github.com/GeyserMC/Geyser/issues/6026

[^527^] BisectHosting Geyser/Floodgate install guide: https://help.bisecthosting.com/hc/de/articles/40526648951707-How-to-Install-Geyser-and-FloodGate-on-a-Minecraft-Server

---

*End of Research Report*
