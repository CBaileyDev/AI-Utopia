## 9. Geyser/Floodgate Bedrock Assessment

The decision to defer Bedrock client support does not eliminate the need for a rigorous compatibility assessment. If the AI village server is to remain architecturally open to Bedrock players in the future, every design choice made today must be compatible with the constraints Geyser imposes. This chapter evaluates the technical feasibility of connecting Bedrock clients to a modded Fabric server through GeyserMC and Floodgate, identifies specific mod-level incompatibilities, and translates these findings into actionable architectural constraints.

### 9.1 Technical Feasibility

#### 9.1.1 Geyser-Fabric: Actively Maintained, Emulates a Vanilla Java Client

GeyserMC is a protocol-translation proxy that converts Minecraft Bedrock protocol packets to Java Edition protocol packets (and vice versa) in real time. For Fabric servers, **Geyser-Fabric** runs as a mod inside the server's `mods` folder, configured at `/config/Geyser-Fabric/config.yml`. [^383^] Unlike the standalone proxy variant, the Fabric-native build has direct world access, which reduces memory overhead and improves translation accuracy for block states and entity metadata. [^389^]

As of mid-2026, Geyser-Fabric is actively maintained and tracks the latest Java Edition protocol version (1.21.x, protocol 26.1). [^450^] It requires Fabric API to be present and translates all vanilla protocol traffic — movement, block placement, chat, inventory operations, entity spawning, scoreboard updates, and chunk data — between Bedrock and Java formats. [^390^] Anything that deviates from vanilla behavior requires explicit mapping support, which is the source of nearly every incompatibility discussed later in this chapter.

#### 9.1.2 Floodgate-Fabric: Authentication Gap Closed

For years, a major barrier to Bedrock support on Fabric was the absence of **Floodgate-Fabric**, the authentication companion that lets Bedrock players (authenticated via Xbox Live) join Java servers without purchasing a Java Edition account. The original Floodgate was available only for Bukkit/Spigot and proxy layers; a Fabric port was one of the most requested features on the GeyserMC GitHub (issue #71, opened in 2020). [^432^]

That gap closed in 2024–2025. Floodgate-Fabric is now published on Modrinth, actively maintained with regular commits, and handles Xbox Live authentication plus Java-compatible UUID assignment for Bedrock players. [^432^] [^436^] The mod runs server-side only, so it introduces no client-side dependencies. However, it can conflict with mods that modify the login handshake or perform strict UUID validation, which is a concern worth auditing if the AI village implements custom authentication or player-state management logic. [^389^]

#### 9.1.3 The Hard Rule: Vanilla Client Compatibility Is the Gate

Geyser's FAQ states the compatibility rule in one sentence: *"If a vanilla client can join the server, then so can Geyser."* [^389^] Because Geyser emulates a vanilla Java client — not a modded one — any mod that requires a corresponding client-side installation is an immediate and absolute blocker for Bedrock players. This single constraint eliminates the majority of Fabric content mods (Create, Origins, EMI/JEI, most rendering and UI mods). It also means that the server operator cannot rely on Fabric's rich client-side mod ecosystem to enhance the Bedrock player experience; whatever the vanilla Java client sees is exactly what the Bedrock client will see after translation.

### 9.2 Compatibility Analysis

#### 9.2.1 Fabric Mod Compatibility Matrix

The table below rates common Fabric mod categories for Geyser compatibility, using the server-side versus client-side distinction as the primary filter. The ratings assume a Bedrock player connecting through Geyser-Fabric with Floodgate authentication on a 1.21.x Fabric server.

| Mod Category | Example Mods | Bedrock Compatibility | Requirement Type | Notes |
|---|---|---|---|---|
| Performance/optimization | Lithium, Phosphor, Starlight, FerriteCore | **Full** | Server-side only | Pure server-side optimizations with no client observable changes. [^434^] [^438^] |
| Server management | LuckPerms, Ledger, Styled Chat, CoreProtect | **Full** | Server-side only | No client installation needed; all features visible to Bedrock players. [^434^] |
| World generation (server) | Repurposed Structures, Mo' Structures | **Full** | Server-side only | Structures generate server-side; vanilla client (and thus Geyser) renders them normally. [^438^] |
| AI helper (server-side bots) | Carpet Mod, Carpet Extra | **Partial** | Server-side | Fake players (`/player` command) crash or error on Geyser-Fabric; Bedrock players cannot interact with bot entities. [^394^] [^515^] |
| Polymer-based mods | Universal Shops, Polymer ports | **Partial** | Server-side (emulated) | Custom blocks represented via player heads cause null-pointer exceptions when Bedrock players attempt to break them. [^384^] |
| PolyMc-based ports | Various content mods | **Varies** | Server-side (emulated) | PolyMc makes mods work with vanilla clients, but playability depends on specific block/entity mappings. [^462^] [^481^] |
| Content mods (client required) | Create, Origins, EMI, Sodium, Iris | **None** | Client + server | Any mod requiring client-side installation is an absolute blocker for Bedrock. [^389^] |
| Custom networking mods | gRPC/websocket bridge (hypothetical) | **Unknown** | Depends | Standard `CustomPayload` channels may pass through untranslated; complex custom packet structures risk being dropped. [^384^] [^515^] |
| Geyser extensions | Custom entity/item mappings | **Manual** | Geyser-side | Requires Geyser extension development + Bedrock resource pack creation. [^397^] [^400^] |

**Table 9.1.** Fabric mod compatibility with Geyser-Fabric for Bedrock clients. Ratings assume Minecraft 1.21.x and current Geyser-Fabric builds as of mid-2026. "Full" means Bedrock players experience the mod's features without degradation; "Partial" means some features break; "None" means Bedrock players cannot join at all; "Unknown" means no documented test results exist.

The pattern in Table 9.1 is stark: the only reliably safe mods are those with zero client-side footprint. Performance mods (Lithium, FerriteCore) and server management tools (LuckPerms, Ledger) translate cleanly because they do not alter the network protocol in ways visible to Geyser. The moment a mod introduces custom blocks, custom entities, custom GUIs, or custom networking, Bedrock compatibility becomes conditional at best and impossible at worst.

#### 9.2.2 Critical Blockers: Carpet Fake Players and Custom Entity Mapping

Two blockers deserve detailed attention because they intersect directly with the AI village architecture.

**Carpet Mod fake players.** Carpet's `/player` bot system — a common technique for creating server-side AI agents that look like real players — has documented Geyser incompatibilities. Carpet v1.4.57 (January 2022) included a partial fix for "adventure platform libraries, geyser, floodgate etc.," [^387^] but GitHub issue #2251 confirms that spawning a Carpet bot via `/player steve spawn` still throws "An unexpected error occurred trying to execute that command" on Geyser-Fabric. [^394^] More critically, GitHub issue #5634 documents that Bedrock players cannot interact with fake-player entities at all: right-click interactions fail, and the entities may not render correctly. [^515^] For a village where AI agents are represented as fake players, this means Bedrock observers would see broken or non-interactive agents. The practical workaround is to use vanilla entity types — armor stands, named villagers, allays, or named mobs with resource packs — which Geyser translates reliably because they exist in both Java and Bedrock editions natively.

**Custom entities and items.** Any AI agent represented as a custom entity type (a unique mob not present in vanilla Minecraft) requires manual mapping work. Geyser provides the Custom Items API v2 [^397^] and Custom Blocks API [^443^] for this purpose, along with an Extensions framework for registering custom behaviors. [^400^] However, Geyser does not auto-convert Java mod content to Bedrock equivalents. The server operator must create a Bedrock resource pack, write Geyser extension code to register entity mappings, and maintain these assets across Minecraft version updates. Even then, entity visibility issues persist: GitHub issue #5952 reports that Bedrock players sometimes cannot see custom entities at all after Geyser updates, [^448^] and NPC skin rendering has had regressions in recent builds. [^522^] The recommendation is unambiguous: use vanilla entity types with custom names and resource packs rather than custom entity definitions.

Inventory and GUI interactions carry additional limitations that are architectural, not merely implementation gaps. Geyser cannot distinguish left clicks from right clicks in inventories because the Bedrock protocol does not expose this information — this is labeled an unfixable limitation. [^393^] Custom anvil recipes, custom smithing table patterns, and custom furnace cook times all fail to translate. [^393^] GUIs that rely on teleporting the player to a virtual location break because Bedrock requires a physical chest block to open an inventory screen. [^454^] For an AI village with crafting systems or agent interaction menus, these limitations mean Bedrock players would experience degraded or non-functional interfaces.

#### 9.2.3 The gRPC/WebSocket Bridge: Unknown Compatibility

The proposed Java-to-Python bridge (whether gRPC, WebSocket, or Py4J-based) represents the largest unknown in the compatibility stack. Geyser translates standard Minecraft protocol packets but passes unknown packets through untranslated if they use the standard `CustomPayloadS2CPacket` / `CustomPayloadC2SPacket` channels. [^384^] If the bridge uses these standard channels for its messages, it will likely function unchanged for Bedrock-connected players.

However, if the bridge introduces custom entity spawn packets, custom block state updates, or modifies the login handshake, Geyser will drop or corrupt those packets. [^384^] [^515^] There is no documented case of a Fabric mod using custom plugin channels for an AI bridge being tested with Geyser, which means empirical testing is mandatory. The safe design choice is to implement the bridge using only standard Minecraft custom payload channels, ensure it works with an unmodded Java client, and only then evaluate Geyser compatibility. If the bridge fails with a vanilla client, it will certainly fail with Geyser.

### 9.3 Architectural Implications

#### 9.3.1 Server-Side-Only Constraint Eliminates the Richest Observation Source

The Geyser hard rule forces a decision with cascading consequences. A server-side-only architecture means that observation extraction for AI agents must draw exclusively from server-side APIs: world state, block access, entity positions, inventories, and scoreboard data. What is lost is client-side rendering information — the fully rendered scene, precise particle effects, exact GUI state, shader-driven visual cues, and the full suite of HUD data available to a modded Java client. This is the single richest observation source for embodied AI agents, and it is off the table if Bedrock compatibility is ever desired.

The cross-verification analysis flagged this as a true architectural tension: client-side mods provide richer world state for observation spaces, but server-side-only mods are required for Geyser compatibility. Insight 5 from the cross-dimensional analysis captures the dilemma: the choice is between (a) abandoning Bedrock support entirely and using client-side mods for observation, (b) accepting server-side-only observations and preserving Bedrock compatibility, or (c) running agents as actual Java clients with full mods and accepting no Bedrock support. [^389^]

For a phased project, option (b) is the rational default. Server-side APIs provide sufficient information for symbolic observation spaces (block types, entity positions, inventory contents, health/hunger values), which align with the LLM-planner → RL-policy architecture described in preceding chapters. Pixel-level observations, if needed, can be captured via external screen capture of a Java client spectator instance — a technique that does not require client-side mods on the server itself and does not affect Geyser compatibility.

#### 9.3.2 Verdict: Defer Bedrock Support but Design Server-Side-Only from Day One

The assessment yields a clear verdict: **Bedrock support is technically feasible but should be deferred until after the core AI village architecture is stable.** The conditions for successful Bedrock integration are well understood and can be enforced now without committing to Geyser deployment:

1. **Every mod in the server stack must be verifiably server-side-only.** Test this by joining with an unmodded Java client; if any feature is invisible or non-functional, it will be invisible or non-functional for Bedrock players too. [^389^]
2. **AI agents must be represented as vanilla entity types** with custom names and resource packs, not custom entity types or Carpet fake players. [^394^] [^515^]
3. **The Java-to-Python bridge must use standard Minecraft custom payload channels** and be tested with a vanilla Java client before Geyser testing begins.
4. **Scoreboard and team displays, if used for agent state, must respect Geyser's `scoreboard-packet-threshold` configuration** to avoid translation-induced lag. [^455^]
5. **RAM provisioning must account for ~10–15% additional memory per concurrent Bedrock player**, with reports of memory leaks under high Bedrock player counts requiring periodic server restarts. [^485^] [^489^]

If these constraints are followed from the outset, enabling Geyser and Floodgate later becomes a deployment question, not a redesign. If they are violated — for example, by introducing a client-side observation mod or Carpet fake-player agents — Bedrock support becomes a breaking change requiring architectural rework.

#### 9.3.3 Hydraulic: A Future Modded-Content Bridge, Not a Current Solution

The GeyserMC team is developing **Hydraulic**, a companion mod explicitly designed to bridge modded Java content to Bedrock clients. [^433^] [^430^] Hydraulic performs automatic texture conversion, block state mapping, and item registration for select Fabric mods, with the goal of eliminating the manual Geyser extension work described in §9.2.2.

Hydraulic is not a viable solution for the foreseeable future. As of mid-2026, the project is in pre-alpha with no binary releases; it must be compiled from source. It crashes on Fabric 1.21.8 startup [^429^] and fails with complex mods such as Farmer's Delight (throwing `StringIndexOutOfBoundsException` during texture conversion). [^495^] The official documentation states explicitly: "This project is still in very early development and should not be used on production setups!" [^433^]

The recommendation is unambiguous: do not architect the AI village around Hydraulic. Monitor its development as a long-term option, but assume that any custom content requiring Bedrock visibility will need manual Geyser extension development and Bedrock resource pack creation until Hydraulic reaches a stable release — a milestone with no projected date.
