# AI Utopia Fabric Mod

Java side of the Python ↔ Minecraft bridge for the multi-agent village system.
This is a *server-side* mod by design — install on a Fabric 1.21.1 dedicated server.

## Components

- **`Py4JEntryPoint`** — Java methods exposed to Python via Py4J. Implements
  `observationsAll()`, `motorBridge()`, `commBus()`, `resetWorld()`,
  `advanceTickAwaitEvents()`. Forked from UnionClef.
- **Mixins (added incrementally per the task list):**
  - `KickPlayerMixin` (T21) — block `/kick` of Carpet fake players.
  - `ChatMessageMixin` (T23) — intercept `@<agent_name>` chat → emit ChatEvent.
  - `PlayerListMixin` — deferred to M1 (per-recipient `/list` filter requires non-trivial mapping work).

## Build

```bash
./gradlew build
# → build/libs/aiutopia-mod-<version>.jar
```

Install in `<server>/mods/` alongside:
- Fabric API
- Carpet
- Lithium
- FerriteCore
- Krypton (optional)

## Decision gate (T19)

Pinned to MC 1.21.1 because UnionClef baseline targets that version.
If you need a later 1.21.x, regenerate `gradle.properties` AND verify the
UnionClef fork still compiles — bridge surface may need patching.
