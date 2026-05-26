# AI Utopia — Multi-Agent Minecraft AI Village

Persistent multi-agent reinforcement-learning system. Four specialized agents
(gatherer, builder, farmer, defender) cooperatively grow and operate a village
on a private Minecraft Java server. Friends can join, are recognized as
friendly entities, and can NL-chat with agents via `@<agent_name>` messages.

## Status

**Milestone 0 — Infrastructure foundation** (this milestone).

After M0 you should be able to:

```bash
aiutopia agent spawn --role gatherer
```

…and see a Carpet fake player appear in your connected MC client. No
training has happened; the agent stands still. That's M1's job.

## Quickstart

```bash
# 1. Install Python deps
python -m pip install -e ".[dev]"

# 2. Run unit tests
pytest -v -m "not integration and not determinism"

# 3. (Optional, requires GPU) Run determinism harness scaffold
pytest -v -m determinism

# 4. Build the Fabric mod
cd fabric_mod && ./gradlew build && cd ..

# 5. Install a local Fabric 1.21.1 server with:
#      - Fabric API
#      - Carpet
#      - Lithium, FerriteCore
#      - fabric_mod/build/libs/aiutopia-mod-0.0.0-m0.jar
#    Start with: -Daiutopia.py4j.port=25099

# 6. Connect a Minecraft Java 1.21.1 client to localhost:25565

# 7. Run the end-to-end smoke test
PY4J_PRODUCTION_PORT=25099 scripts/smoke-test.sh
```

## Architecture

See `docs/superpowers/specs/2026-05-25-ai-utopia-minecraft-village-design.md`
for the full spec. Two-world topology (training instances + persistent
production server), three-tier brain (LLM planner → goal spec → per-role RL),
Level-D persistent identity, 22–28 weeks to M6.

## Repository layout

| Path | Role |
|---|---|
| `src/aiutopia/` | Python package — env wrapper, identity, schemas, memory, CLI, planner stubs |
| `fabric_mod/` | Java Fabric mod — Py4J bridge + mixins + Carpet integration |
| `tests/{unit,integration,determinism}/` | Test suites (markered for selective runs) |
| `scripts/` | Smoke test + backup automation |
| `docker-compose.production.yml` | M6 deployment skeleton (ZGC JVM args) |
| `docs/superpowers/specs/` | Design specs (committed history) |

## Constraints

- **Python 3.12** pinned (`.python-version`).
- **Minecraft Java 1.21.1** pinned (UnionClef baseline; see T19 in `IMPLEMENTATION_PLAN.md`).
- **Server stack:** Fabric + Carpet + Lithium + FerriteCore. **Generational ZGC**, not G1GC.
- **All identifiers:** ULID Crockford base32.

## Contributing

1. Follow `IMPLEMENTATION_PLAN.md` task-by-task — every change should be a focused commit on a single task's checkbox.
2. `pre-commit` hooks must pass (`ruff`, `mypy`, `pytest`).
3. Integration tests requiring a live Fabric server are marked `@pytest.mark.integration` and skip gracefully when the server isn't reachable.

## License

MIT.
