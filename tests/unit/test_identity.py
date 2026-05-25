from pathlib import Path

import pytest

from aiutopia.common.ids import is_ulid
from aiutopia.identity.service import (
    IdentityService, init_identity_db,
)


@pytest.fixture
def svc(identity_db_path: Path) -> IdentityService:
    init_identity_db(identity_db_path,
                     Path("src/aiutopia/identity/migrations"))
    return IdentityService(identity_db_path)


def test_seeded_roles_present(svc: IdentityService) -> None:
    for role_id in ("gatherer", "builder", "farmer", "defender"):
        role = svc.get_role(role_id)
        assert role.role_id == role_id
        assert role.max_lives == 1
        assert len(role.default_skin_pool) == 12


def test_spawn_agent_creates_alive_row_with_ulid_uuid(svc: IdentityService) -> None:
    agent = svc.spawn_agent("gatherer", "Bjorn", born_at=1_700_000_000)
    assert is_ulid(agent.agent_uuid)
    assert agent.status == "alive"
    assert agent.role_id == "gatherer"
    assert agent.agent_name == "Bjorn"
    assert agent.skill_library_id == f"skill_lib_{agent.agent_uuid}"
    assert agent.memory_id == f"mem_{agent.agent_uuid}"


def test_spawn_agent_rejects_duplicate_living_name(svc: IdentityService) -> None:
    svc.spawn_agent("gatherer", "Bjorn", born_at=1)
    with pytest.raises(Exception):
        svc.spawn_agent("gatherer", "Bjorn", born_at=2)


def test_dry_run_succession_role_persists_uuid_rotates(svc: IdentityService) -> None:
    # 1. Spawn Bjorn (defender)
    bjorn = svc.spawn_agent("defender", "Bjorn", born_at=100)

    # 2. Bjorn dies
    svc.record_death(bjorn.agent_uuid, died_at=200, cause_of_death="creeper")
    dead = svc.get_agent(bjorn.agent_uuid)
    assert dead.status == "dead"
    assert dead.died_at == 200

    # 3. Funeral written
    funeral_id = svc.record_funeral(
        deceased_agent_uuid=bjorn.agent_uuid,
        witness_uuids=[],
        event_summary="Bjorn the defender died defending the south wall.",
        written_at=201,
    )
    assert funeral_id > 0

    # 4. Successor spawns next morning — same role_id, NEW agent_uuid
    gunnar = svc.spawn_agent("defender", "Gunnar", born_at=24_000)
    assert gunnar.role_id == "defender"
    assert gunnar.agent_uuid != bjorn.agent_uuid
    assert gunnar.skill_library_id != bjorn.skill_library_id   # fresh memory
    assert gunnar.memory_id != bjorn.memory_id

    # 5. Both lives recorded
    living = svc.list_living_agents()
    assert [a.agent_name for a in living] == ["Gunnar"]


def test_record_death_rejects_non_ulid(svc: IdentityService) -> None:
    with pytest.raises(ValueError):
        svc.record_death("not-a-ulid", died_at=1, cause_of_death="x")
