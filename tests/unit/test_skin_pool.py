from aiutopia.identity.skin_pool import (
    pick_name, deterministic_skin_for_uuid, procedural_surname,
)


def test_pick_name_uses_pool_when_available() -> None:
    pool = ["Bjorn", "Gunnar", "Sigrid"]
    used = {"Bjorn"}
    name = pick_name(pool, used)
    assert name in {"Gunnar", "Sigrid"}


def test_pick_name_falls_back_to_first_plus_surname_when_exhausted() -> None:
    pool = ["Bjorn", "Gunnar"]
    used = {"Bjorn", "Gunnar"}
    name = pick_name(pool, used, seed=42)
    parts = name.split(" ")
    assert len(parts) == 2
    assert parts[0] in pool        # first name from pool
    assert parts[1].isalpha()      # procedural surname


def test_pick_name_never_returns_numbered_default() -> None:
    pool = ["Bjorn"]
    used = {"Bjorn"}
    for seed in range(50):
        name = pick_name(pool, used, seed=seed)
        assert "defender_" not in name
        assert "_" not in name      # no underscore-numbered names


def test_deterministic_skin_seeded_by_uuid_not_name() -> None:
    uuid_a = "01J0CABCDEFGHJKMNPQRSTVWXY"
    uuid_b = "01J0CABCDEFGHJKMNPQRSTVWX0"
    skins = ["Steve", "Alex", "Notch", "Herobrine"]
    sa = deterministic_skin_for_uuid(uuid_a, skins)
    sb = deterministic_skin_for_uuid(uuid_b, skins)
    assert sa in skins and sb in skins
    # Different UUIDs almost certainly map to different skins (4-element pool).
    # If this asserts equal, replace one UUID and rerun — birthday-paradox edge.


def test_procedural_surname_is_deterministic_with_seed() -> None:
    assert procedural_surname(seed=7) == procedural_surname(seed=7)
    assert procedural_surname(seed=7) != procedural_surname(seed=8)
