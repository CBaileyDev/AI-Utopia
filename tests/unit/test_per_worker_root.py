"""Guard the per-worker AIUTOPIA_ROOT suffix against the _w0_w0_w0 compounding bug
(CLAUDE.md "Related bug — per-worker root compounds"): env/wrapper.py mutates
os.environ["AIUTOPIA_ROOT"] in place on every __init__, so the suffix must be
applied idempotently or re-instantiation scatters Chroma/identity across phantom
dirs. Tested via the pure helper (light import — no wrapper/py4j/chroma)."""

from aiutopia.common.config import per_worker_root


def test_appends_suffix_once():
    assert per_worker_root("/var/lib/aiutopia", 0) == "/var/lib/aiutopia_w0"
    assert per_worker_root("/var/lib/aiutopia", 3) == "/var/lib/aiutopia_w3"


def test_idempotent_no_compounding():
    once = per_worker_root("/data/root", 2)
    twice = per_worker_root(once, 2)
    thrice = per_worker_root(twice, 2)
    assert once == "/data/root_w2"
    assert twice == once == thrice  # the bug would give _w2_w2_w2


def test_normalizes_trailing_separators():
    assert per_worker_root("/data/root/", 0) == "/data/root_w0"
    assert per_worker_root("C:\\data\\root\\", 1) == "C:\\data\\root_w1"


def test_distinct_widx_still_appends():
    # A different worker index is NOT this widx's suffix, so it appends (the
    # idempotency key is THIS process's constant widx).
    assert per_worker_root("/data/root_w0", 1) == "/data/root_w0_w1"
