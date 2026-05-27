"""Section 5.10 promotion service — copies weights + bumps roles.policy_version."""
from __future__ import annotations

import shutil
import sqlite3
import time
from pathlib import Path

from aiutopia.common.config import Paths
from aiutopia.identity.service import IdentityService


def promote_weights(*,
                     role_id:        str,
                     checkpoint_dir: Path,
                     paths:          Paths,
                     notes:          str = "",
                     deployed_by:    str = "manual:cli") -> dict:
    svc = IdentityService(paths.identity_db)
    role = svc.get_role(role_id)
    from_version = role.policy_version
    to_version   = from_version + 1

    target_dir = paths.weights_dir / role_id / f"v{to_version}"
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(checkpoint_dir, target_dir)

    with sqlite3.connect(paths.identity_db) as conn:
        conn.execute(
            "UPDATE roles SET policy_weights_path=?, policy_version=? WHERE role_id=?",
            (str(target_dir), to_version, role_id),
        )
        cur = conn.execute(
            """INSERT INTO policy_deployments
                  (role_id, from_version, to_version, deployed_at, deployed_by, notes)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (role_id, from_version, to_version,
             int(time.time()), deployed_by, notes),
        )
        deployment_id = cur.lastrowid
    return {
        "role_id":       role_id,
        "from_version":  from_version,
        "to_version":    to_version,
        "weights_path":  str(target_dir),
        "deployment_id": deployment_id,
    }
