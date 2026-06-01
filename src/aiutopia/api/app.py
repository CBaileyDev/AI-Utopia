"""FastAPI app factory + routers for the AI Utopia GUI backend.

Design rules (see gui/API_CONTRACT.md):
  * Heavy deps (chromadb, py4j, ray, torch) are imported INSIDE handlers, never
    at module top, so the server boots in <1s and a dead Minecraft server only
    fails the agent routes — /api/health and /api/training/* stay alive with no
    heavy deps loaded.
  * Every handler is wrapped so failures return {ok:false, error:str} JSON via a
    global exception handler; the frontend never sees an HTML 500.
  * CORS is open (allow all) for the Tauri webview (tauri://localhost /
    http://tauri.localhost) during dev.
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from aiutopia.api import schemas
from aiutopia.common.config import Paths, Py4JConfig


# --- single-process training-job tracker ---
class _TrainingJob:
    """Tracks the at-most-one training subprocess the API launched."""

    def __init__(self) -> None:
        self.proc: subprocess.Popen[bytes] | None = None
        self.run_id: str | None = None
        self.backend: str | None = None
        self.max_iters: int = 0
        self.log_path: Path | None = None
        self.pid_path: Path | None = None

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def clear_if_dead(self) -> None:
        if self.proc is not None and self.proc.poll() is not None:
            self.proc = None


_JOB = _TrainingJob()


# --- helpers (no heavy imports) ---
def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _safe_float(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _scalar_field(v: Any) -> float | None:
    """Coerce an obs scalar field (1-elem array or plain number) to float."""
    if isinstance(v, list):
        return _safe_float(v[0]) if v else None
    return _safe_float(v)


def _runs_dir() -> Path:
    return Paths.from_env().runs_dir


def _latest_progress_csv(run_dir: Path) -> Path | None:
    """Newest PPO_*/progress.csv under a run directory, or None."""
    candidates = sorted(
        run_dir.glob("PPO_*/progress.csv"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _run_dir_mtime(run_dir: Path) -> float:
    csv_path = _latest_progress_csv(run_dir)
    return csv_path.stat().st_mtime if csv_path else 0.0


_RETURN_COL = "env_runners/episode_return_mean"
_ENTROPY_COL = "learners/gatherer_policy/entropy"
_KL_COL = "learners/gatherer_policy/mean_kl_loss"
_ITER_COL = "training_iteration"
_SPS_COL = "env_runners/num_env_steps_sampled_lifetime_throughput/throughput_since_last_reduce"


def _read_progress_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _backend_from_run_dir(run_dir: Path) -> str | None:
    """Infer backend from the PPO_* subdir name.

    Ray names it PPO_aiutopia_sim_* for sim and PPO_aiutopia_minecraft_* for real.
    """
    for sub in run_dir.glob("PPO_*"):
        n = sub.name
        if "_sim_" in n:
            return "sim"
        if "_minecraft_" in n:
            return "real"
    return None


def _seed_from_run_id(run_id: str) -> int | None:
    # run_id looks like aiutopia_M1_seed1
    marker = "seed"
    idx = run_id.rfind(marker)
    if idx < 0:
        return None
    try:
        return int(run_id[idx + len(marker) :])
    except ValueError:
        return None


def _run_status(run_dir: Path, run_id: str) -> str:
    """Status: 'running' iff our live job, else 'done' (external PIDs not probed on Win)."""
    if _JOB.is_alive() and _JOB.run_id == run_id:
        return "running"
    return "done"


# --- app factory ---
def create_app() -> FastAPI:
    app = FastAPI(title="AI Utopia GUI backend", version="1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def _unhandled(_req: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=200,
            content={"ok": False, "error": f"{type(exc).__name__}: {exc}"},
        )

    # --- health ---
    @app.get("/api/health")
    def health() -> dict[str, Any]:
        cfg = Py4JConfig.from_env()
        bridge_state = "offline"
        instances = 0
        try:
            from aiutopia.env.bridge import FabricBridge  # lazy

            with FabricBridge(port=cfg.production_port) as bridge:
                if bridge.health() == "ok":
                    bridge_state = "online"
                    instances = 1
        except Exception:
            bridge_state = "offline"
        return schemas.HealthResponse(
            bridge=bridge_state,  # type: ignore[arg-type]
            py4j_port=cfg.production_port,
            instances=instances,
            mc_version="1.21.1",
            server_time=_now_iso(),
        ).model_dump()

    # --- agents ---
    @app.get("/api/agents")
    def list_agents() -> Any:
        from aiutopia.identity.service import IdentityService  # lazy

        paths = Paths.from_env()
        if not paths.identity_db.exists():
            return []
        svc = IdentityService(paths.identity_db)
        living = svc.list_living_agents()
        if not living:
            return []

        # Best-effort: join live obs (x/z/health/hunger) keyed by agent_name.
        obs_by_name: dict[str, dict[str, Any]] = {}
        try:
            from aiutopia.env.bridge import FabricBridge  # lazy

            cfg = Py4JConfig.from_env()
            with FabricBridge(port=cfg.production_port) as bridge:
                if bridge.health() == "ok":
                    raw = bridge.observations_all()
                    for _aid, ob in raw.items():
                        nm = ob.get("agent_name") or _aid
                        obs_by_name[str(nm)] = ob
        except Exception:
            obs_by_name = {}

        out: list[dict[str, Any]] = []
        for a in living:
            ob = obs_by_name.get(a.agent_name, {})
            pos = ob.get("position") or []
            x = _safe_float(pos[0]) if len(pos) >= 1 else None
            z = _safe_float(pos[2]) if len(pos) >= 3 else None
            health_v = ob.get("health")
            hunger_v = ob.get("hunger")
            out.append(
                schemas.AgentModel(
                    id=a.agent_uuid,
                    name=a.agent_name,
                    role=a.role_id,
                    status=a.status,
                    uuid=a.agent_uuid,
                    skin=a.current_skin,
                    born=a.born_at,
                    x=x,
                    z=z,
                    rewards=0.0,  # not tracked in identity.db; stub
                    health=_scalar_field(health_v),
                    hunger=_scalar_field(hunger_v),
                ).model_dump()
            )
        return out

    @app.post("/api/agents/spawn")
    def spawn_agent(req: schemas.SpawnRequest) -> dict[str, Any]:
        from aiutopia.common.ids import memory_id_for, skill_library_id_for  # lazy
        from aiutopia.identity.service import IdentityService, init_identity_db
        from aiutopia.identity.skin_pool import deterministic_skin_for_uuid, pick_name

        paths = Paths.from_env()
        paths.ensure()
        init_identity_db(paths.identity_db, Path("src/aiutopia/identity/migrations"))
        svc = IdentityService(paths.identity_db)

        try:
            role_obj = svc.get_role(req.role)
        except KeyError as e:
            return schemas.SpawnResponse(ok=False, error=str(e)).model_dump()

        living_names = {a.agent_name for a in svc.list_living_agents()}
        chosen_name = req.name or pick_name(role_obj.default_skin_pool, used=living_names)
        agent = svc.spawn_agent(req.role, chosen_name, born_at=int(time.time()))
        skin = deterministic_skin_for_uuid(agent.agent_uuid, role_obj.default_skin_pool)

        # Chroma collections (best-effort, non-fatal if chroma unavailable)
        try:
            from aiutopia.memory.client import open_chroma  # lazy

            chroma = open_chroma(paths.chroma_dir)
            chroma.get_or_create_collection(memory_id_for(agent.agent_uuid))
            chroma.get_or_create_collection(skill_library_id_for(agent.agent_uuid))
        except Exception:
            pass

        # Carpet spawn (best-effort; identity row already persisted)
        carpet_ok = False
        carpet_err: str | None = None
        try:
            from aiutopia.env.bridge import FabricBridge  # lazy

            cfg = Py4JConfig.from_env()
            with FabricBridge(port=cfg.production_port) as bridge:
                if bridge.health() == "ok":
                    carpet_ok = bridge.carpet_spawn(chosen_name, skin=skin, role=req.role)
                else:
                    carpet_err = "Fabric server unhealthy"
        except Exception as e:
            carpet_err = f"{type(e).__name__}: {e}"

        model = schemas.AgentModel(
            id=agent.agent_uuid,
            name=agent.agent_name,
            role=agent.role_id,
            status=agent.status,
            uuid=agent.agent_uuid,
            skin=skin,
            born=agent.born_at,
            rewards=0.0,
        )
        resp = schemas.SpawnResponse(ok=True, agent=model)
        out = resp.model_dump()
        out["carpet_spawned"] = carpet_ok
        if carpet_err:
            out["carpet_error"] = carpet_err
        return out

    @app.post("/api/agents/{uuid}/kill")
    def kill_agent(uuid: str, req: schemas.KillRequest | None = None) -> dict[str, Any]:
        from aiutopia.identity.service import IdentityService  # lazy

        paths = Paths.from_env()
        if not paths.identity_db.exists():
            return schemas.OkResponse(ok=False, error="identity.db not found").model_dump()
        svc = IdentityService(paths.identity_db)
        cause = (req.cause if req else None) or "gui_kill"
        try:
            svc.record_death(uuid, died_at=int(time.time()), cause_of_death=cause)
        except (ValueError, KeyError) as e:
            return schemas.OkResponse(ok=False, error=str(e)).model_dump()
        return schemas.OkResponse(ok=True).model_dump()

    # --- training ---
    @app.get("/api/training/runs")
    def training_runs() -> Any:
        runs_dir = _runs_dir()
        if not runs_dir.exists():
            return []
        out: list[dict[str, Any]] = []
        for run_dir in sorted(runs_dir.glob("aiutopia_*")):
            if not run_dir.is_dir():
                continue
            run_id = run_dir.name
            csv_path = _latest_progress_csv(run_dir)
            iters = 0
            last_return: float | None = None
            if csv_path is not None:
                try:
                    rows = _read_progress_rows(csv_path)
                    if rows:
                        iters = int(_safe_float(rows[-1].get(_ITER_COL)) or len(rows))
                        last_return = _safe_float(rows[-1].get(_RETURN_COL))
                except Exception:
                    pass
            out.append(
                schemas.TrainingRun(
                    run_id=run_id,
                    seed=_seed_from_run_id(run_id),
                    backend=_backend_from_run_dir(run_dir),
                    iters=iters,
                    last_return=last_return,
                    status=_run_status(run_dir, run_id),  # type: ignore[arg-type]
                    path=str(run_dir),
                ).model_dump()
            )
        return out

    @app.get("/api/training/status")
    def training_status() -> dict[str, Any]:
        _JOB.clear_if_dead()
        runs_dir = _runs_dir()
        job_alive = _JOB.is_alive()

        # Pick the run dir to read metrics from: the live job's dir if it has
        # produced a progress.csv, else the most-recently modified run dir.
        target_dir: Path | None = None
        if job_alive and _JOB.run_id:
            cand = runs_dir / _JOB.run_id
            if cand.exists() and _latest_progress_csv(cand):
                target_dir = cand
        if target_dir is None and runs_dir.exists():
            dirs: list[Path] = []
            for d in runs_dir.glob("aiutopia_*"):
                if d.is_dir() and _latest_progress_csv(d):
                    dirs.append(d)
            if dirs:
                target_dir = max(dirs, key=_run_dir_mtime)

        # Live job that hasn't written its first progress.csv row yet: report it
        # as running with iter=0 / empty history (don't fall through to a stale
        # finished run, which would make the GUI think the launch failed).
        if job_alive and (target_dir is None or target_dir.name != _JOB.run_id):
            return schemas.TrainingStatus(
                running=True,
                run_id=_JOB.run_id,
                backend=_JOB.backend,
                iter=0,
                max_iters=_JOB.max_iters,
            ).model_dump()

        if target_dir is None:
            return schemas.TrainingStatus(running=job_alive).model_dump()

        run_id = target_dir.name
        csv_path = _latest_progress_csv(target_dir)
        rows = _read_progress_rows(csv_path) if csv_path else []
        running = job_alive and _JOB.run_id == run_id

        metrics = schemas.TrainingMetrics()
        cur_iter = 0
        sps: float | None = None
        history: list[schemas.HistoryPoint] = []
        if rows:
            last = rows[-1]
            cur_iter = int(_safe_float(last.get(_ITER_COL)) or len(rows))
            sps = _safe_float(last.get(_SPS_COL))
            metrics = schemas.TrainingMetrics(
                return_mean=_safe_float(last.get(_RETURN_COL)),
                entropy=_safe_float(last.get(_ENTROPY_COL)),
                kl=_safe_float(last.get(_KL_COL)),
                clipfrac=None,  # no PPO clip-fraction column in progress.csv
                term_rate=None,
            )
            # last 50 points for a sparkline
            for r in rows[-50:]:
                it = _safe_float(r.get(_ITER_COL))
                history.append(
                    schemas.HistoryPoint(
                        iter=int(it) if it is not None else 0,
                        return_mean=_safe_float(r.get(_RETURN_COL)),
                        entropy=_safe_float(r.get(_ENTROPY_COL)),
                        kl=_safe_float(r.get(_KL_COL)),
                    )
                )

        return schemas.TrainingStatus(
            running=running,
            run_id=run_id,
            backend=_JOB.backend if running else _backend_from_run_dir(target_dir),
            iter=cur_iter,
            max_iters=_JOB.max_iters if running else cur_iter,
            sps=sps,
            metrics=metrics,
            history=history,
        ).model_dump()

    @app.post("/api/training/start")
    def training_start(req: schemas.TrainingStartRequest) -> dict[str, Any]:
        _JOB.clear_if_dead()
        if _JOB.is_alive():
            return schemas.TrainingStartResponse(
                ok=False, error=f"a training run is already active (run_id={_JOB.run_id})"
            ).model_dump()

        paths = Paths.from_env()
        paths.ensure()
        seed = req.seed if req.seed is not None else 1
        run_id = f"aiutopia_M1_seed{seed}"

        research = Path("Research")
        research.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_path = research / f"train-gui-{ts}.log"
        pid_path = research / f"train-gui-{ts}.pid"

        cmd = [
            sys.executable,
            "-u",
            "scripts/train.py",
            "--milestone",
            "M1",
            "--backend",
            req.backend,
            "--max-iters",
            str(req.iters),
            "--seed",
            str(seed),
            "--evaluation-interval",
            "999",  # disable eval by default for GUI-launched runs
        ]
        if req.num_envs is not None:
            cmd += ["--num-env-runners", str(req.num_envs)]
        if req.entropy_coeff is not None:
            cmd += ["--entropy-coeff", str(req.entropy_coeff)]
        if req.spawn_jitter is not None:
            cmd += ["--spawn-jitter", str(req.spawn_jitter)]
        if req.approach_shaping:
            cmd += ["--approach-shaping"]
        if req.force_masked_spawn:
            cmd += ["--force-masked-spawn"]

        env = dict(os.environ)
        env.setdefault("PYTHONPATH", "src")
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("AIUTOPIA_ROOT", str(paths.root))

        try:
            log_fh = log_path.open("wb")
            proc = subprocess.Popen(
                cmd,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=str(Path.cwd()),
            )
        except Exception as e:
            return schemas.TrainingStartResponse(
                ok=False, error=f"failed to launch: {type(e).__name__}: {e}"
            ).model_dump()

        pid_path.write_text(str(proc.pid), encoding="utf-8")
        _JOB.proc = proc
        _JOB.run_id = run_id
        _JOB.backend = req.backend
        _JOB.max_iters = req.iters
        _JOB.log_path = log_path
        _JOB.pid_path = pid_path

        return schemas.TrainingStartResponse(ok=True, pid=proc.pid, run_id=run_id).model_dump()

    @app.post("/api/training/stop")
    def training_stop() -> dict[str, Any]:
        _JOB.clear_if_dead()
        if not _JOB.is_alive() or _JOB.proc is None:
            return schemas.OkResponse(ok=False, error="no active training run").model_dump()
        try:
            _JOB.proc.terminate()
            try:
                _JOB.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                _JOB.proc.kill()
        except Exception as e:
            return schemas.OkResponse(ok=False, error=f"{type(e).__name__}: {e}").model_dump()
        _JOB.proc = None
        return schemas.OkResponse(ok=True).model_dump()

    # --- rewards ---
    @app.get("/api/rewards")
    def get_rewards() -> dict[str, Any]:
        from aiutopia.env.reward import load_reward_config, reward_config_to_serializable

        cfg = reward_config_to_serializable(load_reward_config())
        return schemas.RewardsConfig(**cfg).model_dump()

    @app.put("/api/rewards")
    async def put_rewards(req: Request) -> dict[str, Any]:
        from aiutopia.env.reward import (  # lazy
            build_default_reward_config,
            deep_merge_reward_config,
            load_reward_config,
            reward_config_path,
            reward_config_to_serializable,
        )

        try:
            overlay = await req.json()
        except Exception as e:
            resp = schemas.RewardsPutResponse(ok=False, error=f"invalid JSON body: {e}")
            return resp.model_dump()
        if not isinstance(overlay, dict):
            resp = schemas.RewardsPutResponse(ok=False, error="body must be a JSON object")
            return resp.model_dump()

        # Merge the partial overlay over the CURRENT on-disk config (so repeated
        # partial PUTs accumulate), then persist the full merged result.
        current = load_reward_config()
        merged = deep_merge_reward_config(current, overlay)
        # Guard: only persist keys we recognize (defends against junk top-levels)
        allowed = set(build_default_reward_config().keys())
        merged = {k: v for k, v in merged.items() if k in allowed}
        serial = reward_config_to_serializable(merged)

        path = reward_config_path()
        if path is None:
            err = "could not resolve reward config path"
            return schemas.RewardsPutResponse(ok=False, error=err).model_dump()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(serial, indent=2), encoding="utf-8")
        return schemas.RewardsPutResponse(ok=True, saved_path=str(path)).model_dump()

    # --- logs ---
    @app.get("/api/logs")
    def logs(tail: int = 200) -> Any:
        out: list[dict[str, Any]] = []
        # Pick the GUI job's log if active, else the newest train-*.log
        log_path: Path | None = _JOB.log_path if _JOB.log_path and _JOB.log_path.exists() else None
        if log_path is None:
            research = Path("Research")
            if research.exists():
                candidates = sorted(
                    research.glob("train*.log"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if candidates:
                    log_path = candidates[0]
        if log_path is not None and log_path.exists():
            try:
                lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            except Exception:
                lines = []
            ts = _now_iso()
            for line in lines[-max(1, tail) :]:
                out.append(schemas.LogEntry(ts=ts, type="TRAIN", message=line).model_dump())
        if not out:
            out.append(
                schemas.LogEntry(
                    ts=_now_iso(),
                    type="SYSTEM",
                    message="no training log available yet",
                ).model_dump()
            )
        return out

    return app


app = create_app()
