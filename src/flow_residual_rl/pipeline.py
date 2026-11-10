"""Shared setup used by all entry points (one place, one implementation)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .basis import ResidualMap, build_fixed_basis, load_basis
from .budget import StepCounter
from .config import collect_manifest, read_json, splits, write_json
from .envs.chunk_env import ObsNormalizer


def runs_dir(cfg) -> Path:
    return Path(cfg["runs_dir"])


def task_dir(cfg, task_id) -> Path:
    return runs_dir(cfg) / f"{cfg['suite']}_task{task_id}"


def make_provider(cfg, counter=None):
    from .providers.gr00t import Gr00tProvider

    return Gr00tProvider(cfg["checkpoint_dir"], cfg["embodiment_tag"], device=cfg["device"], dtype=cfg["model_dtype"],
                         chunk=cfg["execution_horizon"], context_dim=cfg["context_dim"],
                         projection_path=runs_dir(cfg) / "shared" / "context_projection.npy",
                         projection_seed=cfg["context_projection_seed"], counter=counter)


def make_task(cfg, task_id, counter, provider):
    from .envs.libero_task import LiberoTask

    hl, hs = provider.history_spec() if provider is not None else (1, 1)
    return LiberoTask(cfg["suite"], task_id, counter, cfg["episode_cap"], cfg["camera_size"], cfg["settle_steps"], hl, hs)


def task_splits(cfg, task) -> dict:
    p = task_dir(cfg, task.task_id) / "splits.json"
    s = read_json(p)
    if s is None:
        s = splits(cfg, task.n_init_states)
        s["task"] = task.task_info()
        write_json(p, s)
    return s


def n_dims(cfg) -> int:
    return len(cfg["corrected_dims"])


def basis_for_arm(cfg, arm: str, task_id: int) -> np.ndarray:
    C, nd, r = cfg["execution_horizon"], n_dims(cfg), cfg["rank"]
    if arm in ("outcome12", "pca12"):
        return load_basis(task_dir(cfg, task_id) / "basis" / arm)
    return build_fixed_basis(arm, C, nd, r if not arm.startswith("full") else C * nd, cfg["random_basis_seed"])


def residual_map(cfg, basis, task) -> ResidualMap:
    return ResidualMap(basis, cfg["execution_horizon"], tuple(cfg["corrected_dims"]), cfg["epsilon_rms"],
                       task.ACTION_LOW, task.ACTION_HIGH)


def obs_dim(cfg) -> int:
    return cfg["context_dim"] + 8 + cfg["execution_horizon"] * 7 + 1


def load_normalizer(cfg, task_id, allow_identity=False) -> ObsNormalizer:
    d = read_json(task_dir(cfg, task_id) / "bank" / "obs_norm.json")
    if d is None:
        if allow_identity:
            return ObsNormalizer.identity(obs_dim(cfg))
        raise FileNotFoundError("obs_norm.json missing: collect the bank first (normalisation is fit on its train part)")
    return ObsNormalizer.from_dict(d)


def write_manifest(cfg, path: Path, provider=None, extra=None):
    e = dict(extra or {})
    if provider is not None:
        e["provider"] = provider.metadata()
        e["frozen_check"] = provider.check_frozen()
        e["checkpoint_revision"] = _checkpoint_revision(cfg["checkpoint_dir"])
    write_json(path, collect_manifest(cfg, e))


def _checkpoint_revision(ckpt_dir) -> dict:
    p = Path(ckpt_dir)
    out = {"path": str(p.resolve()) if p.exists() else str(p)}
    for f in ["config.json", "model.safetensors.index.json", "processor_config.json", "statistics.json"]:
        q = p / f
        if q.exists():
            import hashlib

            out[f + ".sha256"] = hashlib.sha256(q.read_bytes()).hexdigest()
    # huggingface-cli download into --local-dir leaves .cache/huggingface/download/*.metadata with the commit hash
    meta = list((p.parent / ".cache" / "huggingface" / "download").rglob("*.metadata")) if p.exists() else []
    if meta:
        out["hf_commit"] = meta[0].read_text().splitlines()[0]
    return out


def counter_for(path: Path) -> StepCounter:
    return StepCounter(path)
