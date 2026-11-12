"""Shared setup used by all entry points (one place, one implementation)."""
from __future__ import annotations

import json
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
        e["checkpoint_revision"] = _checkpoint_revision(cfg["checkpoint_dir"], weights_cache_path(cfg))
    write_json(path, collect_manifest(cfg, e))


WEIGHT_INDEXES = ("model.safetensors.index.json", "pytorch_model.bin.index.json")
WEIGHT_SINGLE = ("model.safetensors", "pytorch_model.bin")
BACKBONE_REPOS = ("nvidia/Cosmos-Reason2-2B",)


def _sha256_stream(path: Path, block: int = 1 << 24) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def weights_identity(ckpt_dir, cache_path: Path | None = None) -> dict:
    """sha256 of the model weight files that from_pretrained actually loads: the shards listed in
    model.safetensors.index.json (or a single model.safetensors / pytorch_model.bin). Training leftovers
    (global_step*/ optimizer states, rng_state_*.pth, scheduler.pt, training_args.bin) are not hashed.
    Full hashes are computed once and cached by (size, mtime_ns); a modified file is re-hashed."""
    p = Path(ckpt_dir)
    cache = read_json(cache_path, {}) if cache_path else {}
    out, changed = {}, False
    files = []
    for idx in WEIGHT_INDEXES:
        if (p / idx).exists():
            files = sorted({p / v for v in json.loads((p / idx).read_text())["weight_map"].values()})
            break
    if not files:
        files = [p / f for f in WEIGHT_SINGLE if (p / f).exists()]
    for f in files:
        if not f.exists():
            raise FileNotFoundError(f"weight shard listed in the index is missing: {f}")
        st = f.stat()
        key = str(f.resolve())
        c = cache.get(key)
        if not c or c["size"] != st.st_size or c["mtime_ns"] != st.st_mtime_ns:
            c = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": _sha256_stream(f)}
            cache[key], changed = c, True
        out[str(f.relative_to(p))] = c["sha256"]
    if cache_path and changed:
        write_json(cache_path, cache)
    return out


def _hf_hub_cache() -> Path:
    import os

    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"


def backbone_revisions() -> dict:
    """Commit of the HF backbone snapshots GR00T fetches at construction (refs/main of the hub cache)."""
    out = {}
    for repo in BACKBONE_REPOS:
        ref = _hf_hub_cache() / ("models--" + repo.replace("/", "--")) / "refs" / "main"
        out[repo] = ref.read_text().strip() if ref.exists() else None
    return out


def _checkpoint_revision(ckpt_dir, cache_path: Path | None = None) -> dict:
    p = Path(ckpt_dir)
    out = {"path": str(p.resolve()) if p.exists() else str(p)}
    out["weights_sha256"] = weights_identity(p, cache_path) if p.exists() else {}
    out["backbone_revisions"] = backbone_revisions()
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


def weights_cache_path(cfg) -> Path:
    return runs_dir(cfg) / "shared" / "weights_sha256_cache.json"


def counter_for(path: Path) -> StepCounter:
    return StepCounter(path)
