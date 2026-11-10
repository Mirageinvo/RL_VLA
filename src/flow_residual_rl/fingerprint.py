"""Experiment-contract fingerprints for every cached / resumable artefact.

Each stage directory (smoke check, bank, basis, base_eval, cell) stores
`fingerprint.json`. Before a cached result is returned or a run is resumed, the
current fingerprint is compared with the stored one; any difference in a
contract field raises ContractMismatch (start a new run directory instead of a
silent resume). Fields listed in cfg["fingerprint_ignore"] are reported as
warnings only (e.g. `code.*` after a verified cosmetic fix).

Deliberately NOT in the fingerprint (allowed to change between launches):
online-step targets, block sizes, eval-episode counts, wall-hour ceilings,
logging/checkpoint cadence.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .config import read_json, write_json

SRC = Path(__file__).resolve().parent

COMMON_KEYS = ("suite", "checkpoint_dir", "embodiment_tag", "model_dtype", "execution_horizon", "episode_cap",
               "settle_steps", "camera_size", "context_dim", "context_projection_seed", "split_seed",
               "split_fractions", "restore_mode", "restore_atol", "restore_rtol")
ACTION_KEYS = ("rank", "corrected_dims", "epsilon_rms", "random_basis_seed")
BANK_KEYS = ("bank_budget_steps", "bank_max_steps_with_rescue", "bank_anchor_target", "bank_pool_max_steps",
             "bank_branch_horizon", "bank_val_every", "bank_seed", "bank_max_restore_failures")
BASIS_KEYS = ("basis_train_updates", "basis_lr", "basis_weight_decay", "basis_pair_weight", "basis_batch_groups",
              "basis_hidden", "basis_eval_every", "basis_min_discordant_train", "basis_min_discordant_val")
SAC_KEYS = ("sac_hidden", "sac_batch_size", "sac_lr", "sac_buffer_size", "sac_learning_starts_chunks",
            "sac_train_freq_chunks", "sac_gradient_steps", "sac_gamma_chunk", "sac_tau", "sac_ent_coef",
            "sac_log_std_bias_init", "warmup_radius")

# source files that define the action / observation / restore / reward contract
CODE = {
    "common": ("envs/libero_task.py", "providers/gr00t.py", "envs/chunk_env.py", "restore.py", "basis.py"),
    "bank": ("bank.py",),
    "basis": ("outcome_predictor.py",),
    "cell": ("train_sac.py", "evaluate.py"),
}


class ContractMismatch(RuntimeError):
    pass


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def sha_file(p) -> str | None:
    p = Path(p)
    return sha_bytes(p.read_bytes()) if p.exists() else None


def sha_array(a) -> str:
    a = np.ascontiguousarray(a)
    return sha_bytes(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())


def sha_obj(o) -> str:
    return sha_bytes(json.dumps(o, sort_keys=True, default=str).encode())


def code_hashes(groups) -> dict:
    return {f"code.{f}": sha_file(SRC / f) for g in groups for f in CODE[g]}


def _cfg(cfg, keys):
    return {f"cfg.{k}": cfg.get(k) for k in keys}


def checkpoint_hash(cfg) -> str:
    """config/statistics files + sha256 of every weight file + backbone HF revisions (not the path)."""
    from .pipeline import _checkpoint_revision, weights_cache_path

    rev = _checkpoint_revision(cfg["checkpoint_dir"], weights_cache_path(cfg))
    return sha_obj({k: v for k, v in rev.items() if k != "path"})


def common(cfg, task_id, provider=None, splits=None) -> dict:
    fp = {"task_id": task_id, **_cfg(cfg, COMMON_KEYS), **code_hashes(["common"]),
          "checkpoint": checkpoint_hash(cfg)}
    if provider is not None:
        md = provider.metadata()
        fp["provider"] = sha_obj({k: v for k, v in md.items() if k != "hook_shape"})
        if hasattr(provider, "projection_digest"):  # the projection matrix actually in use
            fp["context_projection"] = provider.projection_digest()
    if splits is not None:
        fp["splits"] = sha_obj({k: splits[k] for k in ("train", "validation", "evaluation")})
    return fp


def bank_fp(cfg, task_id, provider, splits) -> dict:
    return {**common(cfg, task_id, provider, splits), **_cfg(cfg, ACTION_KEYS + BANK_KEYS), **code_hashes(["bank"])}


def basis_fp(cfg, task_id, bank_dir: Path) -> dict:
    return {"task_id": task_id, "bank_fingerprint": sha_file(Path(bank_dir) / "fingerprint.json"),
            "bank_npz": sha_file(Path(bank_dir) / "bank.npz"), "obs_norm": sha_file(Path(bank_dir) / "obs_norm.json"),
            **_cfg(cfg, ACTION_KEYS + BASIS_KEYS), **code_hashes(["basis"])}


def base_eval_fp(cfg, task_id, provider, splits) -> dict:
    return {**common(cfg, task_id, provider, splits), **_cfg(cfg, ACTION_KEYS)}


def cell_fp(cfg, task_id, provider, splits, arm, seed, basis, norm_dict) -> dict:
    return {**common(cfg, task_id, provider, splits), **_cfg(cfg, ACTION_KEYS + SAC_KEYS), **code_hashes(["cell"]),
            "arm": arm, "seed": int(seed), "basis": sha_array(np.asarray(basis, np.float32)),
            "obs_norm": sha_obj(norm_dict)}


def smoke_fp(cfg, task_id, provider, splits) -> dict:
    return {**common(cfg, task_id, provider, splits), **_cfg(cfg, ACTION_KEYS + SAC_KEYS + ("smoke_env_step_limit",)),
            **code_hashes(["bank", "cell"]), "code.smoke": sha_file(SRC.parents[1] / "scripts" / "smoke.py")}


def diff(old: dict, new: dict) -> dict:
    keys = set(old) | set(new)
    return {k: {"stored": old.get(k), "current": new.get(k)} for k in sorted(keys) if old.get(k) != new.get(k)}


def _ignored(key: str, ignore) -> bool:
    return any(key == i or (i.endswith("*") and key.startswith(i[:-1])) for i in ignore)


def check_or_write(directory: Path, fp: dict, cfg, log=print, name: str = "fingerprint.json") -> dict:
    """First call writes the fingerprint; later calls must match it (except ignored keys)."""
    p = Path(directory) / name
    old = read_json(p)
    if old is None:
        write_json(p, fp)
        return {}
    d = diff(old, fp)
    hard = {k: v for k, v in d.items() if not _ignored(k, cfg.get("fingerprint_ignore") or [])}
    if hard:
        raise ContractMismatch(
            f"{p}: the experiment contract changed since this artefact was produced; resuming/reusing it would "
            f"mix experiments. Use a new runs_dir (or delete the directory), or, only for a verified no-op change, "
            f"list the key in fingerprint_ignore. Differences: {json.dumps(hard, indent=1, default=str)}")
    if d:
        log(f"[fingerprint] {p}: ignored differences {sorted(d)}")
    return d
