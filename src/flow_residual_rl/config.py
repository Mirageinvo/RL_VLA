from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


def load_config(path: str, overrides: list[str] | None = None) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    for ov in overrides or []:
        k, v = ov.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    cfg["_config_path"] = str(path)
    return cfg


def base_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--config", default="configs/pilot.yaml")
    p.add_argument("--set", nargs="*", default=[], help="key=value overrides (YAML values)")
    return p


def splits(cfg: dict, n_states: int) -> dict:
    """Fixed permutation of init-state ids with split_seed -> train/val/eval."""
    rng = np.random.default_rng(cfg["split_seed"])
    perm = rng.permutation(n_states).tolist()
    f_tr, f_va, _ = cfg["split_fractions"]
    n_tr = int(round(f_tr * n_states))
    n_va = int(round(f_va * n_states))
    out = {"train": perm[:n_tr], "validation": perm[n_tr : n_tr + n_va], "evaluation": perm[n_tr + n_va :]}
    assert all(len(v) > 0 for v in out.values()), out
    out["n_unique_states"] = n_states
    return out


def derive_seed(*parts) -> int:
    """Stable 63-bit seed from arbitrary parts (strings/ints)."""
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:8], "little") & ((1 << 63) - 1)


def _git(cmd, cwd):
    try:
        return subprocess.check_output(["git", *cmd], cwd=cwd, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def collect_manifest(cfg: dict, extra: dict | None = None) -> dict:
    import importlib

    pk = {}
    for name in ["torch", "numpy", "stable_baselines3", "gymnasium", "mujoco", "robosuite", "transformers", "libero", "gr00t"]:
        try:
            m = importlib.import_module(name)
            pk[name] = getattr(m, "__version__", "unknown")
            if name in ("gr00t", "libero"):
                pk[name + "_path"] = str(Path(m.__file__).resolve().parent)
                pk[name + "_git_sha"] = _git(["rev-parse", "HEAD"], Path(m.__file__).resolve().parent)
        except Exception as e:  # noqa: BLE001
            pk[name] = f"not importable: {type(e).__name__}"
    hw = {"python": sys.version, "platform": platform.platform(), "hostname": platform.node()}
    try:
        import torch

        if torch.cuda.is_available():
            hw["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            hw["cuda"] = torch.version.cuda
            hw["capability"] = list(torch.cuda.get_device_capability(0))
            hw["bf16_native"] = torch.cuda.is_bf16_supported(including_emulation=False)
    except Exception:
        pass
    repo = Path(__file__).resolve().parents[2]
    m = {
        "project_git_sha": _git(["rev-parse", "HEAD"], repo),
        "project_git_dirty": bool(_git(["status", "--porcelain"], repo)),
        "packages": pk,
        "hardware": hw,
        "env": {k: os.environ.get(k) for k in ["MUJOCO_GL", "PYOPENGL_PLATFORM", "CUDA_VISIBLE_DEVICES", "HF_HOME"]},
        "resolved_config": {k: v for k, v in cfg.items()},
    }
    if extra:
        m.update(extra)
    return m


def write_json(path, obj):
    from .budget import atomic_write_json

    atomic_write_json(Path(path), obj)


def read_json(path, default=None):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else default
