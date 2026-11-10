"""The one restore path used everywhere (smoke checks, bank branches), selected by
cfg["restore_mode"]: "snapshot" (MuJoCo integration state + python-side
controller/gripper state) or "replay" (reset to the init state and replay the
recorded action prefix, charged to `replay_bucket`).

`compare` applies atol + rtol element-wise (np.allclose semantics) to every
component; tolerances come from the config and are never loosened in code.
"""
from __future__ import annotations

import numpy as np

RESTORE_MODES = ("snapshot", "replay")


def restore_to(task, snap, mode: str, replay_bucket: str):
    assert mode in RESTORE_MODES, mode
    if mode == "snapshot":
        return task.restore(snap)
    return task.replay_restore(snap, bucket=replay_bucket)


def tolerances(cfg) -> tuple[float, float]:
    return float(cfg["restore_atol"]), float(cfg["restore_rtol"])


def compare(a, b, atol: float, rtol: float) -> dict:
    """b is the reference. ok iff |a-b| <= atol + rtol*|b| for every element (and shapes match)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.shape != b.shape:
        return {"ok": False, "max_abs_err": float("inf"), "shape": [list(a.shape), list(b.shape)]}
    err = np.abs(a - b)
    viol = err - (atol + rtol * np.abs(b))
    return {"ok": bool(np.all(viol <= 0)), "max_abs_err": float(err.max()) if err.size else 0.0,
            "max_violation": float(viol.max()) if viol.size else float("-inf")}
