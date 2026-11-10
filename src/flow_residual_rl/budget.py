"""Env-step accounting. Every real env.step is charged to exactly one bucket.

The counter persists to cost.json atomically and is never reset on retry:
on construction it loads the previous totals from disk.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

BUCKETS = (
    "bank_pool",
    "bank_prefix_replay",
    "bank_branch",
    "bank_settling",
    "online_train",
    "online_settling",
    "evaluation",
    "eval_settling",
    "smoke",
    "aborted",
)


class BudgetExceeded(RuntimeError):
    pass


def atomic_write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, default=float))
    os.replace(tmp, path)


class StepCounter:
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else None
        self.steps = {b: 0 for b in BUCKETS}
        self.base_calls = 0
        self.wall_time = 0.0
        self.gpu_seconds = 0.0
        self.bucket = "aborted"
        self.limits: dict[tuple[str, ...], int] = {}
        if self.path and self.path.exists():
            d = json.loads(self.path.read_text())
            self.steps.update(d.get("steps", {}))
            self.base_calls = d.get("base_calls", 0)
            self.wall_time = d.get("wall_time_s", 0.0)
            self.gpu_seconds = d.get("gpu_seconds", 0.0)
        self._t0 = time.time()

    @contextmanager
    def charge(self, bucket: str):
        assert bucket in BUCKETS, bucket
        prev, self.bucket = self.bucket, bucket
        try:
            yield
        finally:
            self.bucket = prev

    def set_limit(self, buckets: tuple[str, ...], limit: int) -> None:
        """Hard limit over the *sum* of buckets (e.g. all bank_* buckets)."""
        self.limits[tuple(buckets)] = int(limit)

    def total(self, buckets=None) -> int:
        return sum(self.steps[b] for b in (buckets or BUCKETS))

    def remaining(self, buckets: tuple[str, ...]) -> int:
        lim = self.limits.get(tuple(buckets))
        return (lim - self.total(buckets)) if lim is not None else 1 << 62

    def on_env_step(self, n: int = 1) -> None:
        for bs, lim in self.limits.items():
            if self.bucket in bs and self.total(bs) + n > lim:
                raise BudgetExceeded(f"{bs} limit {lim} reached")
        self.steps[self.bucket] += n

    def on_base_call(self, gpu_s: float = 0.0) -> None:
        self.base_calls += 1
        self.gpu_seconds += gpu_s

    def to_dict(self) -> dict:
        now = time.time()
        wall = self.wall_time + (now - self._t0)
        bank = sum(self.steps[b] for b in BUCKETS if b.startswith("bank_"))
        return {
            "steps": dict(self.steps),
            "bank_total": bank,
            "total": self.total(),
            "base_calls": self.base_calls,
            "wall_time_s": wall,
            "gpu_seconds": self.gpu_seconds,
            "gpu_hours": self.gpu_seconds / 3600.0,
        }

    def save(self) -> None:
        if self.path is None:
            return
        d = self.to_dict()
        atomic_write_json(self.path, d)
        self.wall_time, self._t0 = d["wall_time_s"], time.time()
