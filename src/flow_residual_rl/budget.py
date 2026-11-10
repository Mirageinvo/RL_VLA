"""Env-step accounting. Every real env.step is charged to exactly one bucket.

The counter persists to cost.json atomically and is never reset on retry:
on construction it loads the previous totals from disk.

Durability: with a path, the counter is flushed to disk every `flush_every`
charged steps, and callers save it in `finally` blocks. A clean exception
therefore loses nothing; a SIGKILL / OOM-kill can lose at most
`flush_every - 1` steps of the interrupted, uncommitted portion. Committed work
(a finished SAC block, an eval file, a cached episode) is always saved before
the commit marker, and `reconcile` books the excess of an interrupted portion
into the matching `aborted_*` bucket on the next start (never resetting it).

Aborted steps keep their origin (`aborted_bank`, `aborted_online`,
`aborted_eval`), so they keep counting toward the corresponding hard limit.
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
    "aborted_bank",
    "aborted_online",
    "aborted_eval",
    "aborted",
)
ABORTED = ("aborted_bank", "aborted_online", "aborted_eval", "aborted")
DEFAULT_FLUSH_EVERY = 200


class BudgetExceeded(RuntimeError):
    pass


def atomic_write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, default=float))
    os.replace(tmp, path)


class StepCounter:
    def __init__(self, path: Path | None = None, flush_every: int = DEFAULT_FLUSH_EVERY):
        self.path = Path(path) if path else None
        self.flush_every = int(flush_every)
        self._unsaved = 0
        self.events: list[dict] = []
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
            self.events = d.get("events", [])
            if "aborted" in d.get("steps", {}) and "aborted_bank" not in d.get("steps", {}):
                self.events.append({"note": "legacy cost.json: 'aborted' has no origin"})
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
        self._unsaved += n
        if self.path is not None and self._unsaved >= self.flush_every:
            self.save()

    def move(self, src: str, dst: str, n: int, reason: str) -> None:
        """Re-book n already-charged steps (e.g. interrupted work -> aborted_*). Never deletes steps."""
        assert src in BUCKETS and dst in BUCKETS and 0 <= n <= self.steps[src], (src, dst, n)
        if n:
            self.steps[src] -= n
            self.steps[dst] += n
            self.events.append({"t": time.time(), "move": [src, dst, int(n)], "reason": reason})

    def reconcile(self, bucket: str, committed: int, aborted_bucket: str, reason: str) -> int:
        """`committed` = steps of `bucket` that belong to committed results. Excess -> aborted_bucket.
        A deficit (steps lost by a hard kill after the last flush) is added back, since the committed
        results prove those steps happened; returns the signed correction."""
        diff = self.steps[bucket] - int(committed)
        if diff > 0:
            self.move(bucket, aborted_bucket, diff, reason)
        elif diff < 0:
            self.steps[bucket] -= diff
            self.events.append({"t": time.time(), "unflushed_committed_steps_restored": [bucket, int(-diff)],
                                "reason": reason})
        if diff:
            self.save()
        return diff

    def on_base_call(self, gpu_s: float = 0.0) -> None:
        self.base_calls += 1
        self.gpu_seconds += gpu_s

    def to_dict(self) -> dict:
        now = time.time()
        wall = self.wall_time + (now - self._t0)
        bank = sum(self.steps[b] for b in BUCKETS if b.startswith("bank_"))
        return {
            "steps": dict(self.steps),
            "bank_total": bank + self.steps["aborted_bank"],  # physical bank cost incl. crashed attempts
            "bank_committed": bank,
            "aborted_total": sum(self.steps[b] for b in ABORTED),
            "total": self.total(),
            "base_calls": self.base_calls,
            "wall_time_s": wall,
            "gpu_seconds": self.gpu_seconds,
            "gpu_hours": self.gpu_seconds / 3600.0,
            "flush_every": self.flush_every,
            "max_unflushed_loss_on_hard_kill": max(self.flush_every - 1, 0),
            "events": self.events[-200:],
        }

    def save(self) -> None:
        if self.path is None:
            return
        d = self.to_dict()
        atomic_write_json(self.path, d)
        self.wall_time, self._t0 = d["wall_time_s"], time.time()
        self._unsaved = 0
