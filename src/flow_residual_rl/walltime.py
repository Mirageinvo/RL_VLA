"""Project-wide wall-time ledger.

Every entry point (smoke, eval_base, collect_bank, train_basis, run_pilot) opens one
attempt record under runs/wall/ and a daemon thread refreshes its `last` timestamp
every `heartbeat_s` seconds (plus on clean exit). Each process writes only its own
file, so parallel commands do not race. A crashed / killed attempt is counted up to
its last heartbeat: at most `heartbeat_s` of wall time per hard kill is unaccounted.

Ceilings (run_pilot): Stage A stops when the project total reaches
stage_a_gpu_hours (preparation commands included); Stage B when it reaches
project_gpu_hours. One clock for the whole project.
"""
from __future__ import annotations

import atexit
import json
import os
import threading
import time
from pathlib import Path

from .budget import atomic_write_json

HEARTBEAT_S = 60.0


def ledger_dir(runs_dir) -> Path:
    return Path(runs_dir) / "wall"


def project_hours(runs_dir) -> float:
    tot = 0.0
    for f in ledger_dir(runs_dir).glob("*.json"):
        try:
            d = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        tot += max(0.0, d["last"] - d["start"])
    return tot / 3600.0


def breakdown(runs_dir) -> dict:
    out = {}
    for f in ledger_dir(runs_dir).glob("*.json"):
        d = json.loads(f.read_text())
        out[d["command"]] = out.get(d["command"], 0.0) + max(0.0, d["last"] - d["start"]) / 3600.0
    return out


class WallLedger:
    def __init__(self, runs_dir, command: str, heartbeat_s: float = HEARTBEAT_S):
        self.start = time.time()
        self.rec = {"command": command, "pid": os.getpid(), "start": self.start, "last": self.start, "finished": False}
        self.path = ledger_dir(runs_dir) / f"{command}_{int(self.start)}_{os.getpid()}.json"
        self.runs_dir, self.heartbeat_s = runs_dir, heartbeat_s
        self._stop = threading.Event()
        self.beat()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        atexit.register(self.close)

    def beat(self, finished: bool = False) -> None:
        self.rec["last"] = time.time()
        self.rec["finished"] = finished
        atomic_write_json(self.path, self.rec)

    def _loop(self):
        while not self._stop.wait(self.heartbeat_s):
            self.beat()

    def close(self):
        if not self._stop.is_set():
            self._stop.set()
            self.beat(finished=True)

    def project_hours(self) -> float:
        self.beat()
        return project_hours(self.runs_dir)
