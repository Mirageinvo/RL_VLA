"""Single LIBERO task instance with step accounting, GR00T-format observations
and snapshot/restore.

Observation processing and gripper conventions replicate
Isaac-GR00T gr00t/eval/sim/LIBERO/libero_env.py (LiberoEnv._process_observation
and LiberoEnv.step): images flipped [::-1, ::-1], state = eef xyz, axis-angle,
gripper qpos(2); model gripper in [0,1] -> env: -sign(2g-1).
Here that conversion happens in the provider, so env.step takes a raw 7-D
LIBERO command (a0_env convention).
"""
from __future__ import annotations

import copy
import math
import os
from collections import deque
from dataclasses import dataclass, field

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from ..budget import StepCounter


def load_init_states_file(path) -> np.ndarray:
    """Load one LIBERO `.pruned_init` file (a pickled numpy array saved with torch.save).

    LIBERO's Benchmark.get_task_init_states calls torch.load(path) without weights_only; since
    PyTorch 2.6 that defaults to weights_only=True and fails on numpy arrays (UnpicklingError,
    numpy.core.multiarray._reconstruct). These files ship with the pinned LIBERO checkout and are
    trusted, so they -- and only they -- are loaded with weights_only=False. No global torch.load change.
    """
    import torch

    path = str(path)
    assert path.endswith(".pruned_init") or path.endswith(".init"), path
    arr = torch.load(path, weights_only=False)
    arr = np.asarray(arr)
    assert arr.ndim == 2 and arr.shape[0] > 0, (path, arr.shape)
    return arr


def load_task_init_states(bm, task_id: int) -> np.ndarray:
    """Same file as bm.get_task_init_states(task_id), loaded via load_init_states_file."""
    from libero.libero import get_libero_path

    t = bm.get_task(task_id)
    return load_init_states_file(os.path.join(get_libero_path("init_states"), t.problem_folder, t.init_states_file))


def quat2axisangle(quat):
    quat = np.array(quat, dtype=np.float64)
    quat[3] = min(max(quat[3], -1.0), 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def process_obs(obs: dict) -> dict:
    xyz = obs["robot0_eef_pos"]
    rpy = quat2axisangle(obs["robot0_eef_quat"])
    return {
        "video": {
            "image": np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]),
            "wrist_image": np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]),
        },
        "state": {
            "x": np.array([xyz[0]], np.float32),
            "y": np.array([xyz[1]], np.float32),
            "z": np.array([xyz[2]], np.float32),
            "roll": np.array([rpy[0]], np.float32),
            "pitch": np.array([rpy[1]], np.float32),
            "yaw": np.array([rpy[2]], np.float32),
            "gripper": np.asarray(obs["robot0_gripper_qpos"], np.float32),
        },
    }


def proprio_vector(pobs: dict) -> np.ndarray:
    s = pobs["state"]
    return np.concatenate([s[k] for k in ("x", "y", "z", "roll", "pitch", "yaw", "gripper")]).astype(np.float32)


# robosuite Observable fields that define WHEN the next sample is taken and WHAT is returned.
# env.step() runs ~25 MuJoCo substeps but each observable samples on the first substep after its
# timer wraps; the timer phase is python state, so restoring only MuJoCo state makes the returned
# proprio come from a different substep (smoke check3: sim state exact, eef pose off by ~1e-3).
OBSERVABLE_FIELDS = ("_time_since_last_sample", "_sampled", "_current_delay", "_current_observed_value")


def observable_state(env) -> dict:
    """Sampling-timer state of every robosuite observable plus the shared observation cache.

    Values are kept by reference, not deep-copied: robosuite 1.4 Observable.update and the obs
    cache always bind NEW arrays (never write in place), and the bank keeps ~500 snapshots alive,
    so copying the camera frames would cost hundreds of MB.
    """
    obs = getattr(env, "_observables", None) or {}
    cache = getattr(env, "_obs_cache", None)
    return {"observables": {n: {k: getattr(o, k) for k in OBSERVABLE_FIELDS if hasattr(o, k)} for n, o in obs.items()},
            "obs_cache": dict(cache) if cache is not None else None}


def set_observable_state(env, st: dict) -> None:
    obs = getattr(env, "_observables", None) or {}
    missing = set(st["observables"]) ^ set(obs)
    if missing:
        raise RuntimeError(f"observable set changed between snapshot and restore: {sorted(missing)}")
    for n, attrs in st["observables"].items():
        for k, v in attrs.items():
            setattr(obs[n], k, v)
    if st["obs_cache"] is not None:
        env._obs_cache = dict(st["obs_cache"])


@dataclass
class Snapshot:
    init_state_id: int
    elapsed: int
    sim_flat: np.ndarray
    mj_full: np.ndarray | None
    py_state: dict
    history: list = field(repr=False)
    success: bool = False
    action_prefix: np.ndarray | None = None  # [elapsed, 7] executed actions since reset (replay fallback)


class LiberoTask:
    ACTION_LOW = -np.ones(7, np.float32)
    ACTION_HIGH = np.ones(7, np.float32)

    def __init__(self, suite: str, task_id: int, counter: StepCounter, episode_cap: int, camera_size: int = 256,
                 settle_steps: int = 0, history_len: int = 1, history_stride: int = 1):
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        bm = benchmark.get_benchmark_dict()[suite]()
        self.task = bm.get_task(task_id)
        self.task_id = task_id
        self.suite = suite
        self.language = self.task.language
        self.init_states = load_task_init_states(bm, task_id)
        bddl = os.path.join(get_libero_path("bddl_files"), self.task.problem_folder, self.task.bddl_file)
        self.env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=camera_size, camera_widths=camera_size,
                                      ignore_done=True)
        self.env.seed(0)
        self.env.reset()
        self.counter = counter
        self.episode_cap = episode_cap
        self.settle_steps = settle_steps
        self.history_len, self.history_stride = history_len, history_stride
        self._hist: deque = deque(maxlen=(history_len - 1) * history_stride + 1)
        self.elapsed = 0
        self.success = False
        self.init_state_id = -1
        self.prefix: list[np.ndarray] = []

    # ---------------- basic control ----------------
    @property
    def n_init_states(self) -> int:
        return len(self.init_states)

    def task_info(self) -> dict:
        return {"suite": self.suite, "task_id": self.task_id, "name": self.task.name, "language": self.language,
                "n_init_states": self.n_init_states}

    def reset(self, init_state_id: int, settle_bucket: str | None = None) -> dict:
        self.env.reset()
        raw = self.env.set_init_state(self.init_states[init_state_id])
        self.init_state_id = init_state_id
        self.elapsed = 0
        self.prefix = []
        self.success = bool(self.env.check_success())
        if self.settle_steps:
            dummy = np.array([0, 0, 0, 0, 0, 0, -1], np.float32)
            with self.counter.charge(settle_bucket or "aborted"):
                for _ in range(self.settle_steps):
                    raw, *_ = self.env.step(dummy)
                    self.counter.on_env_step()
        pobs = process_obs(raw)
        self._hist.clear()
        for _ in range(self._hist.maxlen):
            self._hist.append(pobs)
        return pobs

    def step(self, action7: np.ndarray) -> tuple[dict, bool]:
        """One real env.step. Charged to the counter's current bucket."""
        a = np.asarray(action7, np.float32).reshape(7)
        self.counter.on_env_step()
        raw, _r, _d, _info = self.env.step(a.astype(np.float64))
        self.elapsed += 1
        self.prefix.append(a.copy())
        self.success = bool(self.env.check_success())
        pobs = process_obs(raw)
        self._hist.append(pobs)
        return pobs, self.success

    @property
    def horizon_left(self) -> int:
        return self.episode_cap - self.elapsed

    def policy_observation(self) -> dict:
        """Observation history in Gr00tPolicy nested format, batch=1."""
        frames = list(self._hist)[:: -self.history_stride][: self.history_len][::-1]
        video = {k: np.stack([f["video"][k] for f in frames])[None] for k in frames[-1]["video"]}
        state = {k: np.stack([f["state"][k] for f in frames])[None].astype(np.float32) for k in frames[-1]["state"]}
        return {"video": video, "state": state, "language": self.language}

    def proprio(self) -> np.ndarray:
        return proprio_vector(self._hist[-1])

    # ---------------- snapshot / restore ----------------
    def _robot_objs(self):
        e = self.env.env
        objs = {}
        for i, r in enumerate(e.robots):
            objs[f"robot{i}"] = r
            if getattr(r, "controller", None) is not None:
                objs[f"robot{i}.controller"] = r.controller
            g = getattr(r, "gripper", None)
            if isinstance(g, dict):
                for k, v in g.items():
                    objs[f"robot{i}.gripper.{k}"] = v
            elif g is not None:
                objs[f"robot{i}.gripper"] = g
        return objs

    def _py_state(self) -> dict:
        e = self.env.env
        out = {"env": {k: copy.deepcopy(getattr(e, k)) for k in ("timestep", "cur_time", "done") if hasattr(e, k)},
               "observables": observable_state(e)}
        for name, o in self._robot_objs().items():
            out[name] = {k: copy.deepcopy(v) for k, v in vars(o).items()
                         if isinstance(v, (np.ndarray, float, int, bool, np.floating, np.integer)) and not k.startswith("_")}
        return out

    def _mj_raw(self):
        sim = self.env.env.sim
        m = getattr(sim.model, "_model", None)
        d = getattr(sim.data, "_data", None)
        return m, d

    def snapshot(self, with_prefix: bool = True) -> Snapshot:
        import mujoco

        sim = self.env.env.sim
        m, d = self._mj_raw()
        full = None
        if m is not None and hasattr(mujoco, "mj_getState"):
            spec = mujoco.mjtState.mjSTATE_INTEGRATION
            full = np.empty(mujoco.mj_stateSize(m, spec))
            mujoco.mj_getState(m, d, full, spec)
        return Snapshot(
            init_state_id=self.init_state_id,
            elapsed=self.elapsed,
            sim_flat=np.array(sim.get_state().flatten()),
            mj_full=full,
            py_state=self._py_state(),
            history=list(self._hist),
            success=self.success,
            action_prefix=np.array(self.prefix, np.float32) if with_prefix else None,
        )

    def restore(self, snap: Snapshot) -> dict:
        """Restore MuJoCo integration state + python-side controller/gripper state."""
        import mujoco

        self.env.set_init_state(snap.sim_flat)  # LIBERO bookkeeping (forward, post_process, observables)
        m, d = self._mj_raw()
        if snap.mj_full is not None:
            mujoco.mj_setState(m, d, snap.mj_full, mujoco.mjtState.mjSTATE_INTEGRATION)
            mujoco.mj_forward(m, d)
        e = self.env.env
        for k, v in snap.py_state["env"].items():
            setattr(e, k, copy.deepcopy(v))
        objs = self._robot_objs()
        if "observables" in snap.py_state:  # after set_init_state's forced update, which advances the timers
            set_observable_state(e, snap.py_state["observables"])
        for name, attrs in snap.py_state.items():
            if name in ("env", "observables") or name not in objs:
                continue
            for k, v in attrs.items():
                setattr(objs[name], k, copy.deepcopy(v))
        self.init_state_id = snap.init_state_id
        self.elapsed = snap.elapsed
        self.success = snap.success
        self.prefix = list(snap.action_prefix) if snap.action_prefix is not None else []
        self._hist = deque(snap.history, maxlen=self._hist.maxlen)
        return self._hist[-1]

    def replay_restore(self, snap: Snapshot, bucket: str = "bank_prefix_replay") -> dict:
        """Fallback: reset to the init state and replay the recorded action prefix (charged)."""
        assert snap.action_prefix is not None
        with self.counter.charge(bucket):
            self.reset(snap.init_state_id, settle_bucket=bucket)
            for a in snap.action_prefix:
                self.step(a)
        return self._hist[-1]

    def state_vector(self) -> np.ndarray:
        return np.array(self.env.env.sim.get_state().flatten())

    def close(self):
        self.env.close()
