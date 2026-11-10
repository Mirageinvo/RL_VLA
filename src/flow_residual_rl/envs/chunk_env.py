"""Gymnasium env whose step is one action chunk (plan sec. 12).

step(u):
  1. delta/w from the fixed ResidualMap; a_exec = clip(a0 + delta) (arm only, gripper unchanged)
  2. execute up to C real env.steps, stopping on success or benchmark horizon
  3. if the episode continues, query the frozen VLA on the *actual* new observation
  4. reward = 1 only on first success; success and horizon exhaustion are terminal
     (horizon exhaustion = terminal failure, zero bootstrap). truncated is never set here:
     the global training-budget stop is handled outside the env.
Non-terminal transitions always contain exactly C env steps (asserted).
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from ..basis import ResidualMap
from ..config import derive_seed


class ObsNormalizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = np.asarray(mean, np.float32)
        self.std = np.maximum(np.asarray(std, np.float32), 1e-3)

    @classmethod
    def identity(cls, dim: int):
        return cls(np.zeros(dim), np.ones(dim))

    @classmethod
    def fit(cls, x: np.ndarray):
        return cls(x.mean(0), x.std(0))

    def __call__(self, v):
        return ((v - self.mean) / self.std).astype(np.float32)

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(np.array(d["mean"]), np.array(d["std"]))


def raw_obs_vector(proposal, elapsed: int, cap: int) -> np.ndarray:
    return np.concatenate([
        proposal.context, proposal.proprio, proposal.a0_env.reshape(-1),
        np.array([1.0 - elapsed / cap], np.float32),
    ]).astype(np.float32)


class EpisodeSchedule:
    """Deterministic sequence of (init_state_id, episode_key). Identical across arms."""

    def __init__(self, state_ids, stream: str, seed: int, start: int = 0, shuffle: bool = True):
        self.state_ids = list(state_ids)
        self.stream, self.seed, self.idx, self.shuffle = stream, seed, start, shuffle

    def get(self, i: int):
        n = len(self.state_ids)
        epoch, pos = divmod(i, n)
        order = np.random.default_rng(derive_seed("sched", self.stream, self.seed, epoch)).permutation(n) if self.shuffle else np.arange(n)
        return self.state_ids[int(order[pos])], f"{self.stream}/{self.seed}/{i}"

    def next(self):
        out = self.get(self.idx)
        self.idx += 1
        return out


class ChunkEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, task, provider, rmap: ResidualMap, normalizer: ObsNormalizer, schedule: EpisodeSchedule,
                 bucket: str, settle_bucket: str, log_fn=None):
        super().__init__()
        self.task, self.provider, self.rmap, self.norm = task, provider, rmap, normalizer
        self.schedule, self.bucket, self.settle_bucket = schedule, bucket, settle_bucket
        self.C = rmap.chunk
        self.cap = task.episode_cap
        assert self.cap % self.C == 0, "episode cap must be a multiple of C for fixed-duration transitions"
        dim = normalizer.mean.shape[0]
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (dim,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (rmap.dim,), np.float32)
        self.log_fn = log_fn
        self.proposal = None
        self.episode = None

    def _propose(self):
        seed = derive_seed("base", self.episode["key"], self.episode["chunks"])
        self.proposal = self.provider.propose(self.task.policy_observation(), None, seed)
        self.episode["chunks"] += 1
        return self.norm(raw_obs_vector(self.proposal, self.task.elapsed, self.cap))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options and "episode" in options:
            state_id, key = options["episode"]
        else:
            state_id, key = self.schedule.next()
        self.task.reset(state_id, settle_bucket=self.settle_bucket)
        self.episode = {"init_state_id": int(state_id), "key": key, "chunks": 0, "rms": [], "clip": [], "env_steps": 0}
        with self.task.counter.charge(self.bucket):
            obs = self._propose()
        return obs, {"episode_key": key}

    def step(self, u):
        u = np.asarray(u, np.float32)
        a_exec, st = self.rmap.apply(self.proposal.a0_env, u)
        n, success = 0, False
        with self.task.counter.charge(self.bucket):
            for t in range(self.C):
                _, success = self.task.step(a_exec[t])
                n += 1
                if success or self.task.horizon_left <= 0:
                    break
            terminated = success or self.task.horizon_left <= 0
            if not terminated:
                assert n == self.C
                obs = self._propose()
            else:
                obs = self.norm(raw_obs_vector(self.proposal, self.task.elapsed, self.cap))  # unused (no bootstrap)
        ep = self.episode
        ep["env_steps"] += n
        ep["rms"].append(st["rms_actual"])
        ep["clip"].append(st["clip_frac"])
        info = {"n_env_steps": n, "success": success, **st}
        if terminated:
            rec = {"init_state_id": ep["init_state_id"], "key": ep["key"], "success": bool(success),
                   "length": int(self.task.elapsed), "chunks": ep["chunks"],
                   "rms_mean": float(np.mean(ep["rms"])), "rms_median": float(np.median(ep["rms"])),
                   "clip_mean": float(np.mean(ep["clip"]))}
            info["episode_record"] = rec
            if self.log_fn:
                self.log_fn(rec)
        return obs, float(success), bool(terminated), False, info
