"""CPU-only fakes for the provider and LIBERO task. Used by unit tests ONLY;
results obtained with them are not simulator experiments."""
from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import numpy as np

from flow_residual_rl.budget import StepCounter
from flow_residual_rl.providers.gr00t import Proposal, model_to_env


class FakeTask:
    ACTION_LOW = -np.ones(7, np.float32)
    ACTION_HIGH = np.ones(7, np.float32)

    def __init__(self, counter: StepCounter, episode_cap=64, goal=0.5, task_id=0, y_gate=None):
        self.counter, self.episode_cap, self.goal, self.task_id = counter, episode_cap, goal, task_id
        self.y_gate = y_gate  # if set, success also needs pos[1] > y_gate (makes +/- corrections matter)
        self.n_init_states = 10
        self.elapsed, self.success, self.pos, self.init_state_id = 0, False, np.zeros(3), -1
        self.prefix = []

    def reset(self, sid, settle_bucket=None):
        self.init_state_id, self.elapsed, self.success, self.prefix = sid, 0, False, []
        self.pos = np.array([0.05 * sid, 0.0, 0.0])
        return {}

    def step(self, a):
        self.counter.on_env_step()
        self.pos = self.pos + 0.02 * np.asarray(a[:3])
        self.elapsed += 1
        self.prefix.append(np.asarray(a, np.float32).copy())
        self.success = bool(self.pos[0] > self.goal and (self.y_gate is None or self.pos[1] > self.y_gate))
        return {}, self.success

    @property
    def horizon_left(self):
        return self.episode_cap - self.elapsed

    def policy_observation(self):
        return {"pos": self.pos.copy(), "language": "fake"}

    def proprio(self):
        return np.concatenate([self.pos, np.zeros(5)]).astype(np.float32)

    def snapshot(self):
        return SimpleNamespace(sim_flat=self.pos.copy(), elapsed=self.elapsed, success=self.success,
                               init_state_id=self.init_state_id, action_prefix=np.array(self.prefix, np.float32))

    def restore(self, s):
        self.pos, self.elapsed, self.success, self.init_state_id = s.sim_flat.copy(), s.elapsed, s.success, s.init_state_id
        self.prefix = list(s.action_prefix)

    def replay_restore(self, s, bucket="bank_prefix_replay"):
        with self.counter.charge(bucket):
            self.reset(s.init_state_id)
            for a in s.action_prefix:
                self.step(a)

    def task_info(self):
        return {"task_id": self.task_id, "name": "fake"}

    def close(self):
        pass

    def state_vector(self):
        return self.pos.copy()


class FakeProvider:
    def __init__(self, chunk=8, context_dim=128, drift=0.3):
        self.chunk, self.context_dim, self.drift = chunk, context_dim, drift
        self.counter = None
        self.calls = 0

    def propose(self, observation, task, proposal_seed):
        self.calls += 1
        rng = np.random.default_rng(proposal_seed)
        raw = np.zeros((16, 7), np.float32)
        raw[:, 0] = self.drift + 0.1 * rng.standard_normal(16)
        raw[:, 6] = 0.9
        return Proposal(a0_env=model_to_env(raw[: self.chunk]), context=rng.standard_normal(self.context_dim).astype(np.float32),
                        proprio=np.concatenate([observation["pos"], np.zeros(5)]).astype(np.float32),
                        base_seed=int(proposal_seed), metadata={}, raw_chunk=raw)
