"""SB3 SAC on chunk-level compact observations (plan sec. 12, 14).

* actor mean head zero-initialised, log_std head weights 0 / bias -3 — only when a
  fresh model is created (never after load);
* warmup samples u = r * direction, direction uniform on the unit sphere of R^d,
  r ~ U[warmup_radius]; since delta = eps*sqrt(N)*B w with orthonormal B, the
  physical RMS distribution is identical for every arm;
* training proceeds in env-step blocks; at a block boundary the policy, optimizer
  and replay buffer are saved, so a later call continues the same run.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from .config import read_json, write_json


class ResidualSAC(SAC):
    def __init__(self, *args, warmup_radius=(0.25, 1.0), warmup_seed: int = 0, **kw):
        self.warmup_radius = tuple(warmup_radius)
        self._warm_rng = np.random.default_rng(warmup_seed)
        super().__init__(*args, **kw)

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        if self.num_timesteps < learning_starts:
            d = self.action_space.shape[0]
            v = self._warm_rng.standard_normal((n_envs, d))
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            r = self._warm_rng.uniform(*self.warmup_radius, size=(n_envs, 1))
            a = (v * r).astype(np.float32)
            return a, a  # Box[-1,1]: scaled == unscaled
        return super()._sample_action(learning_starts, action_noise, n_envs)


def init_actor_heads(model: SAC, log_std_bias: float) -> None:
    actor = model.actor
    with torch.no_grad():
        actor.mu.weight.zero_()
        actor.mu.bias.zero_()
        actor.log_std.weight.zero_()
        actor.log_std.bias.fill_(log_std_bias)
    # SB3 SAC (use_sde=False) has no separate log_std_init parameter: log_std is a Linear head.


def make_sac(env, cfg: dict, seed: int, smoke: bool = False) -> ResidualSAC:
    d = env.action_space.shape[0]
    model = ResidualSAC(
        "MlpPolicy", env,
        learning_rate=cfg["sac_lr"],
        buffer_size=cfg["sac_buffer_size"],
        learning_starts=cfg["smoke_sac_learning_starts"] if smoke else cfg["sac_learning_starts_chunks"],
        batch_size=cfg["smoke_sac_batch_size"] if smoke else cfg["sac_batch_size"],
        tau=cfg["sac_tau"], gamma=cfg["sac_gamma_chunk"],
        train_freq=(cfg["sac_train_freq_chunks"], "step"), gradient_steps=cfg["sac_gradient_steps"],
        ent_coef=cfg["sac_ent_coef"], target_entropy=-float(d), use_sde=False,
        policy_kwargs=dict(net_arch=dict(pi=list(cfg["sac_hidden"]), qf=list(cfg["sac_hidden"]))),
        device=cfg["sac_device"], seed=seed, verbose=0,
        warmup_radius=cfg["warmup_radius"], warmup_seed=seed + 1,
    )
    init_actor_heads(model, cfg["sac_log_std_bias_init"])
    return model


class BudgetCallback(BaseCallback):
    """Counts real env steps of this run; stops after the transition that reaches `target`."""

    def __init__(self, start_steps: int, target: int, ckpt_every: int, ckpt_dir: Path, episodes_log: Path):
        super().__init__()
        self.steps, self.target = start_steps, target
        self.ckpt_every, self.ckpt_dir, self.episodes_log = ckpt_every, ckpt_dir, episodes_log
        self.next_ckpt = (start_steps // ckpt_every + 1) * ckpt_every
        self.stats = []

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            self.steps += info["n_env_steps"]
            self.stats.append((info["rms_actual"], info["clip_frac"]))
            rec = info.get("episode_record")
            if rec is not None:
                rec = dict(rec, online_steps_end=self.steps, wall=time.time())
                with open(self.episodes_log, "a") as f:
                    f.write(json.dumps(rec) + "\n")
        if self.steps >= self.next_ckpt:
            self.model.policy.save(self.ckpt_dir / f"policy_{self.next_ckpt:07d}.pt")
            self.next_ckpt += self.ckpt_every
        return self.steps < self.target


def train_block(cell_dir: Path, env, cfg: dict, seed: int, target_steps: int, smoke: bool = False) -> dict:
    """Train (or resume) one arm/task/seed cell until `target_steps` online env steps."""
    cell_dir = Path(cell_dir)
    (cell_dir / "policy_ckpts").mkdir(parents=True, exist_ok=True)
    status = read_json(cell_dir / "status.json", {"online_steps": 0, "episode_idx": 0, "blocks": []})
    if status["online_steps"] >= target_steps:
        return status
    if (cell_dir / "model.zip").exists():
        model = ResidualSAC.load(cell_dir / "model.zip", env=env, device=cfg["sac_device"])
        model.load_replay_buffer(cell_dir / "replay.pkl")
        fresh = False
    else:
        model = make_sac(env, cfg, seed, smoke=smoke)
        fresh = True
    env.schedule.idx = status["episode_idx"]
    model._last_obs = None  # force a real env.reset: never continue from a stale observation
    cb = BudgetCallback(status["online_steps"], target_steps, cfg["policy_checkpoint_every"],
                        cell_dir / "policy_ckpts", cell_dir / "train_episodes.jsonl")
    t0 = time.time()
    model.learn(total_timesteps=10**9, callback=cb, reset_num_timesteps=fresh, log_interval=None, progress_bar=False)
    wall = time.time() - t0
    model.save(cell_dir / "model.zip")
    model.save_replay_buffer(cell_dir / "replay.pkl")
    model.policy.save(cell_dir / "policy_ckpts" / f"policy_block_{cb.steps:07d}.pt")
    st = np.array(cb.stats) if cb.stats else np.zeros((1, 2))
    status["blocks"].append({"from": status["online_steps"], "to": cb.steps, "wall_s": wall,
                             "chunks": int(model.num_timesteps), "rms_mean": float(st[:, 0].mean()),
                             "clip_mean": float(st[:, 1].mean())})
    status["online_steps"] = cb.steps
    status["episode_idx"] = env.schedule.idx
    status["trainable_params"] = int(sum(p.numel() for p in model.policy.parameters()))
    write_json(cell_dir / "status.json", status)
    return status


def grad_report(model: SAC) -> dict:
    out = {}
    for name, mod in [("actor", model.actor), ("critic", model.critic)]:
        gs = [p.grad for p in mod.parameters() if p.requires_grad]
        out[name] = {"n_params_with_grad": sum(g is not None for g in gs),
                     "all_finite": bool(all(torch.isfinite(g).all() for g in gs if g is not None)),
                     "grad_norm": float(sum(float(g.norm()) ** 2 for g in gs if g is not None) ** 0.5)}
    return out
