"""SB3 SAC on chunk-level compact observations (plan sec. 12, 14).

* actor mean head zero-initialised, log_std head weights 0 / bias -3 — only when a
  fresh model is created (never after load);
* warmup samples u = r * direction, direction uniform on the unit sphere of R^d,
  r ~ U[warmup_radius]; since delta = eps*sqrt(N)*B w with orthonormal B, the
  physical RMS distribution is identical for every arm;
* training proceeds in env-step blocks. A block stops right AFTER the transition
  that reaches the target has been stored in the replay buffer (SB3 would drop it
  if the callback returned False: collect_rollouts returns before _store_transition).
* at a block boundary model+optimizer, replay buffer and block policy are written
  into a new generation directory gens/gen_<steps>; status.json (atomic) is then
  switched to it and older generations are removed. A crash at any point leaves the
  previous complete generation referenced, never a mix of two blocks.
* the cost counter is saved before status.json, so committed online steps are
  always on disk; steps beyond status.online_steps are booked as aborted_online
  by the caller on the next start.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from .config import read_json, write_json


class BlockDone(Exception):
    """Raised right after the last transition of a block has been stored."""


class ResidualSAC(SAC):
    def __init__(self, *args, warmup_radius=(0.25, 1.0), warmup_seed: int = 0, **kw):
        self.warmup_radius = tuple(warmup_radius)
        self._warm_rng = np.random.default_rng(warmup_seed)
        self._stop_after_store = False
        super().__init__(*args, **kw)

    def _store_transition(self, *args, **kw):
        super()._store_transition(*args, **kw)
        self._n_stored = getattr(self, "_n_stored", 0) + 1
        if self._stop_after_store:
            raise BlockDone()

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


def ckpt_index(ckpt_dir: Path) -> dict:
    return read_json(Path(ckpt_dir) / "index.json", {})


def save_policy_ckpt(model, ckpt_dir: Path, name: str, actual_steps: int) -> None:
    """Policy weights + the exact number of online env steps they were taken at."""
    model.policy.save(Path(ckpt_dir) / name)
    idx = ckpt_index(ckpt_dir)
    idx[name] = int(actual_steps)
    write_json(Path(ckpt_dir) / "index.json", idx)


def prune_ckpts(ckpt_dir: Path, committed_steps: int) -> list:
    """Drop policy checkpoints taken in an uncommitted (crashed) portion of a block."""
    idx = ckpt_index(ckpt_dir)
    stale = [n for n, s in idx.items() if s > committed_steps]
    for n in stale:
        (Path(ckpt_dir) / n).unlink(missing_ok=True)
        idx.pop(n)
    if stale:
        write_json(Path(ckpt_dir) / "index.json", idx)
    return stale


def policy_for_step(ckpt_dir: Path, at: int, chunk: int) -> tuple[Path, int]:
    """The checkpoint taken when the run first crossed `at` online env steps (at <= actual < at + chunk)."""
    idx = ckpt_index(ckpt_dir)
    hits = sorted((s, n) for n, s in idx.items() if at <= s < at + chunk)
    if not hits:
        raise FileNotFoundError(f"no policy checkpoint at {at} online steps in {ckpt_dir} (have {sorted(idx.values())})")
    s, n = hits[0]
    return Path(ckpt_dir) / n, s


class BudgetCallback(BaseCallback):
    """Counts real env steps of this run; after the transition that reaches `target` (or after
    `deadline`), asks the model to stop once that transition is stored."""

    def __init__(self, start_steps: int, target: int, ckpt_every: int, ckpt_dir: Path, episodes_log: Path,
                 deadline: float | None = None):
        super().__init__()
        self.steps, self.target, self.deadline = start_steps, target, deadline
        self.ckpt_every, self.ckpt_dir, self.episodes_log = ckpt_every, ckpt_dir, episodes_log
        self.next_ckpt = (start_steps // ckpt_every + 1) * ckpt_every
        self.stats = []
        self.chunks = 0
        self.deadline_hit = False

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            self.chunks += 1
            self.steps += info["n_env_steps"]
            self.stats.append((info["rms_actual"], info["clip_frac"]))
            rec = info.get("episode_record")
            if rec is not None:
                rec = dict(rec, online_steps_end=self.steps, wall=time.time())
                with open(self.episodes_log, "a") as f:
                    f.write(json.dumps(rec) + "\n")
        if self.steps >= self.next_ckpt:
            save_policy_ckpt(self.model, self.ckpt_dir, f"policy_{self.next_ckpt:07d}.pt", self.steps)
            while self.next_ckpt <= self.steps:
                self.next_ckpt += self.ckpt_every
        if self.deadline is not None and time.time() > self.deadline:
            self.deadline_hit = True
        if self.steps >= self.target or self.deadline_hit:
            self.model._stop_after_store = True  # this transition is still stored, then BlockDone
        return True


def model_dir(cell_dir: Path, status: dict) -> Path | None:
    """Directory of the committed generation (model.zip, replay.pkl), None for a fresh cell."""
    return Path(cell_dir) / "gens" / status["gen"] if status.get("gen") else None


def train_block(cell_dir: Path, env, cfg: dict, seed: int, target_steps: int, smoke: bool = False,
                deadline: float | None = None) -> dict:
    """Train (or resume) one arm/task/seed cell until `target_steps` online env steps (or `deadline`)."""
    cell_dir = Path(cell_dir)
    ckpt_dir = cell_dir / "policy_ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    status = read_json(cell_dir / "status.json", {"online_steps": 0, "episode_idx": 0, "blocks": []})
    if status["online_steps"] >= target_steps:
        return status
    prune_ckpts(ckpt_dir, status["online_steps"])
    gdir = model_dir(cell_dir, status)
    if gdir is not None:
        model = ResidualSAC.load(gdir / "model.zip", env=env, device=cfg["sac_device"])
        model.load_replay_buffer(gdir / "replay.pkl")
        fresh = False
    else:
        assert status["online_steps"] == 0, "status has steps but no committed generation"
        model = make_sac(env, cfg, seed, smoke=smoke)
        fresh = True
    model._stop_after_store = False
    model._n_stored = 0
    replay0 = model.replay_buffer.size()
    env.schedule.idx = status["episode_idx"]
    model._last_obs = None  # force a real env.reset: never continue from a stale observation
    cb = BudgetCallback(status["online_steps"], target_steps, cfg["policy_checkpoint_every"], ckpt_dir,
                        cell_dir / "train_episodes.jsonl", deadline=deadline)
    t0 = time.time()
    counter = env.task.counter
    try:
        model.learn(total_timesteps=10**12, callback=cb, reset_num_timesteps=fresh, log_interval=None,
                    progress_bar=False)
        raise RuntimeError("SAC.learn returned without reaching the block target")
    except BlockDone:
        pass
    finally:
        counter.save()  # whatever happened, every executed env step is on disk
    model._stop_after_store = False
    wall = time.time() - t0
    stored = model._n_stored
    assert stored == cb.chunks, f"executed {cb.chunks} chunk transitions but stored {stored}"
    gen = f"gen_{cb.steps:07d}"
    new_dir = cell_dir / "gens" / gen
    if new_dir.exists():
        shutil.rmtree(new_dir)
    new_dir.mkdir(parents=True)
    model.save(new_dir / "model.zip")
    model.save_replay_buffer(new_dir / "replay.pkl")
    save_policy_ckpt(model, ckpt_dir, f"policy_block_{cb.steps:07d}.pt", cb.steps)
    st = np.array(cb.stats) if cb.stats else np.zeros((1, 2))
    status["blocks"].append({"from": status["online_steps"], "to": cb.steps, "target": target_steps,
                             "deadline_stop": cb.deadline_hit, "wall_s": wall,
                             "chunks": int(model.num_timesteps), "chunks_this_block": cb.chunks,
                             "replay_added": int(model.replay_buffer.size() - replay0),
                             "rms_mean": float(st[:, 0].mean()), "clip_mean": float(st[:, 1].mean())})
    old_gen = status.get("gen")
    status.update(online_steps=cb.steps, episode_idx=env.schedule.idx, gen=gen,
                  trainable_params=int(sum(p.numel() for p in model.policy.parameters())))
    counter.save()
    write_json(cell_dir / "status.json", status)  # commit point
    if old_gen and old_gen != gen:
        shutil.rmtree(cell_dir / "gens" / old_gen, ignore_errors=True)
    return status


def grad_report(model: SAC) -> dict:
    out = {}
    for name, mod in [("actor", model.actor), ("critic", model.critic)]:
        gs = [p.grad for p in mod.parameters() if p.requires_grad]
        out[name] = {"n_params_with_grad": sum(g is not None for g in gs),
                     "all_finite": bool(all(torch.isfinite(g).all() for g in gs if g is not None)),
                     "grad_norm": float(sum(float(g.norm()) ** 2 for g in gs if g is not None) ** 0.5)}
    return out
