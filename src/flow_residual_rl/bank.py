"""Paired-intervention bank (plan sec. 8-9).

1. Pool: base-policy episodes from train (and every `bank_val_every`-th from
   validation) init states; a snapshot + proposal is kept at every chunk boundary.
2. Anchors: half uniform over pool boundaries, half from the last
   `bank_branch_horizon` steps of their episode.
3. Group per anchor: branches {a0, a0+delta, a0-delta}; the same base chunk a0,
   one direction (1/2 isotropic in R^N, 1/2 in the DCT-12 subspace), radius
   U[0.25,1]*eps*sqrt(N). After the modified chunk all branches continue with the
   frozen base policy (noise seeds shared relative to the anchor) for at most
   `bank_branch_horizon` steps from the anchor (incl. the modified chunk).
   y = 1 iff success inside the window.
Every real env.step is charged to a bank_* bucket; the sum is hard-limited.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .basis import dct_basis
from .budget import BudgetExceeded
from .config import derive_seed, write_json
from .envs.chunk_env import raw_obs_vector

BANK_BUCKETS = ("bank_pool", "bank_prefix_replay", "bank_branch", "bank_settling")


@dataclass
class Boundary:
    root: int
    split: str
    k: int
    elapsed: int
    snap: object
    proposal: object
    ep_len: int = -1
    ep_success: bool = False


def collect_pool(task, provider, cfg, split_ids, counter, log=print):
    """Run base-policy episodes until bank_pool_max_steps. Returns (episodes, boundaries)."""
    C, cap = cfg["execution_horizon"], cfg["episode_cap"]
    tr, va = list(split_ids["train"]), list(split_ids["validation"])
    rng = np.random.default_rng(derive_seed("pool_states", cfg["bank_seed"], task.task_id))
    tr, va = list(rng.permutation(tr)), list(rng.permutation(va))
    bounds, episodes = [], []
    pool_limit = cfg["bank_pool_max_steps"]
    root = 0
    while counter.steps["bank_pool"] + C <= pool_limit:
        split = "validation" if (root % cfg["bank_val_every"] == cfg["bank_val_every"] - 1) else "train"
        ids = va if split == "validation" else tr
        sid = int(ids[(root // (cfg["bank_val_every"] if split == "validation" else 1)) % len(ids)])
        task.reset(sid, settle_bucket="bank_settling")
        ep_b = []
        k = 0
        with counter.charge("bank_pool"):
            while True:
                if counter.steps["bank_pool"] + C > pool_limit:
                    break
                snap = task.snapshot()
                prop = provider.propose(task.policy_observation(), None, derive_seed("pool", task.task_id, root, k))
                ep_b.append(Boundary(root, split, k, task.elapsed, snap, prop))
                done = False
                for t in range(C):
                    _, s = task.step(prop.a0_env[t])
                    if s or task.horizon_left <= 0:
                        done = True
                        break
                k += 1
                if done:
                    break
        for b in ep_b:
            b.ep_len, b.ep_success = task.elapsed, task.success
        episodes.append({"root": root, "split": split, "init_state_id": sid, "length": task.elapsed,
                         "success": task.success, "complete": task.success or task.horizon_left <= 0})
        bounds.extend(ep_b)
        log(f"[pool] root={root} split={split} state={sid} len={task.elapsed} success={task.success} "
            f"pool_steps={counter.steps['bank_pool']}")
        root += 1
        counter.save()
    return episodes, bounds


def select_anchors(bounds, cfg, n_target, exclude=(), late_only=False, seed=0):
    C, W, cap = cfg["execution_horizon"], cfg["bank_branch_horizon"], cfg["episode_cap"]
    ok = [i for i, b in enumerate(bounds) if b.elapsed + C <= cap and i not in exclude]
    late = [i for i in ok if b_late(bounds[i], W)]
    rng = np.random.default_rng(seed)
    if late_only:
        return list(rng.permutation(late)[:n_target])
    n_late = min(n_target // 2, len(late))
    chosen = list(rng.permutation(late)[:n_late]) if n_late else []
    rest = [i for i in ok if i not in set(chosen)]
    chosen += list(rng.permutation(rest)[: n_target - len(chosen)])
    return [int(i) for i in chosen]


def b_late(b: Boundary, W: int) -> bool:
    return b.elapsed >= b.ep_len - W


def sample_direction(rng, chunk, n_dims, rank, eps):
    N = chunk * n_dims
    if rng.random() < 0.5:
        v = rng.standard_normal(N)
        kind = "isotropic"
    else:
        B = dct_basis(chunk, n_dims, rank // n_dims)
        v = B @ rng.standard_normal(B.shape[1])
        kind = "dct12"
    v = v / np.linalg.norm(v)
    radius = rng.uniform(0.25, 1.0) * eps * np.sqrt(N)
    return (radius * v).reshape(chunk, n_dims).astype(np.float32), kind, float(radius)


def run_branch(task, provider, cfg, anchor_id, b: Boundary, delta, counter, restore_mode):
    C, W = cfg["execution_horizon"], cfg["bank_branch_horizon"]
    dims = list(cfg["corrected_dims"])
    t0, calls0 = time.time(), counter.base_calls
    if restore_mode == "snapshot":
        task.restore(b.snap)
    else:
        task.replay_restore(b.snap, bucket="bank_prefix_replay")
    a0 = b.proposal.a0_env
    a = a0.copy()
    a[:, dims] = np.clip(a0[:, dims] + delta, task.ACTION_LOW[dims], task.ACTION_HIGH[dims])
    planned = (a[:, dims] - a0[:, dims]).astype(np.float32)
    window = min(W, task.episode_cap - b.elapsed)
    executed, first_success, reason = [], -1, "window"
    first_chunk_state = None
    with counter.charge("bank_branch"):
        chunk_actions, k = a, 0
        while True:
            for t in range(C):
                _, s = task.step(chunk_actions[t])
                executed.append(chunk_actions[t].copy())
                if s:
                    first_success, reason = len(executed), "success"
                    break
                if len(executed) >= window:
                    reason = "horizon" if task.horizon_left <= 0 else "window"
                    break
            if k == 0:
                first_chunk_state = task.state_vector()
            if reason == "success" or len(executed) >= window:
                break
            k += 1
            prop = provider.propose(task.policy_observation(), None, derive_seed("bank", anchor_id, k))
            chunk_actions = prop.a0_env
    return {
        "y": int(first_success > 0), "first_success_step": first_success, "executed_steps": len(executed),
        "termination_reason": reason, "branch_window": window, "planned": planned,
        "actual_actions": np.array(executed, np.float32), "first_chunk_state": first_chunk_state,
        "model_calls": counter.base_calls - calls0, "wall_s": time.time() - t0,
    }


def collect_groups(task, provider, cfg, bounds, anchor_idx, counter, out_dir: Path, start_id=0, log=print):
    """Returns list of group dicts; stops cleanly if the bank budget would be exceeded."""
    C, n_dims = cfg["execution_horizon"], len(cfg["corrected_dims"])
    W = cfg["bank_branch_horizon"]
    groups = []
    succ_ref = {(b.root, b.k): b for b in bounds}
    for j, i in enumerate(anchor_idx):
        b = bounds[i]
        aid = start_id + j
        est = 3 * min(W, cfg["episode_cap"] - b.elapsed)
        if cfg["restore_mode"] == "replay":
            est += 3 * (b.elapsed + cfg["settle_steps"])
        if counter.remaining(BANK_BUCKETS) < est:
            log(f"[bank] stop: remaining {counter.remaining(BANK_BUCKETS)} < estimated group cost {est}")
            break
        rng = np.random.default_rng(derive_seed("dir", cfg["bank_seed"], task.task_id, aid))
        delta, kind, radius = sample_direction(rng, C, n_dims, cfg["rank"], cfg["epsilon_rms"])
        branches = []
        try:
            for sign in (0.0, 1.0, -1.0):
                branches.append(run_branch(task, provider, cfg, aid, b, sign * delta, counter, cfg["restore_mode"]))
        except BudgetExceeded as e:
            log(f"[bank] budget exceeded inside group {aid}: {e}; group discarded (steps stay charged)")
            break
        # on-the-fly restore check: zero branch's first chunk == pool's own next boundary
        nxt = succ_ref.get((b.root, b.k + 1))
        restore_err = float(np.abs(branches[0]["first_chunk_state"] - nxt.snap.sim_flat).max()) if nxt else float("nan")
        g = {
            "anchor_id": aid, "task_id": task.task_id, "root_episode_id": b.root, "split": b.split,
            "initial_state_id": b.snap.init_state_id, "elapsed_steps": b.elapsed, "boundary_k": b.k,
            "late_anchor": b_late(b, W), "direction_kind": kind, "radius": radius,
            "base_seed": b.proposal.base_seed, "branch_seed_key": f"bank/{aid}/k",
            "a0_env": b.proposal.a0_env, "context": b.proposal.context, "proprio": b.proposal.proprio,
            "x_raw": raw_obs_vector(b.proposal, b.elapsed, cfg["episode_cap"]),
            "sim_state": b.snap.sim_flat, "restore_err_vs_pool": restore_err,
            "branches": branches,
        }
        groups.append(g)
        ys = [br["y"] for br in branches]
        log(f"[bank] group {aid} root={b.root} t={b.elapsed} split={b.split} y={ys} kind={kind} "
            f"restore_err={restore_err:.2e} bank_steps={counter.total(BANK_BUCKETS)}")
        counter.save()
    return groups


def discordant(groups, split):
    return sum(1 for g in groups if g["split"] == split and len({br["y"] for br in g["branches"]}) > 1)


def save_bank(out_dir: Path, groups, episodes, meta):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    G = len(groups)
    N = groups[0]["branches"][0]["planned"].size if G else 0
    arr = {
        "x_raw": np.stack([g["x_raw"] for g in groups]) if G else np.zeros((0, 1)),
        "delta": np.stack([[br["planned"].reshape(-1) for br in g["branches"]] for g in groups]) if G else np.zeros((0, 3, N)),
        "y": np.array([[br["y"] for br in g["branches"]] for g in groups], np.float32).reshape(G, 3),
        "split": np.array([g["split"] for g in groups]),
        "root": np.array([g["root_episode_id"] for g in groups]),
        "a0_env": np.stack([g["a0_env"] for g in groups]) if G else np.zeros((0,)),
        "sim_state": np.array([g["sim_state"] for g in groups], dtype=object),
    }
    np.savez_compressed(out_dir / "bank.npz", **{k: v for k, v in arr.items() if k != "sim_state"})
    np.save(out_dir / "anchor_sim_states.npy", arr["sim_state"], allow_pickle=True)
    rows = []
    for g in groups:
        r = {k: v for k, v in g.items() if k not in ("a0_env", "context", "proprio", "x_raw", "sim_state", "branches")}
        r["branches"] = [{k: (v.tolist() if isinstance(v, np.ndarray) and k == "actual_actions" else v)
                          for k, v in br.items() if k not in ("planned", "first_chunk_state")} for br in g["branches"]]
        rows.append(r)
    with open(out_dir / "groups.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=float) + "\n")
    write_json(out_dir / "bank_metadata.json", dict(meta, n_groups=G, n_discordant_train=discordant(groups, "train"),
                                                    n_discordant_val=discordant(groups, "validation"),
                                                    pool_episodes=episodes))


def pool_obs_matrix(bounds, cfg, split="train"):
    return np.stack([raw_obs_vector(b.proposal, b.elapsed, cfg["episode_cap"]) for b in bounds if b.split == split])
