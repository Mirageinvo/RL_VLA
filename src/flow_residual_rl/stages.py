"""Stage logic shared by the CLI scripts: bank, basis, baseline eval, pilot cells."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from . import bank as bk
from .basis import check_orthonormal, pca_basis, save_basis
from .budget import StepCounter
from .config import read_json, write_json
from .envs.chunk_env import ChunkEnv, EpisodeSchedule, ObsNormalizer
from .evaluate import eval_schedule, run_episodes, sac_policy, zero_policy
from .outcome_predictor import GroupBatch, train_outcome_basis
from .pipeline import basis_for_arm, load_normalizer, make_task, residual_map, task_dir, task_splits


def log_to(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    def _log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(path, "a") as f:
            f.write(line + "\n")

    return _log


# ---------------------------------------------------------------- bank
def build_bank(cfg, task_id, provider) -> dict:
    out = task_dir(cfg, task_id) / "bank"
    meta = read_json(out / "bank_metadata.json")
    if meta is not None and meta.get("complete"):
        return meta
    log = log_to(out / "collect.log")
    counter = StepCounter(out / "cost.json")
    stale = counter.total(bk.BANK_BUCKETS)
    if stale:  # an earlier attempt crashed: its steps stay recorded, as aborted
        log(f"[bank] previous incomplete attempt spent {stale} steps -> moved to 'aborted'")
        for b in bk.BANK_BUCKETS:
            counter.steps["aborted"] += counter.steps[b]
            counter.steps[b] = 0
    counter.set_limit(bk.BANK_BUCKETS, cfg["bank_budget_steps"])
    provider.counter = counter
    task = make_task(cfg, task_id, counter, provider)
    sp = task_splits(cfg, task)
    episodes, bounds = bk.collect_pool(task, provider, cfg, sp, counter, log)
    norm = ObsNormalizer.fit(bk.pool_obs_matrix(bounds, cfg, "train"))
    write_json(out / "obs_norm.json", dict(norm.to_dict(), fit_on="bank pool boundaries, train split",
                                           n=int(sum(b.split == "train" for b in bounds))))
    anchors = bk.select_anchors(bounds, cfg, cfg["bank_anchor_target"], seed=cfg["bank_seed"])
    groups = bk.collect_groups(task, provider, cfg, bounds, anchors, counter, out, log=log)
    rescue = False
    if (bk.discordant(groups, "train") < cfg["basis_min_discordant_train"]
            or bk.discordant(groups, "validation") < cfg["basis_min_discordant_val"]):
        rescue = True
        counter.set_limit(bk.BANK_BUCKETS, cfg["bank_max_steps_with_rescue"])
        used = set(anchors)
        extra = bk.select_anchors(bounds, cfg, 10**6, exclude=used, late_only=True, seed=cfg["bank_seed"] + 1)
        log(f"[bank] rescue portion: discordant train={bk.discordant(groups, 'train')} "
            f"val={bk.discordant(groups, 'validation')}; {len(extra)} late candidates")
        groups += bk.collect_groups(task, provider, cfg, bounds, extra, counter, out, start_id=len(groups), log=log)
    restore_errs = [g["restore_err_vs_pool"] for g in groups if np.isfinite(g["restore_err_vs_pool"])]
    meta = {"task": task.task_info(), "rescue_used": rescue, "restore_mode": cfg["restore_mode"],
            "restore_err_vs_pool_max": float(max(restore_errs)) if restore_errs else None,
            "restore_err_vs_pool_median": float(np.median(restore_errs)) if restore_errs else None,
            "cost": counter.to_dict(), "complete": True}
    bk.save_bank(out, groups, episodes, meta)
    counter.save()
    task.close()
    return read_json(out / "bank_metadata.json")


# ---------------------------------------------------------------- basis
def _group_batch(d, norm: ObsNormalizer, idx) -> GroupBatch:
    x = torch.as_tensor(norm(d["x_raw"][idx]))
    delta = torch.as_tensor(d["delta"][idx], dtype=torch.float32)
    y = torch.as_tensor(d["y"][idx], dtype=torch.float32)
    return GroupBatch(x, delta, y, torch.ones_like(y, dtype=torch.bool))


def build_bases(cfg, task_id) -> dict:
    tdir = task_dir(cfg, task_id)
    bdir = tdir / "bank"
    meta = read_json(bdir / "bank_metadata.json")
    assert meta and meta.get("complete"), "bank missing or incomplete"
    d = dict(np.load(bdir / "bank.npz"))
    norm = load_normalizer(cfg, task_id)
    tr = np.where(d["split"] == "train")[0]
    va = np.where(d["split"] == "validation")[0]
    C, nd = cfg["execution_horizon"], len(cfg["corrected_dims"])
    N = C * nd
    result = {"task_id": task_id}
    # pca12: label-free control on planned corrections of the train bank (non-zero branches)
    dtr = d["delta"][tr][:, 1:].reshape(-1, N)
    bp = pca_basis(dtr, cfg["rank"])
    save_basis(tdir / "basis" / "pca12", bp, {"arm": "pca12", "fit_on": "train planned corrections (+/- branches)",
                                               "n": int(dtr.shape[0])})
    n_disc_tr = int(sum(len(set(r)) > 1 for r in d["y"][tr]))
    n_disc_va = int(sum(len(set(r)) > 1 for r in d["y"][va]))
    result.update(n_groups_train=len(tr), n_groups_val=len(va), n_discordant_train=n_disc_tr, n_discordant_val=n_disc_va)
    if n_disc_tr < cfg["basis_min_discordant_train"] or n_disc_va < cfg["basis_min_discordant_val"]:
        result["status"] = "insufficient_outcome_signal"
        write_json(tdir / "basis" / "outcome12_status.json", result)
        return result
    res = train_outcome_basis(
        _group_batch(d, norm, tr), _group_batch(d, norm, va), cfg["rank"], cfg["epsilon_rms"] * np.sqrt(N),
        tdir / "basis" / "outcome12" / "predictor", updates=cfg["basis_train_updates"], lr=cfg["basis_lr"],
        weight_decay=cfg["basis_weight_decay"], batch_groups=cfg["basis_batch_groups"],
        pair_weight=cfg["basis_pair_weight"], hidden=tuple(cfg["basis_hidden"]), eval_every=cfg["basis_eval_every"],
        seed=cfg["bank_seed"])
    b = res["basis"]
    check_orthonormal(b)
    last_val = res["log"][-1]["val"] if res["log"] else {}
    best = next((r for r in res["log"] if r["update"] == res["best_update"]), {})
    save_basis(tdir / "basis" / "outcome12", b, {"arm": "outcome12", "best_update": res["best_update"],
                                                  "best_val": best.get("val"), "last_val": last_val})
    # diagnostics: how much of each control subspace the learned one overlaps
    def overlap(a, c):
        return float(np.linalg.norm(a.T @ c) ** 2 / a.shape[1])

    result.update(status="ok", best_update=res["best_update"], best_val=best.get("val"),
                  overlap_with={"dct12": overlap(b, basis_for_arm(cfg, "dct12", task_id)),
                                "random12": overlap(b, basis_for_arm(cfg, "random12", task_id)),
                                "pca12": overlap(b, bp)})
    write_json(tdir / "basis" / "outcome12_status.json", result)
    return result


# ---------------------------------------------------------------- evaluation helpers
def eval_env(cfg, task, provider, rmap, norm, eval_ids):
    return ChunkEnv(task, provider, rmap, norm, eval_schedule(eval_ids), bucket="evaluation", settle_bucket="eval_settling")


def eval_base(cfg, task_id, provider, task, n_episodes, counter) -> dict:
    """Base VLA (u=0) on the fixed eval schedule; episodes are cached and reused."""
    out = task_dir(cfg, task_id) / "base_eval"
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "episodes.jsonl"
    have = [json.loads(l) for l in cache.read_text().splitlines()] if cache.exists() else []
    sp = task_splits(cfg, task)
    norm = load_normalizer(cfg, task_id, allow_identity=True)
    rmap = residual_map(cfg, basis_for_arm(cfg, "random12", task_id), task)
    env = eval_env(cfg, task, provider, rmap, norm, sp["evaluation"])
    provider.counter = counter
    for e in range(len(have), n_episodes):
        obs, _ = env.reset(options={"episode": env.schedule.get(e)})
        while True:
            obs, r, term, _, info = env.step(np.zeros(rmap.dim, np.float32))
            if term:
                break
        rec = dict(info["episode_record"], episode=e)
        have.append(rec)
        with open(cache, "a") as f:
            f.write(json.dumps(rec) + "\n")
        counter.save()
    recs = have[:n_episodes]
    return {"n_episodes": len(recs), "success_rate": float(np.mean([r["success"] for r in recs])),
            "n_unique_states": len({r["init_state_id"] for r in recs})}


# ---------------------------------------------------------------- pilot cells
def cell_dir(cfg, task_id, arm, seed) -> Path:
    return task_dir(cfg, task_id) / "cells" / f"{arm}_seed{seed}"


def run_cell_block(cfg, task, provider, arm, seed, target, sp, log) -> dict:
    from .train_sac import train_block

    cdir = cell_dir(cfg, task.task_id, arm, seed)
    cdir.mkdir(parents=True, exist_ok=True)
    counter = StepCounter(cdir / "cost.json")
    st = read_json(cdir / "status.json", {"online_steps": 0})
    # steps spent after the last committed block (crash) are kept, as aborted
    lost = counter.steps["online_train"] - st["online_steps"]
    if lost > 0:
        log(f"[{arm} s{seed}] {lost} uncommitted online steps from a crashed attempt -> aborted")
        counter.steps["aborted"] += lost
        counter.steps["online_train"] -= lost
    task.counter = counter
    provider.counter = counter
    basis = basis_for_arm(cfg, arm, task.task_id)
    rmap = residual_map(cfg, basis, task)
    norm = load_normalizer(cfg, task.task_id)
    env = ChunkEnv(task, provider, rmap, norm, EpisodeSchedule(sp["train"], "train", seed),
                   bucket="online_train", settle_bucket="online_settling")
    if not (cdir / "basis.npy").exists():
        np.save(cdir / "basis.npy", basis)
    else:
        assert np.array_equal(np.load(cdir / "basis.npy"), basis), "basis changed: this is a new experiment, new dir"
    status = train_block(cdir, env, cfg, seed, target)
    counter.save()
    log(f"[{arm} s{seed}] online={status['online_steps']} last block: {status['blocks'][-1] if status['blocks'] else None}")
    return status


def eval_cell(cfg, task, provider, arm, seed, n_episodes, at_steps, sp, log, policy_ckpt: Path | None = None) -> dict:
    from stable_baselines3 import SAC
    from stable_baselines3.sac.policies import SACPolicy

    cdir = cell_dir(cfg, task.task_id, arm, seed)
    out = cdir / "evals" / f"eval_{at_steps:07d}_n{n_episodes}.json"
    if out.exists():
        return json.loads(out.read_text())["summary"]
    counter = StepCounter(cdir / "cost.json")
    task.counter = counter
    provider.counter = counter
    rmap = residual_map(cfg, np.load(cdir / "basis.npy"), task)
    env = eval_env(cfg, task, provider, rmap, load_normalizer(cfg, task.task_id), sp["evaluation"])
    if policy_ckpt is None:
        model = SAC.load(cdir / "model.zip", device=cfg["sac_device"])
        pol = sac_policy(model)
    else:
        policy = SACPolicy.load(policy_ckpt, device=cfg["sac_device"])
        pol = lambda o: policy.predict(o, deterministic=True)[0]  # noqa: E731
    summ = run_episodes(env, pol, n_episodes, out, video_dir=cdir / "videos" / f"{at_steps:07d}", n_videos=2)
    summ.update(arm=arm, seed=seed, task_id=task.task_id, online_steps=at_steps,
                policy_ckpt=str(policy_ckpt) if policy_ckpt else "model.zip")
    payload = json.loads(out.read_text())
    payload["summary"] = summ
    write_json(out, payload)
    counter.save()
    log(f"[{arm} s{seed}] eval@{at_steps}: SR={summ['success_rate']:.3f} (n={n_episodes})")
    return summ
