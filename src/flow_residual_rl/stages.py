"""Stage logic shared by the CLI scripts: bank, basis, baseline eval, pilot cells."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from . import bank as bk
from . import fingerprint as fpr
from .basis import check_orthonormal, pca_basis, save_basis
from .budget import StepCounter
from .config import read_json, write_json
from .envs.chunk_env import ChunkEnv, EpisodeSchedule, ObsNormalizer
from .evaluate import eval_schedule, run_episodes
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
    out.mkdir(parents=True, exist_ok=True)
    log = log_to(out / "collect.log")
    counter = StepCounter(out / "cost.json")
    provider.counter = counter
    task = make_task(cfg, task_id, counter, provider)
    sp = task_splits(cfg, task)
    fpr.check_or_write(out, fpr.bank_fp(cfg, task_id, provider, sp), cfg, log)
    meta = read_json(out / "bank_metadata.json")
    if meta is not None and meta.get("complete"):
        task.close()
        return meta
    stale = counter.total(bk.BANK_BUCKETS)
    if stale:  # an earlier attempt crashed: its steps stay charged (aborted_bank) and count toward the limit
        log(f"[bank] previous incomplete attempt spent {stale} steps -> aborted_bank (still counted in the limit)")
        for b in bk.BANK_BUCKETS:
            counter.move(b, "aborted_bank", counter.steps[b], "crashed bank attempt")
        counter.save()
    counter.set_limit(bk.BANK_LIMIT_BUCKETS, cfg["bank_budget_steps"])
    log(f"[bank] budget {cfg['bank_budget_steps']} incl. {counter.steps['aborted_bank']} aborted; "
        f"remaining {counter.remaining(bk.BANK_LIMIT_BUCKETS)}")
    try:
        return _collect_bank(cfg, task, provider, counter, sp, out, log)
    finally:
        counter.save()
        task.close()


def _collect_bank(cfg, task, provider, counter, sp, out, log) -> dict:
    episodes, bounds = bk.collect_pool(task, provider, cfg, sp, counter, log)
    norm = ObsNormalizer.fit(bk.pool_obs_matrix(bounds, cfg, "train"))
    write_json(out / "obs_norm.json", dict(norm.to_dict(), fit_on="bank pool boundaries, train split",
                                           n=int(sum(b.split == "train" for b in bounds))))
    anchors = bk.select_anchors(bounds, cfg, cfg["bank_anchor_target"], seed=cfg["bank_seed"])
    rejected = []
    groups = bk.collect_groups(task, provider, cfg, bounds, anchors, counter, out, log=log, rejected=rejected)
    rescue = False
    if (bk.discordant(groups, "train") < cfg["basis_min_discordant_train"]
            or bk.discordant(groups, "validation") < cfg["basis_min_discordant_val"]):
        rescue = True
        counter.set_limit(bk.BANK_LIMIT_BUCKETS, cfg["bank_max_steps_with_rescue"])
        used = set(anchors)
        extra = bk.select_anchors(bounds, cfg, 10**6, exclude=used, late_only=True, seed=cfg["bank_seed"] + 1)
        log(f"[bank] rescue portion: discordant train={bk.discordant(groups, 'train')} "
            f"val={bk.discordant(groups, 'validation')}; {len(extra)} late candidates")
        groups += bk.collect_groups(task, provider, cfg, bounds, extra, counter, out,
                                    start_id=len(groups) + len(rejected), log=log, rejected=rejected)
    restore_errs = [g["restore_err_vs_pool"] for g in groups if np.isfinite(g["restore_err_vs_pool"])]
    meta = {"task": task.task_info(), "rescue_used": rescue, "restore_mode": cfg["restore_mode"],
            "restore_err_vs_pool_max": float(max(restore_errs)) if restore_errs else None,
            "restore_err_vs_pool_median": float(np.median(restore_errs)) if restore_errs else None,
            "restore_atol": cfg["restore_atol"], "restore_rtol": cfg["restore_rtol"],
            "n_restore_checked": int(sum(g["restore_ok"] is not None for g in groups)) + len(rejected),
            "cost": counter.to_dict(), "complete": True}
    counter.save()
    bk.save_bank(out, groups, episodes, meta, rejected=rejected)  # bank_metadata.json (complete) = commit point
    return read_json(out / "bank_metadata.json")


# ---------------------------------------------------------------- basis
def _group_batch(d, norm: ObsNormalizer, idx) -> GroupBatch:
    x = torch.as_tensor(norm(d["x_raw"][idx]))
    delta = torch.as_tensor(d["delta"][idx], dtype=torch.float32)
    y = torch.as_tensor(d["y"][idx], dtype=torch.float32)
    return GroupBatch(x, delta, y, torch.ones_like(y, dtype=torch.bool))


def build_bases(cfg, task_id, reuse: bool = True) -> dict:
    """pca12 + outcome12 for one task. With reuse, a finished result with the same fingerprint is returned."""
    tdir = task_dir(cfg, task_id)
    bdir = tdir / "bank"
    meta = read_json(bdir / "bank_metadata.json")
    assert meta and meta.get("complete"), "bank missing or incomplete"
    fp = fpr.basis_fp(cfg, task_id, bdir)
    prev = read_json(tdir / "basis" / "outcome12_status.json")
    if reuse and prev is not None and (tdir / "basis" / "pca12" / "basis.npy").exists():
        fpr.check_or_write(tdir / "basis", fp, cfg)
        return prev
    (tdir / "basis").mkdir(parents=True, exist_ok=True)
    (tdir / "basis" / "fingerprint.json").unlink(missing_ok=True)  # rebuilt below: new contract for later cells
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
        fpr.check_or_write(tdir / "basis", fp, cfg)
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
    fpr.check_or_write(tdir / "basis", fp, cfg)
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
    sp = task_splits(cfg, task)
    fpr.check_or_write(out, fpr.base_eval_fp(cfg, task_id, provider, sp), cfg)
    have = [json.loads(l) for l in cache.read_text().splitlines() if l.strip()] if cache.exists() else []
    counter.reconcile("evaluation", sum(r["length"] for r in have), "aborted_eval",
                      "base eval: steps of an interrupted (uncached) episode")
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
        counter.save()
        with open(cache, "a") as f:  # commit point of this episode
            f.write(json.dumps(rec) + "\n")
    recs = have[:n_episodes]
    return {"n_episodes": len(recs), "success_rate": float(np.mean([r["success"] for r in recs])),
            "n_unique_states": len({r["init_state_id"] for r in recs})}


# ---------------------------------------------------------------- pilot cells
def cell_dir(cfg, task_id, arm, seed) -> Path:
    return task_dir(cfg, task_id) / "cells" / f"{arm}_seed{seed}"


def _cell_env_parts(cfg, task, provider, arm, seed, sp, cdir, log):
    """Basis / normaliser for a cell + contract check before anything is loaded or resumed."""
    basis = basis_for_arm(cfg, arm, task.task_id)
    norm = load_normalizer(cfg, task.task_id)
    fpr.check_or_write(cdir, fpr.cell_fp(cfg, task.task_id, provider, sp, arm, seed, basis, norm.to_dict()), cfg, log)
    if not (cdir / "basis.npy").exists():
        np.save(cdir / "basis.npy", basis)
    assert np.array_equal(np.load(cdir / "basis.npy"), basis), "basis changed: this is a new experiment, new dir"
    return basis, norm


def _committed_eval_steps(cdir: Path) -> int:
    n = 0
    for f in (cdir / "evals").glob("eval_*.json"):
        n += sum(r["length"] for r in json.loads(f.read_text())["episodes"])
    return n


def run_cell_block(cfg, task, provider, arm, seed, target, sp, log, deadline: float | None = None) -> dict:
    from .train_sac import train_block

    cdir = cell_dir(cfg, task.task_id, arm, seed)
    cdir.mkdir(parents=True, exist_ok=True)
    basis, norm = _cell_env_parts(cfg, task, provider, arm, seed, sp, cdir, log)
    counter = StepCounter(cdir / "cost.json")
    st = read_json(cdir / "status.json", {"online_steps": 0})
    # steps spent after the last committed block (crash) stay charged, as aborted_online
    d = counter.reconcile("online_train", st["online_steps"], "aborted_online",
                          "online steps after the last committed block (crashed attempt)")
    if d:
        log(f"[{arm} s{seed}] online_train reconciled with committed status: {d:+d} (>0 = moved to aborted_online)")
    counter.reconcile("evaluation", _committed_eval_steps(cdir), "aborted_eval", "interrupted evaluation")
    task.counter = counter
    provider.counter = counter
    env = ChunkEnv(task, provider, residual_map(cfg, basis, task), norm, EpisodeSchedule(sp["train"], "train", seed),
                   bucket="online_train", settle_bucket="online_settling")
    try:
        status = train_block(cdir, env, cfg, seed, target, deadline=deadline)
    finally:
        counter.save()
    log(f"[{arm} s{seed}] online={status['online_steps']} last block: {status['blocks'][-1] if status['blocks'] else None}")
    return status


def eval_cell(cfg, task, provider, arm, seed, n_episodes, at_steps, sp, log) -> dict:
    """Deterministic held-out evaluation of the policy checkpoint taken when the cell first crossed
    `at_steps` online env steps (never `the latest model`, which may already be further)."""
    from stable_baselines3.sac.policies import SACPolicy

    from .train_sac import policy_for_step

    cdir = cell_dir(cfg, task.task_id, arm, seed)
    out = cdir / "evals" / f"eval_{at_steps:07d}_n{n_episodes}.json"
    basis, norm = _cell_env_parts(cfg, task, provider, arm, seed, sp, cdir, log)
    if out.exists():
        return json.loads(out.read_text())["summary"]
    ckpt, actual = policy_for_step(cdir / "policy_ckpts", at_steps, cfg["execution_horizon"])
    counter = StepCounter(cdir / "cost.json")
    counter.reconcile("evaluation", _committed_eval_steps(cdir), "aborted_eval", "interrupted evaluation")
    task.counter = counter
    provider.counter = counter
    env = eval_env(cfg, task, provider, residual_map(cfg, basis, task), norm, sp["evaluation"])
    policy = SACPolicy.load(ckpt, device=cfg["sac_device"])
    pol = lambda o: policy.predict(o, deterministic=True)[0]  # noqa: E731
    tmp = out.parent / f"partial_{out.name}"  # not matched by eval_*.json until committed
    try:
        summ = run_episodes(env, pol, n_episodes, tmp, video_dir=cdir / "videos" / f"{at_steps:07d}", n_videos=2)
    finally:
        counter.save()
    summ.update(arm=arm, seed=seed, task_id=task.task_id, online_steps=at_steps, policy_ckpt=ckpt.name,
                policy_actual_online_steps=actual, eval_schedule_keys=[env.schedule.get(e)[1] for e in range(n_episodes)])
    payload = json.loads(tmp.read_text())
    payload["summary"] = summ
    write_json(out, payload)  # commit point
    tmp.unlink(missing_ok=True)
    log(f"[{arm} s{seed}] eval@{at_steps} (ckpt {ckpt.name}, actual {actual}): SR={summ['success_rate']:.3f} (n={n_episodes})")
    return summ
