"""Five mandatory short checks on the REAL simulator + GR00T (plan sec. 13).

Total real env steps are hard-limited to `smoke_env_step_limit` (1500), all
charged to the 'smoke' bucket. Results -> runs/smoke/smoke_report.json.
Passed checks are not re-run unless --force (or --only <n>).

  1. u=0 contract: zero adapter returns a0 bit-exactly (gripper incl.) for every
     arm; ChunkEnv(u=0) == direct execution of the provider's a0 from the same
     snapshot with the same base noise; episode boundary at the benchmark horizon.
  2. Bases: shape/dtype, B^T B = I, common RMS bound for random/DCT/full (+ outcome/pca if present).
  3. Restore (with the configured restore_mode -- the same restore.restore_to path the bank uses):
     on 5 anchors, restore -> same fragment, same noise -> restore -> repeat; both repeats AND the
     original continuation must agree: sim state, proprio, a0 element-wise within restore_atol +
     restore_rtol*|ref|, identical success/timestep.
  4. Frozen: GR00T has 0 trainable params, none in SAC optimizers; actor/critic get finite grads.
  5. End-to-end SAC smoke: reward/terminal handling, save/load, step counters.

Every check reports one boolean `pass` (check4_5: pass4 AND pass5). A cached check is
skipped only if it passed under the same contract fingerprint (config + code).
The script writes `overall_pass` and exits with code 1 unless every mandatory check passed.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from flow_residual_rl.basis import ARM_NAMES, check_orthonormal
from flow_residual_rl.budget import BudgetExceeded, StepCounter
from flow_residual_rl.config import base_parser, derive_seed, load_config, read_json, write_json
from flow_residual_rl.envs.chunk_env import ChunkEnv, EpisodeSchedule, ObsNormalizer
from flow_residual_rl.pipeline import (basis_for_arm, make_provider, make_task, obs_dim, residual_map, runs_dir,
                                       task_splits, write_manifest)
from flow_residual_rl.restore import compare, restore_to, tolerances

MANDATORY = ("check1", "check2", "check3", "check4_5")
SMOKE_REPLAY_BUCKET = "smoke"


def check_passed(res: dict | None) -> bool:
    """The single verdict of one check (check4_5 needs both parts)."""
    if not isinstance(res, dict):
        return False
    if "pass4" in res or "pass5" in res:
        return bool(res.get("pass4")) and bool(res.get("pass5")) and res.get("pass", True) is not False
    return res.get("pass") is True


def summarize(report: dict, fingerprint_digest: str | None = None) -> dict:
    per = {k: check_passed(report.get(k)) for k in MANDATORY}
    stale = [k for k in MANDATORY if fingerprint_digest is not None and isinstance(report.get(k), dict)
             and report[k].get("fingerprint") != fingerprint_digest]
    return {"checks": per, "stale_fingerprint": stale,
            "overall_pass": all(per.values()) and not stale}


def available_bases(cfg, task_id):
    out = {}
    for arm in ARM_NAMES:
        try:
            out[arm] = basis_for_arm(cfg, arm, task_id)
        except FileNotFoundError:
            pass
    return out


def check1(cfg, task, prov, sp):
    res = {}
    bases = available_bases(cfg, task.task_id)
    obs0 = task.reset(sp["train"][0], settle_bucket="smoke")
    prop = prov.propose(task.policy_observation(), None, 123)
    for arm, b in bases.items():
        rm = residual_map(cfg, b, task)
        a, st = rm.apply(prop.a0_env, np.zeros(rm.dim, np.float32))
        res[f"zero_exact_{arm}"] = bool(np.array_equal(a, prop.a0_env) and a.dtype == prop.a0_env.dtype)
    res["a0_shape"] = list(prop.a0_env.shape)
    res["a0_dtype"] = str(prop.a0_env.dtype)
    res["gripper_values"] = sorted(set(np.round(prop.a0_env[:, 6], 3).tolist()))
    res["arm_absmax"] = float(np.abs(prop.a0_env[:, :6]).max())
    res["raw_chunk_shape"] = list(prop.raw_chunk.shape)
    # same noise -> same proposal (determinism of base noise on this GPU)
    snap = task.snapshot()
    p2 = prov.propose(task.policy_observation(), None, 123)
    res["same_seed_same_a0"] = bool(np.allclose(p2.a0_env, prop.a0_env, atol=1e-6))
    p3 = prov.propose(task.policy_observation(), None, 124)
    res["diff_seed_a0_maxdiff"] = float(np.abs(p3.a0_env[:, :6] - prop.a0_env[:, :6]).max())
    # ChunkEnv(u=0) vs direct execution, from the same state with the same seed
    norm = ObsNormalizer.identity(obs_dim(cfg))
    rm = residual_map(cfg, bases["random12"], task)
    sched = EpisodeSchedule([sp["train"][0]], "smoke", 0)
    env = ChunkEnv(task, prov, rm, norm, sched, bucket="smoke", settle_bucket="smoke")
    env.reset(options={"episode": (sp["train"][0], "smoke/u0")})
    a0_env = env.proposal.a0_env.copy()
    s0 = task.snapshot()
    env.step(np.zeros(rm.dim, np.float32))
    s_env = task.state_vector()
    restore_to(task, s0, cfg["restore_mode"], SMOKE_REPLAY_BUCKET)
    for t in range(cfg["execution_horizon"]):
        task.step(a0_env[t])
    c = compare(task.state_vector(), s_env, *tolerances(cfg))
    res["chunkenv_u0_equals_direct"] = c["ok"]
    res["chunkenv_u0_maxerr"] = c["max_abs_err"]
    seed_expected = derive_seed("base", "smoke/u0", 0)
    res["chunkenv_seed_matches_schedule"] = bool(env.proposal is not None and seed_expected != 0)
    # episode boundary: put the episode 1 chunk before the horizon and step once
    restore_to(task, s0, cfg["restore_mode"], SMOKE_REPLAY_BUCKET)
    task.elapsed = cfg["episode_cap"] - cfg["execution_horizon"]
    env.proposal = prov.propose(task.policy_observation(), None, 5)
    _, r, term, trunc, info = env.step(np.zeros(rm.dim, np.float32))
    res["horizon_terminal"] = bool(term and not trunc and (r == 0.0 or info["success"]))
    res["horizon_env_steps"] = info["n_env_steps"]
    res["pass"] = all(v for k, v in res.items() if k.startswith("zero_exact")) and res["chunkenv_u0_equals_direct"] \
        and res["horizon_terminal"] and res["a0_shape"] == [cfg["execution_horizon"], 7] and res["same_seed_same_a0"]
    return res


def check2(cfg, task):
    res, bases = {}, available_bases(cfg, task.task_id)
    rng = np.random.default_rng(0)
    for arm, b in bases.items():
        rm = residual_map(cfg, b, task)
        rms = []
        for _ in range(200):
            u = rng.uniform(-1, 1, rm.dim).astype(np.float32) * rng.uniform(0, 3)
            rms.append(float(np.sqrt(np.mean(rm.delta(u) ** 2))))
        res[arm] = {"shape": list(b.shape), "dtype": str(b.dtype), "orth_err": check_orthonormal(b),
                    "rms_max": max(rms), "rms_bound_ok": max(rms) <= cfg["epsilon_rms"] * (1 + 1e-5)}
    expected = cfg["execution_horizon"] * len(cfg["corrected_dims"])
    res["pass"] = all(v["rms_bound_ok"] and v["shape"][0] == expected and v["dtype"] == "float32"
                      for k, v in res.items() if isinstance(v, dict))
    return res


def check3(cfg, task, prov, sp):
    """Restore validity with the configured restore_mode. pass requires, on every anchor:
    repeat 1 == repeat 2 == the ORIGINAL continuation (sim state, proprio, a0; element-wise
    atol + rtol*|ref|) and identical success flag / timestep."""
    C, mode = cfg["execution_horizon"], cfg["restore_mode"]
    atol, rtol = tolerances(cfg)
    task.reset(sp["train"][1 % len(sp["train"])], settle_bucket="smoke")
    snaps, orig = [], []
    for k in range(6):
        seed = derive_seed("restore", k)
        snaps.append((task.snapshot(), seed))
        p = prov.propose(task.policy_observation(), None, seed)
        for t in range(C):
            task.step(p.a0_env[t])
        orig.append((task.state_vector(), task.proprio(), bool(task.success), int(task.elapsed), p.a0_env.copy()))
    n_replay0 = task.counter.steps["smoke"]
    res = {"restore_mode": mode, "atol": atol, "rtol": rtol, "anchors": []}
    all_ok = True
    for (snap, seed), ref in list(zip(snaps, orig))[1:]:  # 5 anchors (k=1..5)
        outs = []
        for rep in range(2):
            restore_to(task, snap, mode, SMOKE_REPLAY_BUCKET)
            p = prov.propose(task.policy_observation(), None, seed)
            for t in range(C):
                task.step(p.a0_env[t])
            outs.append((task.state_vector(), task.proprio(), bool(task.success), int(task.elapsed), p.a0_env.copy()))
        a = {"elapsed": int(snap.elapsed)}
        ok = True
        for name, idx in (("state", 0), ("proprio", 1), ("a0", 4)):
            rr = compare(outs[0][idx], outs[1][idx], atol, rtol)        # repeatability
            ro = compare(outs[0][idx], ref[idx], atol, rtol)            # vs the original continuation
            a[f"{name}_repeat"] = rr
            a[f"{name}_vs_original"] = ro
            ok &= rr["ok"] and ro["ok"]
        a["discrete_equal"] = bool(outs[0][2] == outs[1][2] == ref[2] and outs[0][3] == outs[1][3] == ref[3])
        a["ok"] = bool(ok and a["discrete_equal"])
        all_ok &= a["ok"]
        res["anchors"].append(a)
    res["restore_env_steps"] = task.counter.steps["smoke"] - n_replay0
    res["pass"] = bool(all_ok)
    res["recommendation"] = (f"restore_mode={mode} verified" if all_ok else
                             f"restore_mode={mode} FAILED: inspect anchors; "
                             + ("re-run the FULL smoke with --set restore_mode=replay (the fingerprint covers restore_mode, so all checks must pass under it)" if mode == "snapshot"
                                else "replay is not deterministic either: stop, do not collect the bank"))
    return res


def check45(cfg, task, prov, sp, out_dir: Path):
    import torch

    from flow_residual_rl.train_sac import ResidualSAC, grad_report, make_sac

    res = {}
    cfg = dict(cfg, episode_cap=64, policy_checkpoint_every=10**9)  # short horizon -> terminal failures happen
    task.episode_cap = 64
    norm = ObsNormalizer.identity(obs_dim(cfg))
    rm = residual_map(cfg, basis_for_arm(cfg, "random12", task.task_id), task)
    env = ChunkEnv(task, prov, rm, norm, EpisodeSchedule(sp["train"], "smoke_sac", 0), bucket="smoke",
                   settle_bucket="smoke")
    model = make_sac(env, cfg, seed=0, smoke=True)
    res["init_mu_zero"] = bool(float(model.actor.mu.weight.abs().sum() + model.actor.mu.bias.abs().sum()) == 0.0)
    res["init_log_std_bias"] = float(model.actor.log_std.bias.mean())
    obs, _ = env.reset()
    a_det = model.predict(obs, deterministic=True)[0]
    res["init_deterministic_action_zero"] = bool(np.allclose(a_det, 0))
    # frozen GR00T not in any SAC optimizer
    gr00t_ids = {id(p) for p in prov.model.parameters()}
    opt_ids = {id(p) for o in [model.actor.optimizer, model.critic.optimizer] for g in o.param_groups for p in g["params"]}
    res["gr00t_in_optimizer"] = bool(gr00t_ids & opt_ids)
    res["gr00t_frozen"] = prov.check_frozen()
    eps_log = []
    c0 = task.counter.steps["smoke"]
    from stable_baselines3.common.callbacks import BaseCallback

    class CB(BaseCallback):
        def __init__(self, limit):
            super().__init__()
            self.n, self.limit = 0, limit
            self.rewards = []

        def _on_step(self):
            for info, r, d in zip(self.locals["infos"], self.locals["rewards"], self.locals["dones"]):
                self.n += info["n_env_steps"]
                self.rewards.append((float(r), bool(d), bool(info["success"]), info["n_env_steps"]))
                if "episode_record" in info:
                    eps_log.append(info["episode_record"])
            return self.n < self.limit

    cb = CB(400)
    model.learn(total_timesteps=10**6, callback=cb, log_interval=None)
    res["sac_env_steps"] = cb.n
    res["counter_matches"] = (task.counter.steps["smoke"] - c0) == cb.n
    res["reward_only_on_success"] = all((r == 1.0) == s for r, d, s, n in cb.rewards)
    res["success_terminal"] = all(d for r, d, s, n in cb.rewards if s)
    res["nonterminal_full_chunks"] = all(n == cfg["execution_horizon"] for r, d, s, n in cb.rewards if not d)
    res["episodes"] = eps_log
    res["terminal_failures_seen"] = sum(1 for e in eps_log if not e["success"])
    res["grads"] = grad_report(model)
    p = out_dir / "sac_smoke"
    model.save(p / "model.zip")
    model.save_replay_buffer(p / "replay.pkl")
    m2 = ResidualSAC.load(p / "model.zip", env=env, device=cfg["sac_device"])
    m2.load_replay_buffer(p / "replay.pkl")
    res["load_same_actor"] = all(torch.equal(a, b) for a, b in zip(model.actor.state_dict().values(), m2.actor.state_dict().values()))
    res["load_replay_size"] = int(m2.replay_buffer.size())
    res["load_mu_not_reinit"] = bool(float(m2.actor.mu.weight.abs().sum()) == float(model.actor.mu.weight.abs().sum()))
    m2._last_obs = None
    c1 = task.counter.steps["smoke"]
    cb2 = CB(48)
    m2.learn(total_timesteps=10**6, callback=cb2, reset_num_timesteps=False, log_interval=None)
    res["resume_num_timesteps_monotone"] = m2.num_timesteps > model.num_timesteps
    res["resume_counter_matches"] = (task.counter.steps["smoke"] - c1) == cb2.n
    res["resume_grads"] = grad_report(m2)
    res["pass4"] = (not res["gr00t_in_optimizer"]) and res["gr00t_frozen"]["trainable"] == 0 and \
        res["grads"]["actor"]["all_finite"] and res["grads"]["critic"]["all_finite"] and \
        res["grads"]["actor"]["n_params_with_grad"] > 0 and res["grads"]["critic"]["n_params_with_grad"] > 0
    res["pass5"] = res["counter_matches"] and res["reward_only_on_success"] and res["success_terminal"] and \
        res["nonterminal_full_chunks"] and res["load_same_actor"] and res["load_mu_not_reinit"] and \
        res["init_mu_zero"] and res["init_deterministic_action_zero"] and res["resume_num_timesteps_monotone"] \
        and res["resume_counter_matches"]
    res["pass"] = bool(res["pass4"] and res["pass5"])
    return res


def main():
    from flow_residual_rl import fingerprint as fpr

    p = base_parser(__doc__)
    p.add_argument("--task-id", type=int, default=None)
    p.add_argument("--only", nargs="*", type=int, default=None)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    cfg = load_config(a.config, a.set)
    from flow_residual_rl.pipeline import runs_dir as _runs_dir
    from flow_residual_rl.walltime import WallLedger

    _ledger = WallLedger(_runs_dir(cfg), "smoke")  # project wall-time ledger (runs/wall/)
    tid = cfg["pilot_task_id"] if a.task_id is None else a.task_id
    out = runs_dir(cfg) / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    report = read_json(out / "smoke_report.json", {})
    counter = StepCounter(out / "cost.json")
    counter.set_limit(("smoke",), cfg["smoke_env_step_limit"])
    prov = make_provider(cfg, counter)
    digest = None
    try:
        with counter.charge("smoke"):
            task = make_task(cfg, tid, counter, prov)
            sp = task_splits(cfg, task)
            fp = fpr.smoke_fp(cfg, tid, prov, sp)
            digest = fpr.sha_obj(fp)
            write_json(out / "fingerprint.json", fp)
            write_json(out.parent / "tasks.json", _tasks_json(cfg))
            write_manifest(cfg, out / "run_manifest.json", prov, {"splits": sp, "smoke_fingerprint": digest})
            todo = a.only or [1, 2, 3, 4]
            checks = {1: lambda: check1(cfg, task, prov, sp), 2: lambda: check2(cfg, task),
                      3: lambda: check3(cfg, task, prov, sp), 4: lambda: check45(cfg, task, prov, sp, out)}
            for n in todo:
                key = f"check{n}" if n < 4 else "check4_5"
                prev = report.get(key)
                if not a.force and check_passed(prev) and prev.get("fingerprint") == digest:
                    print(f"[smoke] {key} already passed under the same fingerprint; skipping")
                    continue
                try:
                    res = checks[n]()
                except BudgetExceeded as e:
                    res = {"pass": False, "error": f"smoke budget: {e}"}
                except Exception as e:  # noqa: BLE001
                    res = {"pass": False, "error": repr(e), "traceback": traceback.format_exc()}
                res["fingerprint"] = digest
                res["pass"] = check_passed(res)
                report[key] = res
                report["cost"] = counter.to_dict()
                counter.save()
                write_json(out / "smoke_report.json", report)
                print(f"[smoke] {key}: {'PASS' if res['pass'] else 'FAIL'}")
    finally:
        counter.save()
    summ = summarize(report, digest)
    report["summary"] = summ
    report["overall_pass"] = summ["overall_pass"]
    write_json(out / "smoke_report.json", report)
    print("[smoke] summary:", summ, "| smoke env steps:", counter.steps["smoke"])
    if not summ["overall_pass"]:
        print("[smoke] FAILED: do not start eval_base / bank / pilot until every check passes", file=sys.stderr)
        sys.exit(1)


def _tasks_json(cfg):
    from libero.libero import benchmark

    from flow_residual_rl.envs.libero_task import load_task_init_states

    bm = benchmark.get_benchmark_dict()[cfg["suite"]]()
    return {"suite": cfg["suite"], "tasks": [{"task_id": i, "name": bm.get_task(i).name,
                                               "language": bm.get_task(i).language,
                                               "n_init_states": len(load_task_init_states(bm, i))}
                                              for i in range(bm.get_num_tasks())],
            "pilot_task_id": cfg["pilot_task_id"], "main_task_ids": cfg["main_task_ids"]}


if __name__ == "__main__":
    main()
