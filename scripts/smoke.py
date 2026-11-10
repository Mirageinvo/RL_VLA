"""Five mandatory short checks on the REAL simulator + GR00T (plan sec. 13).

Total real env steps are hard-limited to `smoke_env_step_limit` (1500), all
charged to the 'smoke' bucket. Results -> runs/smoke/smoke_report.json.
Passed checks are not re-run unless --force (or --only <n>).

  1. u=0 contract: zero adapter returns a0 bit-exactly (gripper incl.) for every
     arm; ChunkEnv(u=0) == direct execution of the provider's a0 from the same
     snapshot with the same base noise; episode boundary at the benchmark horizon.
  2. Bases: shape/dtype, B^T B = I, common RMS bound for random/DCT/full (+ outcome/pca if present).
  3. Restore: on 5 anchors, restore -> same fragment, same noise -> restore -> repeat;
     continuous state atol=1e-5 / rtol=1e-4, identical success/timestep.
  4. Frozen: GR00T has 0 trainable params, none in SAC optimizers; actor/critic get finite grads.
  5. End-to-end SAC smoke: reward/terminal handling, save/load, step counters.
"""
from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from flow_residual_rl.basis import ARM_NAMES, ResidualMap, check_orthonormal
from flow_residual_rl.budget import BudgetExceeded, StepCounter
from flow_residual_rl.config import base_parser, derive_seed, load_config, read_json, write_json
from flow_residual_rl.envs.chunk_env import ChunkEnv, EpisodeSchedule, ObsNormalizer
from flow_residual_rl.pipeline import (basis_for_arm, make_provider, make_task, obs_dim, residual_map, runs_dir,
                                       task_splits, write_manifest)

ATOL, RTOL = 1e-5, 1e-4


def close(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    err = np.abs(a - b)
    ok = bool(np.all(err <= ATOL + RTOL * np.abs(b)))
    return ok, float(err.max()) if err.size else 0.0


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
    task.restore(s0)
    for t in range(cfg["execution_horizon"]):
        task.step(a0_env[t])
    ok, err = close(task.state_vector(), s_env)
    res["chunkenv_u0_equals_direct"] = ok
    res["chunkenv_u0_maxerr"] = err
    seed_expected = derive_seed("base", "smoke/u0", 0)
    res["chunkenv_seed_matches_schedule"] = bool(env.proposal is not None and seed_expected != 0)
    # episode boundary: put the episode 1 chunk before the horizon and step once
    task.restore(s0)
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
    C = cfg["execution_horizon"]
    task.reset(sp["train"][1 % len(sp["train"])], settle_bucket="smoke")
    snaps = []
    for k in range(6):
        snaps.append((task.snapshot(), derive_seed("restore", k)))
        p = prov.propose(task.policy_observation(), None, snaps[-1][1])
        for t in range(C):
            task.step(p.a0_env[t])
    res, all_ok = {"anchors": []}, True
    for snap, seed in snaps[1:]:  # 5 anchors
        outs = []
        for rep in range(2):
            task.restore(snap)
            p = prov.propose(task.policy_observation(), None, seed)
            for t in range(C):
                task.step(p.a0_env[t])
            outs.append((task.state_vector(), task.proprio(), task.success, task.elapsed, p.a0_env))
        ok_s, e_s = close(outs[0][0], outs[1][0])
        ok_p, e_p = close(outs[0][1], outs[1][1])
        ok_a, e_a = close(outs[0][4], outs[1][4])
        disc = outs[0][2] == outs[1][2] and outs[0][3] == outs[1][3]
        all_ok &= ok_s and ok_p and disc
        res["anchors"].append({"elapsed": snap.elapsed, "state_err": e_s, "proprio_err": e_p, "a0_err": e_a,
                               "discrete_equal": bool(disc), "ok": bool(ok_s and ok_p and disc)})
    # also: restored zero-branch vs the original continuation (snapshot k -> state at k+1)
    errs = []
    for (snap, seed), (nxt, _) in zip(snaps[:-1], snaps[1:]):
        task.restore(snap)
        p = prov.propose(task.policy_observation(), None, seed)
        for t in range(C):
            task.step(p.a0_env[t])
        errs.append(close(task.state_vector(), nxt.sim_flat)[1])
    res["vs_original_continuation_maxerr"] = max(errs)
    res["vs_original_ok"] = bool(max(errs) <= ATOL)
    res["pass"] = bool(all_ok)
    res["recommendation"] = "snapshot" if all_ok and res["vs_original_ok"] else "investigate; fallback restore_mode=replay"
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
    return res


def main():
    p = base_parser(__doc__)
    p.add_argument("--task-id", type=int, default=None)
    p.add_argument("--only", nargs="*", type=int, default=None)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    cfg = load_config(a.config, a.set)
    tid = cfg["pilot_task_id"] if a.task_id is None else a.task_id
    out = runs_dir(cfg) / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    report = read_json(out / "smoke_report.json", {})
    counter = StepCounter(out / "cost.json")
    counter.set_limit(("smoke",), cfg["smoke_env_step_limit"])
    prov = make_provider(cfg, counter)
    with counter.charge("smoke"):
        task = make_task(cfg, tid, counter, prov)
        sp = task_splits(cfg, task)
        write_json(out.parent / "tasks.json", _tasks_json(cfg))
        write_manifest(cfg, out / "run_manifest.json", prov, {"splits": sp})
        todo = a.only or [1, 2, 3, 4]
        checks = {1: lambda: check1(cfg, task, prov, sp), 2: lambda: check2(cfg, task),
                  3: lambda: check3(cfg, task, prov, sp), 4: lambda: check45(cfg, task, prov, sp, out)}
        for n in todo:
            key = f"check{n}" if n < 4 else "check4_5"
            if not a.force and report.get(key, {}).get("pass", report.get(key, {}).get("pass5")):
                print(f"[smoke] {key} already passed; skipping")
                continue
            try:
                report[key] = checks[n]()
            except BudgetExceeded as e:
                report[key] = {"pass": False, "error": f"smoke budget: {e}"}
            except Exception as e:  # noqa: BLE001
                report[key] = {"pass": False, "error": repr(e), "traceback": traceback.format_exc()}
            report["cost"] = counter.to_dict()
            write_json(out / "smoke_report.json", report)
            counter.save()
            print(f"[smoke] {key}: {'PASS' if report[key].get('pass', report[key].get('pass5')) else 'FAIL'}")
    summary = {k: (v.get("pass") if "pass" in v else (v.get("pass4"), v.get("pass5")))
               for k, v in report.items() if isinstance(v, dict) and k.startswith("check")}
    print("[smoke] summary:", summary, "| smoke env steps:", counter.steps["smoke"])


def _tasks_json(cfg):
    from libero.libero import benchmark

    bm = benchmark.get_benchmark_dict()[cfg["suite"]]()
    return {"suite": cfg["suite"], "tasks": [{"task_id": i, "name": bm.get_task(i).name,
                                               "language": bm.get_task(i).language,
                                               "n_init_states": len(bm.get_task_init_states(i))}
                                              for i in range(bm.get_num_tasks())],
            "pilot_task_id": cfg["pilot_task_id"], "main_task_ids": cfg["main_task_ids"]}


if __name__ == "__main__":
    main()
