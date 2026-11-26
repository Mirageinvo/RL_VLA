"""Short RL pilot with FIXED bases on one harder task (no interventions bank, no outcome basis).

Arms (rank 12, same linear latent map, epsilon, SAC settings, seed, eval schedule):
    dct12    DCT modes 0 and 1                                           control
    tail12   DCT mode 0 (mean command) + 6 lag_tail shape directions     candidate
             (fitted on the task-0 Jacobian diagnostic, transferred frozen: tail_basis.py)

Steps (each cached / resumable; rerunning continues where it stopped):
  1. observation normaliser from base-GR00T rollouts on TRAIN starts (<task>/obs_norm/) unless the
     task has a bank normaliser;
  2. tail12 basis -> <task>/basis/tail12/ (from --tail-source);
  3. base GR00T on the EVALUATION starts (<task>/base_eval/, the paired reference);
  4. SAC blocks of --block-steps online env steps, arms interleaved, up to --steps;
  5. deterministic evaluation of each arm at --steps on the same evaluation starts;
  6. paired summary vs the frozen base on the same episodes: SR, rescued (base fail -> arm success),
     harmed (base success -> arm fail), actual correction RMS, clipping; tail12 vs dct12 paired.
Choose the task BEFORE running (eval_base.py --split validation) and keep it fixed.

  python scripts/run_fixed_pilot.py --config configs/pilot.yaml --task-id 3
Output: runs/<suite>_task<id>/cells/{dct12,tail12}_seed<s>/, fixed_pilot_summary.{json,md}
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from flow_residual_rl.budget import StepCounter
from flow_residual_rl.config import base_parser, load_config, write_json
from flow_residual_rl.pipeline import load_normalizer, make_provider, make_task, runs_dir, task_dir, task_splits, write_manifest


def paired(base: list, arm: list) -> dict:
    """Episode-by-episode comparison on the same evaluation schedule."""
    n = min(len(base), len(arm))
    for b, x in zip(base[:n], arm[:n]):
        if (b["init_state_id"], b["key"]) != (x["init_state_id"], x["key"]):
            raise ValueError(f"episode schedules differ: {b['init_state_id'], b['key']} vs {x['init_state_id'], x['key']}")
    bs = np.array([bool(b["success"]) for b in base[:n]])
    xs = np.array([bool(x["success"]) for x in arm[:n]])
    return {"n": n, "sr_ref": float(bs.mean()) if n else None, "sr": float(xs.mean()) if n else None,
            "rescued": int(np.sum(~bs & xs)), "harmed": int(np.sum(bs & ~xs)),
            "ref_failures": int(np.sum(~bs)), "ref_successes": int(np.sum(bs))}


def summarize(base_recs, arm_recs: dict, at_steps: int) -> dict:
    out = {"at_online_steps": at_steps, "arms": {}}
    for arm, (recs, summ) in arm_recs.items():
        out["arms"][arm] = dict(paired(base_recs, recs), rms_median=summ.get("rms_median"),
                                clip_mean=summ.get("clip_mean"), mean_length=summ.get("mean_length"))
    if "dct12" in arm_recs and "tail12" in arm_recs:
        out["tail12_vs_dct12"] = paired(arm_recs["dct12"][0], arm_recs["tail12"][0])
    return out


def render(s, task_id, seed) -> str:
    L = [f"# Fixed-basis RL pilot, task {task_id}, seed {seed}, eval at {s['at_online_steps']} online steps", "",
         f"base GR00T SR on the same {s['base']['n']} evaluation episodes: {s['base']['sr']:.3f}", "",
         "| arm | SR | rescued (of base failures) | harmed (of base successes) | rms median | clip mean |",
         "|---|---|---|---|---|---|"]
    for arm, v in s["arms"].items():
        L.append(f"| {arm} | {v['sr']:.3f} | {v['rescued']} / {v['ref_failures']} | {v['harmed']} / {v['ref_successes']} | "
                 f"{v['rms_median']:.4f} | {v['clip_mean']:.4f} |")
    if "tail12_vs_dct12" in s:
        v = s["tail12_vs_dct12"]
        L += ["", f"tail12 vs dct12 on the same episodes: tail12 succeeds where dct12 fails {v['rescued']}x, "
                  f"fails where dct12 succeeds {v['harmed']}x (n={v['n']})"]
    return "\n".join(L) + "\n"


def main():
    p = base_parser(__doc__)
    p.add_argument("--task-id", type=int, required=True)
    p.add_argument("--arms", nargs="*", default=["dct12", "tail12"])
    p.add_argument("--steps", type=int, default=20000, help="online env steps per arm")
    p.add_argument("--block-steps", type=int, default=5000)
    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=None, help="default: training_seeds[0]")
    p.add_argument("--obs-norm-episodes", type=int, default=10)
    p.add_argument("--tail-source", default=None, help="Jacobian diagnostic dir (default runs/diagnostics/jacobian_task0)")
    a = p.parse_args()
    cfg = load_config(a.config, a.set)
    from flow_residual_rl.stages import base_eval_dir, eval_base, eval_cell, fit_nominal_normalizer, log_to, run_cell_block
    from flow_residual_rl.tail_basis import build_tail12
    from flow_residual_rl.walltime import WallLedger

    tid, seed = a.task_id, a.seed if a.seed is not None else cfg["training_seeds"][0]
    tdir = task_dir(cfg, tid)
    tdir.mkdir(parents=True, exist_ok=True)
    log = log_to(tdir / "fixed_pilot.log")
    _ledger = WallLedger(runs_dir(cfg), "run_fixed_pilot")
    prov = make_provider(cfg)
    write_manifest(cfg, tdir / "fixed_pilot_run_manifest.json", prov, {"args": vars(a)})

    # 1. normaliser
    try:
        load_normalizer(cfg, tid)
        log(f"[task {tid}] observation normaliser present")
    except FileNotFoundError:
        d = fit_nominal_normalizer(cfg, tid, prov, a.obs_norm_episodes)
        log(f"[task {tid}] normaliser fit on {d['n']} boundaries of {d['n_episodes']} base train episodes "
            f"(base train SR {d['base_success_rate_train']:.2f})")
    # 2. tail12 basis
    if "tail12" in a.arms and not (tdir / "basis" / "tail12" / "basis.npy").exists():
        src = Path(a.tail_source) if a.tail_source else runs_dir(cfg) / "diagnostics" / "jacobian_task0"
        meta = build_tail12(src, tdir / "basis" / "tail12")
        log(f"[task {tid}] tail12 built from {src}: overlap of its shape part with DCT mode 1 "
            f"{meta['shape_overlap_with_dct_mode1']:.3f}, sum leak {meta['shape_sum_leak']:.1e}")
    # 3. base reference on the evaluation starts
    bdir = base_eval_dir(cfg, tid, "evaluation")
    counter = StepCounter(bdir / "cost.json")
    task = make_task(cfg, tid, counter, prov)
    try:
        b = eval_base(cfg, tid, prov, task, a.eval_episodes, counter)
        log(f"[task {tid}] base GR00T on evaluation starts: SR={b['success_rate']:.3f} (n={b['n_episodes']})")
        sp = task_splits(cfg, task)
        # 4. training blocks, arms interleaved
        for target in list(range(a.block_steps, a.steps, a.block_steps)) + [a.steps]:
            for arm in a.arms:
                run_cell_block(cfg, task, prov, arm, seed, target, sp, log)
        # 5. evaluation at the final step count
        summ = {arm: eval_cell(cfg, task, prov, arm, seed, a.eval_episodes, a.steps, sp, log) for arm in a.arms}
    finally:
        task.close()
    # 6. paired summary
    from flow_residual_rl.stages import cell_dir

    base_recs = [json.loads(x) for x in (bdir / "episodes.jsonl").read_text().splitlines() if x.strip()][:a.eval_episodes]
    arm_recs = {}
    for arm in a.arms:
        f = cell_dir(cfg, tid, arm, seed) / "evals" / f"eval_{a.steps:07d}_n{a.eval_episodes}.json"
        arm_recs[arm] = (json.loads(f.read_text())["episodes"], summ[arm])
    s = summarize(base_recs, arm_recs, a.steps)
    s["base"] = {"n": len(base_recs), "sr": float(np.mean([r["success"] for r in base_recs]))}
    s.update(task_id=tid, seed=seed, arms_order=a.arms, block_steps=a.block_steps)
    write_json(tdir / "fixed_pilot_summary.json", s)
    (tdir / "fixed_pilot_summary.md").write_text(render(s, tid, seed))
    print((tdir / "fixed_pilot_summary.md").read_text())


if __name__ == "__main__":
    main()
