"""Stage A / Stage B pilot launcher with resume (plan sec. 14).

Stage A: task pilot_task_id, seed training_seeds[0], arms stage_a_arms;
         blocks of stage_a_block_steps up to stage_a_online_final_steps, arms
         interleaved per block; eval at the steps listed in stage_a_eval_episodes.
Stage B: tasks main_task_ids x seeds training_seeds x arms stage_b_arms to
         stage_b_online_steps; Stage A cells are continued, not restarted.
         Requires reports/stageA_decision.json == continue (or --force).
Completed blocks/evals are skipped; crashed blocks resume from the last block
boundary (uncommitted steps are booked as aborted_*, never reset). Every cached
artefact is checked against its contract fingerprint before reuse.

Wall-hour ceilings use the project ledger (walltime.py: every command, incl. separately run
smoke/eval_base/collect_bank/train_basis and crashed attempts up to their last heartbeat):
Stage A stops at project total >= stage_a_gpu_hours; Stage B at >= project_gpu_hours. The ceiling
is also passed into each SAC block as a deadline, so a long block stops early
(committed cleanly) instead of overrunning; evaluations are not interrupted.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flow_residual_rl.budget import StepCounter
from flow_residual_rl.config import base_parser, load_config, read_json, write_json
from flow_residual_rl.pipeline import make_provider, make_task, runs_dir, task_dir, task_splits, write_manifest
from flow_residual_rl.walltime import WallLedger, breakdown
from flow_residual_rl.stages import build_bank, build_bases, eval_base, eval_cell, log_to, run_cell_block


def ensure_task_ready(cfg, tid, prov, log):
    meta = build_bank(cfg, tid, prov)
    st = build_bases(cfg, tid, reuse=True)  # reuses a finished basis only if its fingerprint matches
    log(f"[task {tid}] bank groups={meta['n_groups']} disc(train/val)={meta['n_discordant_train']}/"
        f"{meta['n_discordant_val']} basis={st['status']}")
    return st["status"] == "ok"


def main():
    p = base_parser(__doc__)
    p.add_argument("--stage", choices=["A", "B"], required=True)
    p.add_argument("--resume", action="store_true", help="(default behaviour) skip completed blocks/evals")
    p.add_argument("--force", action="store_true", help="Stage B without a 'continue' Stage A decision")
    a = p.parse_args()
    cfg = load_config(a.config, a.set)
    root = runs_dir(cfg)
    log = log_to(root / f"stage{a.stage}.log")
    if a.stage == "A":
        tasks, seeds, arms = [cfg["pilot_task_id"]], cfg["training_seeds"][:1], list(cfg["stage_a_arms"])
        final, block = cfg["stage_a_online_final_steps"], cfg["stage_a_block_steps"]
        evals = {int(k): v for k, v in cfg["stage_a_eval_episodes"].items()}
        gpu_h_limit = cfg["stage_a_gpu_hours"]  # project total so far (preparation included)
    else:
        dec = read_json(Path("reports") / "stageA_decision.json", {})
        if dec.get("decision") != "continue" and not a.force:
            sys.exit(f"Stage A decision is {dec.get('decision')!r}; run report first or pass --force")
        tasks, seeds, arms = cfg["main_task_ids"], cfg["training_seeds"], list(cfg["stage_b_arms"])
        final, block = cfg["stage_b_online_steps"], cfg["stage_b_block_steps"]
        evals = {int(k): v for k, v in cfg["stage_b_eval_episodes"].items()}
        gpu_h_limit = cfg["project_gpu_hours"]  # whole project, Stage A and preparation included
    stage_status_p = root / f"stage{a.stage}_status.json"
    sst = read_json(stage_status_p, {"done": False})
    ledger = WallLedger(root, f"run_pilot_{a.stage}")
    wall0 = ledger.project_hours()
    deadline = time.time() + (gpu_h_limit - wall0) * 3600
    log(f"[stage {a.stage}] project wall-hour ceiling {gpu_h_limit:.2f} h, already used {wall0:.2f} h "
        f"{ {k: round(v, 2) for k, v in breakdown(root).items()} }")
    if wall0 >= gpu_h_limit:
        log(f"[stop] project wall-hour ceiling {gpu_h_limit:.2f} already reached")
        return
    prov = make_provider(cfg)
    write_manifest(cfg, root / f"stage{a.stage}_run_manifest.json", prov)

    def wall_hours():
        return ledger.project_hours()

    def stop(where):
        log(f"[stop] wall-hour ceiling {gpu_h_limit:.2f} reached ({where}); partial report")
        write_json(stage_status_p, dict(sst, wall_hours=wall_hours(), stopped_at=where))

    ready = {tid: ensure_task_ready(cfg, tid, prov, log) for tid in tasks}
    task_objs, sps = {}, {}
    for tid in tasks:
        base_counter = StepCounter(task_dir(cfg, tid) / "base_eval" / "cost.json")
        task_objs[tid] = make_task(cfg, tid, base_counter, prov)
        sps[tid] = task_splits(cfg, task_objs[tid])
        n_base = max(evals.values())
        r = eval_base(cfg, tid, prov, task_objs[tid], n_base, base_counter)
        log(f"[task {tid}] base VLA SR={r['success_rate']:.3f} on {r['n_episodes']} eval episodes "
            f"({r['n_unique_states']} unique init states)")
    targets = list(range(block, final + 1, block))
    if targets[-1] != final:
        targets.append(final)
    bad = [at for at in evals if at not in targets and at % cfg["policy_checkpoint_every"]]
    assert not bad, f"eval points {bad} are neither block targets nor multiples of policy_checkpoint_every"
    for target in targets:
        for tid in tasks:
            arms_t = [x for x in arms if ready[tid] or x != "outcome12"]
            for seed in seeds:
                for arm in arms_t:
                    if wall_hours() > gpu_h_limit:
                        return stop(f"before block {target}")
                    st = run_cell_block(cfg, task_objs[tid], prov, arm, seed, target, sps[tid], log, deadline=deadline)
                    for at, n in sorted(evals.items()):
                        if at <= st["online_steps"]:  # eval_cell skips evals that are already committed
                            eval_cell(cfg, task_objs[tid], prov, arm, seed, n, at, sps[tid], log)
                    if st["online_steps"] < target:
                        return stop(f"inside block {target}, {arm} s{seed} at {st['online_steps']}")
                    write_json(stage_status_p, dict(sst, wall_hours=wall_hours(), last=[target, tid, seed, arm]))
        log(f"[block] all cells reached {target} online env steps; wall {wall_hours():.2f} h")
    write_json(stage_status_p, dict(sst, wall_hours=wall_hours(), done=True))
    log("[done] run: python -m flow_residual_rl.report --runs runs/ --output reports/pilot/")


if __name__ == "__main__":
    main()
