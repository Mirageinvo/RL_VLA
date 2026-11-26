"""Base VLA (zero correction) on held-out init states (plan sec. 5).

--split evaluation (default): sanity check and the paired reference of every cell evaluation;
cached in runs/<suite>_task<id>/base_eval/episodes.jsonl (same paired schedule as run_pilot /
run_fixed_pilot). --split validation: task selection without touching the evaluation starts;
cached in base_eval_validation/. Can run before the bank exists.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flow_residual_rl.budget import StepCounter
from flow_residual_rl.config import base_parser, load_config
from flow_residual_rl.pipeline import make_provider, make_task, write_manifest
from flow_residual_rl.stages import base_eval_dir, eval_base

if __name__ == "__main__":
    p = base_parser(__doc__)
    p.add_argument("--task-id", type=int, required=True)
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--split", choices=["evaluation", "validation"], default="evaluation")
    a = p.parse_args()
    cfg = load_config(a.config, a.set)
    from flow_residual_rl.pipeline import runs_dir as _runs_dir
    from flow_residual_rl.walltime import WallLedger

    _ledger = WallLedger(_runs_dir(cfg), "eval_base")  # project wall-time ledger (runs/wall/)
    out = base_eval_dir(cfg, a.task_id, a.split)
    counter = StepCounter(out / "cost.json")
    prov = make_provider(cfg, counter)
    task = make_task(cfg, a.task_id, counter, prov)
    r = eval_base(cfg, a.task_id, prov, task, a.episodes or cfg["baseline_eval_episodes"], counter, split=a.split)
    write_manifest(cfg, out / "run_manifest.json", prov)
    print(r, counter.to_dict())
