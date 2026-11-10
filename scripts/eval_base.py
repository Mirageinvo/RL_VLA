"""Base VLA (zero correction) on held-out evaluation init states (plan sec. 5).

Sanity check and ceiling reference, not a replica of the published number.
Episodes are cached in runs/<suite>_task<id>/base_eval/episodes.jsonl and reused
by run_pilot (same paired schedule). Can run before the bank exists.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flow_residual_rl.budget import StepCounter
from flow_residual_rl.config import base_parser, load_config
from flow_residual_rl.pipeline import make_provider, make_task, task_dir, write_manifest
from flow_residual_rl.stages import eval_base

if __name__ == "__main__":
    p = base_parser(__doc__)
    p.add_argument("--task-id", type=int, required=True)
    p.add_argument("--episodes", type=int, default=None)
    a = p.parse_args()
    cfg = load_config(a.config, a.set)
    counter = StepCounter(task_dir(cfg, a.task_id) / "base_eval" / "cost.json")
    prov = make_provider(cfg, counter)
    task = make_task(cfg, a.task_id, counter, prov)
    r = eval_base(cfg, a.task_id, prov, task, a.episodes or cfg["baseline_eval_episodes"], counter)
    write_manifest(cfg, task_dir(cfg, a.task_id) / "base_eval" / "run_manifest.json", prov)
    print(r, counter.to_dict())
