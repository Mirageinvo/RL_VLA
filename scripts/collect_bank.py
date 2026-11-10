"""Collect the paired-intervention bank for one task (plan sec. 8).

Hard limit bank_budget_steps (30k) over all bank_* buckets; one rescue portion up
to bank_max_steps_with_rescue (50k) if discordant groups are insufficient.
Output: runs/<suite>_task<id>/bank/{bank.npz, groups.jsonl, bank_metadata.json, obs_norm.json, cost.json}
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flow_residual_rl.config import base_parser, load_config
from flow_residual_rl.pipeline import make_provider, task_dir, write_manifest
from flow_residual_rl.stages import build_bank

if __name__ == "__main__":
    p = base_parser(__doc__)
    p.add_argument("--task-id", type=int, required=True)
    a = p.parse_args()
    cfg = load_config(a.config, a.set)
    prov = make_provider(cfg)
    meta = build_bank(cfg, a.task_id, prov)
    write_manifest(cfg, task_dir(cfg, a.task_id) / "bank" / "run_manifest.json", prov)
    print({k: meta.get(k) for k in ["n_groups", "n_discordant_train", "n_discordant_val", "rescue_used",
                                    "restore_err_vs_pool_max"]}, meta["cost"]["bank_total"])
