"""Train + freeze the outcome basis (outcome12) and the PCA control (pca12) for one task.

CPU is enough. Exits with code 3 and writes outcome12_status.json with
status=insufficient_outcome_signal if the bank lacks discordant groups.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flow_residual_rl.config import base_parser, load_config
from flow_residual_rl.stages import build_bases

if __name__ == "__main__":
    p = base_parser(__doc__)
    p.add_argument("--task-id", type=int, required=True)
    a = p.parse_args()
    cfg = load_config(a.config, a.set)
    from flow_residual_rl.pipeline import runs_dir as _runs_dir
    from flow_residual_rl.walltime import WallLedger

    _ledger = WallLedger(_runs_dir(cfg), "train_basis")  # project wall-time ledger (runs/wall/)
    r = build_bases(cfg, a.task_id)
    print(json.dumps(r, indent=1, default=float))
    sys.exit(0 if r["status"] == "ok" else 3)
