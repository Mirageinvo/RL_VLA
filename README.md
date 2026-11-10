# flow_residual_rl — outcome-supervised correction subspace for residual SAC on frozen GR00T N1.7

Frozen GR00T N1.7 (LIBERO-10 checkpoint) proposes an 8-step action chunk; a small SB3 SAC policy
outputs `u ∈ [-1,1]^d`, mapped through a fixed basis `B ∈ R^{48×d}` to a bounded arm correction
(`RMS ≤ ε = 0.10` for every rank). Arms: `outcome12` (B learned from paired-intervention outcomes),
`random12`, `dct12`, `pca12`, `full48`. Full spec: the agent plan (sec. 1–20).

**Status:** implemented; contract tested on CPU with fakes only (`pytest`, 20 tests).
**Nothing has run on the simulator or with GR00T yet**; the first real result is `scripts/smoke.py` on the cluster.

## Cluster setup (V100, Docker)

Prerequisites
* HF token with access to `nvidia/GR00T-N1.7-LIBERO` **and** `nvidia/Cosmos-Reason2-2B`
  (the backbone is fetched from HF at model construction, `gr00t/model/modules/qwen3_backbone.py`; it may be gated — accept the license on HF).
* NVIDIA driver able to run CUDA 12.6 user-space (torch 2.9.0 **cu126**: the cu128/cu130 wheels dropped sm_70).

```bash
git clone git@github.com:Mirageinvo/RL_VLA.git ~/code/RL_VLA && cd ~/code/RL_VLA
bash docker/build.sh                 # image x86_64/rlvla:latest (GR00T @1e50c01, LIBERO @8f1084e, SB3 2.7.0, no flash-attn)
bash docker/start.sh && bash docker/into.sh
# inside the container (cwd = repo; $HOME mounted at the same path):
export HF_TOKEN=...   HF_HOME=~/.cache/huggingface
hf download nvidia/GR00T-N1.7-LIBERO --include "libero_10/*" --local-dir checkpoints/GR00T-N1.7-LIBERO
python -m pytest -q                  # CPU contract tests (fakes; not an experiment)
```

## Run order (each step resumes / skips finished work)

```bash
python scripts/smoke.py --config configs/pilot.yaml               # 5 checks on the real sim, ≤1500 env steps
python scripts/eval_base.py --config configs/pilot.yaml --task-id 0   # base VLA, 25 held-out episodes
python scripts/collect_bank.py --config configs/pilot.yaml --task-id 0  # ≤30k env steps (+1 rescue ≤50k)
python scripts/train_basis.py --config configs/pilot.yaml --task-id 0   # outcome12 + pca12, CPU, exit 3 = insufficient signal
python scripts/run_pilot.py --config configs/pilot.yaml --stage A --resume
python -m flow_residual_rl.report --runs runs/ --output reports/pilot/  # table, curves, cost, Stage-A decision
python scripts/run_pilot.py --config configs/pilot.yaml --stage B --resume  # only if decision == continue
```
(`run_pilot` also builds a missing bank/basis itself.) Long runs: inside `tmux`; overrides via `--set key=value`.
Use `PYTHONPATH=src` or `pip install -e .` for `python -m flow_residual_rl.report`.

Smoke results → `runs/smoke/smoke_report.json`. If check 3 (restore) fails, inspect it, then set
`--set restore_mode=replay` (prefix replay, cost charged to `bank_prefix_replay`) — do not loosen tolerances.

## Decisions taken from the actual sources (recorded in the resolved config / manifest)

| Item | Value | Source |
|---|---|---|
| episode cap | **720** (plan default 520 applied only if the recipe had none) | `gr00t/eval/rollout_policy.py` `DEFAULT_MAX_EPISODE_STEPS=720`, README `--max-episode-steps 720` |
| C | 8 | recipe `--n-action-steps 8` |
| settling | 0 steps | GR00T LIBERO path does none |
| obs / gripper convention | images flipped `[::-1, ::-1]`, state xyz + axis-angle + gripper qpos; gripper `-sign(2g-1)` | `gr00t/eval/sim/LIBERO/libero_env.py` |
| arm bound | clip to Box[-1,1] (= robosuite OSC input clip) | — |
| context hook | `model.backbone` output `backbone_features` [B,L,D], masked mean over `backbone_attention_mask` → fixed Gaussian projection to 128 | `gr00t_n1d7.py` |
| base noise | `torch.randn` inside the action head → seeded per call via `torch.random.fork_rng` | `Gr00tN1d7ActionHead.get_action_with_features` |
| model dtype on V100 | `auto` → float32 (no native bf16); identical for all arms | `Gr00tPolicy` hard-codes bf16 |
| init states | LIBERO benchmark init states (50/task) → 30/10/10 train/val/eval, `split_seed=61` | official GR00T eval uses `env.reset(seed)` instead |
| restore | MuJoCo `mjSTATE_INTEGRATION` + python-side robot/controller/gripper state (`gripper.current_action` is not in the MuJoCo state) | verified by smoke check 3 |
| bank train/val | pool episodes from train init states → bank train; every 4th from validation init states → bank val | split by root episode |

Budget note: with cap 720 the worst-case evaluation bound in plan sec. 14 scales by 720/520.

## Layout
```
configs/pilot.yaml                     single config
src/flow_residual_rl/
  basis.py            bases + fixed u→Δ map (ResidualMap)
  outcome_predictor.py  QR basis + MLP predictor, BCE + pairwise loss
  providers/gr00t.py  frozen GR00T provider (propose → a0_env, context, proprio)
  envs/libero_task.py LIBERO task, obs, snapshot/restore/replay
  envs/chunk_env.py   Gymnasium chunk env (one SAC step = one chunk)
  bank.py             pool, anchors, paired branches
  train_sac.py        SB3 SAC (zero-mean init, equal-RMS warmup, block resume)
  evaluate.py         paired held-out evaluation
  stages.py           bank / basis / eval / cell logic shared by scripts
  budget.py           env-step accounting (cost.json, hard limits)
  report.py           tables, curves, cost, Stage-A decision
scripts/{smoke,eval_base,collect_bank,train_basis,run_pilot}.py
docker/               V100 image (cu126 torch, no flash-attn)
runs/<suite>_task<id>/{splits.json, bank/, basis/, base_eval/, cells/<arm>_seed<s>/}
```
