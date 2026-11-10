"""Aggregate runs -> tables, learning curves, cost report, Stage-A decision.

    python -m flow_residual_rl.report --runs runs/ --output reports/pilot/

Decision rule (fixed before running, plan sec. 14), at the Stage-A endpoint:
  partial     unless EVERY required arm (stage_a_arms) has a committed eval at the endpoint with the
              required number of episodes on the identical eval schedule; missing cells are listed.
              Stage B is only launched on `continue`.
  best_fixed = best of {random12, dct12, pca12 if present} by eval SR
  continue    if SR(outcome12) - SR(best_fixed) >= +5 pp  OR  dAUC(train SR curve) >= 0.03
  no_benefit  if SR(outcome12) - SR(best_fixed) <= -5 pp  AND dAUC <= 0
  setup_failure if every SAC arm is >= 10 pp below the base VLA (general RL/setup failure, not evidence about the space)
  uncertain   otherwise (one extra seed on the same task is allowed)
These are compute-allocation rules, not statistical claims.
`train_sr_auc` is the mean success of the stochastic TRAINING episodes over 5k-step bins: a
diagnostic of learning speed, not held-out performance.

Cost convention: `online_steps` per cell; the task's bank (pool + branches, incl. aborted attempts)
is a shared preparation cost: the normaliser of EVERY arm is fit on it and outcome12/pca12 bases
are learned from it. It is shown per task once (`bank_steps_task`) and counted once in the project
total; `steps_incl_bank` = online + bank for every arm.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FIXED = ("random12", "dct12", "pca12")
DEFAULT_REQUIRED = ("outcome12", "random12", "dct12", "full48")


def _jsonl(p: Path):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def load_cells(runs: Path):
    cells = []
    for tdir in sorted(runs.glob("*_task*")):
        tid = int(tdir.name.split("task")[-1])
        bank_meta = json.loads((tdir / "bank" / "bank_metadata.json").read_text()) if (tdir / "bank" / "bank_metadata.json").exists() else {}
        bank_steps = bank_meta.get("cost", {}).get("bank_total", 0)
        base = _jsonl(tdir / "base_eval" / "episodes.jsonl")
        for cdir in sorted((tdir / "cells").glob("*_seed*")):
            arm, seed = cdir.name.rsplit("_seed", 1)
            st = json.loads((cdir / "status.json").read_text()) if (cdir / "status.json").exists() else {}
            evals = {}
            for f in sorted((cdir / "evals").glob("eval_*.json")):
                d = json.loads(f.read_text())
                evals[int(d["summary"]["online_steps"])] = d
            cost = json.loads((cdir / "cost.json").read_text()) if (cdir / "cost.json").exists() else {}
            cells.append({"task_id": tid, "arm": arm, "seed": int(seed), "status": st, "evals": evals, "cost": cost,
                          "train_eps": _jsonl(cdir / "train_episodes.jsonl"), "bank_steps": bank_steps,
                          "base_eval": base, "bank_meta": bank_meta})
    return cells


def train_curve(eps, bin_steps=5000, max_steps=None):
    if not eps:
        return np.array([]), np.array([])
    s = np.array([e["online_steps_end"] for e in eps])
    y = np.array([e["success"] for e in eps], float)
    hi = max_steps or s.max()
    edges = np.arange(0, hi + bin_steps, bin_steps)
    xs, ys = [], []
    for lo, up in zip(edges[:-1], edges[1:]):
        m = (s > lo) & (s <= up)
        if m.any():
            xs.append(up)
            ys.append(y[m].mean())
    return np.array(xs), np.array(ys)


def auc(eps, max_steps, bin_steps=5000):
    x, y = train_curve(eps, bin_steps, max_steps)
    y = y[x <= max_steps]
    return float(y.mean()) if y.size else float("nan")


def paired(base, arm_eps):
    b = {r["episode"]: r for r in base}
    rescued = harmed = n = 0
    for r in arm_eps:
        if r["episode"] in b:
            n += 1
            rescued += int(r["success"] and not b[r["episode"]]["success"])
            harmed += int(b[r["episode"]]["success"] and not r["success"])
    return {"n_paired": n, "rescued": rescued, "harmed": harmed}


def bootstrap_diff(a_eps, b_eps, n_boot=2000, seed=0):
    """Paired diff SR(a)-SR(b) over common episodes, resampling init states (repeats grouped)."""
    bi = {r["episode"]: r for r in b_eps}
    rows = [(r["init_state_id"], float(r["success"]) - float(bi[r["episode"]]["success"])) for r in a_eps if r["episode"] in bi]
    if not rows:
        return {"diff": float("nan"), "ci95": [float("nan")] * 2}
    states = sorted({s for s, _ in rows})
    by = {s: [d for t, d in rows if t == s] for s in states}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(states, size=len(states), replace=True)
        v = [d for s in pick for d in by[s]]
        boots.append(np.mean(v))
    return {"diff": float(np.mean([d for _, d in rows])), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "n_states": len(states), "n_episodes": len(rows)}


def build_table(cells):
    rows = []
    for c in cells:
        for at, d in sorted(c["evals"].items()):
            s = d["summary"]
            rows.append({"task": c["task_id"], "seed": c["seed"], "arm": c["arm"], "online_steps": at,
                         "policy_actual_online_steps": s.get("policy_actual_online_steps"),
                         "bank_steps_task": c["bank_steps"], "steps_incl_bank": at + c["bank_steps"],
                         "SR": s["success_rate"], "n_eval": s["n_episodes"], "n_states": s["n_unique_states"],
                         "rms_median": s["rms_median"], "clip_mean": s["clip_mean"],
                         "trainable_params": c["status"].get("trainable_params"),
                         "train_wall_h": sum(b["wall_s"] for b in c["status"].get("blocks", [])) / 3600,
                         **paired(c["base_eval"], d["episodes"])})
    return rows


def _endpoint_status(cells, task_id, seed, at, required, n_required):
    sel, missing = {}, []
    for arm in required:
        c = next((c for c in cells if c["task_id"] == task_id and c["seed"] == seed and c["arm"] == arm), None)
        if c is None:
            missing.append(f"{arm}: no cell")
        elif at not in c["evals"]:
            missing.append(f"{arm}: no eval at {at} (online={c['status'].get('online_steps', 0)})")
        elif n_required and c["evals"][at]["summary"]["n_episodes"] < n_required:
            missing.append(f"{arm}: eval at {at} has {c['evals'][at]['summary']['n_episodes']} < {n_required} episodes")
        else:
            sel[arm] = c
    keys = {a: tuple(r["key"] for r in c["evals"][at]["episodes"]) for a, c in sel.items()}
    if len(set(keys.values())) > 1:
        missing.append(f"eval schedules differ across arms at {at}: {sorted(keys)}")
    return sel, missing


def compare_at(sel, at):
    """Comparison of all available arms at one common eval point (diagnostic, any block)."""
    sr = {a: c["evals"][at]["summary"]["success_rate"] for a, c in sel.items()}
    fixed = [a for a in FIXED if a in sr]
    out = {"at_online_steps": at, "SR": sr, "train_sr_auc": {a: auc(c["train_eps"], at) for a, c in sel.items()}}
    if "outcome12" in sr and fixed:
        best = max(fixed, key=lambda a: sr[a])
        out.update(best_fixed=best, dSR_vs_best_fixed=sr["outcome12"] - sr[best],
                   dAUC_vs_best_fixed=out["train_sr_auc"]["outcome12"] - out["train_sr_auc"][best],
                   paired_bootstrap_vs_best_fixed=bootstrap_diff(sel["outcome12"]["evals"][at]["episodes"],
                                                                 sel[best]["evals"][at]["episodes"]))
    if "outcome12" in sr and "full48" in sr:
        out["vs_full48"] = sr["outcome12"] - sr["full48"]
    return out


def decide(cells, task_id, seed, at, required=DEFAULT_REQUIRED, n_required=None):
    required = tuple(required)
    sel, missing = _endpoint_status(cells, task_id, seed, at, required, n_required)
    # comparisons at every eval point all required arms share (each completed equal block)
    points = sorted({k for c in cells if c["task_id"] == task_id and c["seed"] == seed for k in c["evals"]})
    per_block = []
    for p_ in points:
        s_, m_ = _endpoint_status(cells, task_id, seed, p_, required, None)
        per_block.append(dict(compare_at(s_, p_), complete=not m_, missing=m_))
    base = {"at_online_steps": at, "task_id": task_id, "seed": seed, "required_arms": list(required),
            "per_eval_point": per_block}
    if missing:
        return dict(base, decision="partial", missing=missing,
                    reason="not every required arm has a committed eval at the endpoint; no Stage-B decision")
    cmp = compare_at(sel, at)
    sr, d_sr, d_auc = cmp["SR"], cmp["dSR_vs_best_fixed"], cmp["dAUC_vs_best_fixed"]
    n_ep = sel["outcome12"]["evals"][at]["summary"]["n_episodes"]
    base_eps = {r["episode"]: r for r in sel["outcome12"]["base_eval"]}
    eval_eps = sel["outcome12"]["evals"][at]["episodes"]
    base_sr = (float(np.mean([base_eps[r["episode"]]["success"] for r in eval_eps]))
               if eval_eps and all(r["episode"] in base_eps for r in eval_eps) else float("nan"))
    if np.isfinite(base_sr) and all(v <= base_sr - 0.10 for v in sr.values()):
        dec = "setup_failure"
    elif d_sr >= 0.05 or d_auc >= 0.03:
        dec = "continue"
    elif d_sr <= -0.05 and d_auc <= 0:
        dec = "no_benefit"
    else:
        dec = "uncertain"
    return dict(base, **cmp, decision=dec, base_SR=base_sr, n_eval_episodes=n_ep)


def plots(cells, out: Path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print("matplotlib unavailable:", e)
        return
    colors = {"outcome12": "#d62728", "random12": "#7f7f7f", "dct12": "#1f77b4", "pca12": "#2ca02c", "full48": "#9467bd"}
    for tid in sorted({c["task_id"] for c in cells}):
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for c in [c for c in cells if c["task_id"] == tid]:
            ls = "-" if c["seed"] == min(x["seed"] for x in cells) else "--"
            x, y = train_curve(c["train_eps"])
            if x.size:
                ax[0].plot(x, y, ls, color=colors.get(c["arm"]), label=f"{c['arm']} s{c['seed']}")
            ev = sorted(c["evals"].items())
            if ev:
                ax[1].plot([k for k, _ in ev], [d["summary"]["success_rate"] for _, d in ev], ls + "o",
                           color=colors.get(c["arm"]), label=f"{c['arm']} s{c['seed']}")
        base = next((c["base_eval"] for c in cells if c["task_id"] == tid), [])
        if base:
            ax[1].axhline(np.mean([r["success"] for r in base]), color="k", lw=1, ls=":", label="base VLA")
        ax[0].set(title=f"task {tid}: online train-episode SR (5k bins)", xlabel="online env steps", ylabel="SR")
        ax[1].set(title=f"task {tid}: held-out eval SR", xlabel="online env steps", ylabel="SR")
        for a in ax:
            a.grid(alpha=.3)
            a.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / f"curves_task{tid}.png", dpi=130)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--output", default="reports/pilot")
    p.add_argument("--stage-a-endpoint", type=int, default=60000)
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--seed", type=int, default=61)
    p.add_argument("--config", default="configs/pilot.yaml",
                   help="read stage_a_arms / stage_a_eval_episodes (required arms and episode count)")
    a = p.parse_args()
    required, n_required = DEFAULT_REQUIRED, None
    if a.config and Path(a.config).exists():
        import yaml

        cfg = yaml.safe_load(Path(a.config).read_text())
        required = tuple(cfg["stage_a_arms"])
        n_required = {int(k): v for k, v in cfg["stage_a_eval_episodes"].items()}.get(a.stage_a_endpoint)
    runs, out = Path(a.runs), Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    cells = load_cells(runs)
    rows = build_table(cells)
    (out / "table.json").write_text(json.dumps(rows, indent=1))
    if rows:
        keys = list(rows[0].keys())
        md = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
        md += ["| " + " | ".join(f"{r[k]:.3f}" if isinstance(r[k], float) else str(r[k]) for k in keys) + " |" for r in rows]
        (out / "table.md").write_text("\n".join(md) + "\n")
    cost = {"cells": {f"t{c['task_id']}_{c['arm']}_s{c['seed']}": c["cost"] for c in cells}}
    for tdir in sorted(runs.glob("*_task*")):
        for sub in ("bank", "base_eval"):
            f = tdir / sub / "cost.json"
            if f.exists():
                cost[f"{tdir.name}/{sub}"] = json.loads(f.read_text())
    tot = {}
    for v in cost.values():
        for vv in (v.values() if isinstance(v, dict) and "steps" not in v else [v]):
            for k, n in vv.get("steps", {}).items():
                tot[k] = tot.get(k, 0) + n
    cost["project_total_steps_by_bucket"] = tot
    cost["project_total_steps"] = int(sum(tot.values()))
    cost["convention"] = ("bank counted once per task (incl. aborted_bank); online/eval per cell; "
                          "aborted_* = steps of interrupted work, kept in the totals")
    (out / "cost_report.json").write_text(json.dumps(cost, indent=1))
    plots(cells, out)
    dec = decide(cells, a.task_id, a.seed, a.stage_a_endpoint, required, n_required)
    (out / "stageA_decision.json").write_text(json.dumps(dec, indent=1))
    Path("reports").mkdir(exist_ok=True)
    (Path("reports") / "stageA_decision.json").write_text(json.dumps(dec, indent=1))
    print(json.dumps(dec, indent=1))
    print(f"table: {out/'table.md'}  curves: {out}/curves_task*.png  cost: {out/'cost_report.json'}")


if __name__ == "__main__":
    main()
