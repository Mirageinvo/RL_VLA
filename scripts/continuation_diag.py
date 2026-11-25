"""Fixed vs re-planned continuation after a corrected chunk (follow-up of jacobian_diag.py).

In the Jacobian validation, methods with a small eef drift at s=C (lag, global_gram, local_eef)
showed a LARGER drift after the GR00T continuation chunk than dct/integrator. Two candidate
causes: (a) physics / controller (the corrected execution ends at the same pose but with a
different velocity / joint state / contact), (b) GR00T re-plans differently from the perturbed
observation. This script separates them on the SAME held-out anchors and the SAME executed
corrected actions as the Jacobian validation:

  nominal:  a0 chunk, then GR00T continuation chunk c0 (seed as in the validation)
  fixed:    corrected chunk, then c0 replayed open loop         -> (a) only
  replanned: corrected chunk, then GR00T re-plans (the saved validation effects; one draw per
            anchor is re-run here as a reproduction check)      -> (a) + (b)

| drift grows under fixed       | physics: velocity, controller memory, contact      |
| grows mostly under replanned  | GR00T reaction                                      |
| grows under both              | both                                                |
| dct shrinks only if replanned | GR00T compensates the perturbation                  |

Snapshots are not stored by jacobian_diag.py, so the pool is re-collected with the same seeds
and the same anchors are selected; every anchor's GR00T proposal a0 must match the stored one
(--a0-tol), otherwise nothing is run (exit 4). The snapshots are then pickled for later
experiments. Also recorded: arm joint position / velocity differences at s=C.

  python scripts/continuation_diag.py --config configs/pilot.yaml --task-id 0
  python scripts/continuation_diag.py --task-id 0 --analyze-only
Output: runs/diagnostics/continuation_task<id>/ (exit 2: incomplete, 4: anchors not reproduced)
"""
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from flow_residual_rl import effects as fx
from flow_residual_rl.config import base_parser, derive_seed, load_config, read_json, write_json

PROTOCOLS = ("fixed", "replanned")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def measure(task):
    """Physical state + arm joint position / velocity."""
    p = fx.read_physical(task)
    e = task.env.env
    r, d = e.robots[0], e.sim.data
    p["qpos"] = np.array(d.qpos[r._ref_joint_pos_indexes], np.float64)
    p["qvel"] = np.array(d.qvel[r._ref_joint_vel_indexes], np.float64)
    return p


def load_source(src: Path) -> dict:
    anc = read_json(src / "anchors.json")
    vmeta = read_json(src / "validation_meta.json")
    if anc is None or vmeta is None or not vmeta.get("complete"):
        sys.exit(f"{src}: no complete Jacobian diagnostic")
    raw = np.load(src / "validation_raw.npz")
    recs = [json.loads(x) for x in (src / "validation.jsonl").read_text().splitlines() if x.strip()]
    ok = {(r["anchor"], r["method"], r["dir"], r["sign"]) for r in recs if "skipped" not in r and r.get("comparable")}
    return {"anc": anc, "vmeta": vmeta, "a0": np.load(src / "jacobians.npz")["a0"], "raw_e": raw["effects"],
            "raw_act": raw["actions"], "raw_key": raw["key"], "ok": ok}


# ----------------------------------------------------------------------------- collection


def collect(cfg, a, src, out):
    from jacobian_diag import collect_pool, select_anchors

    from flow_residual_rl.budget import StepCounter
    from flow_residual_rl.pipeline import make_provider, make_task, task_splits, write_manifest
    from flow_residual_rl.restore import restore_to

    S = load_source(src)
    anc, vmeta = S["anc"], S["vmeta"]
    sargs, C, methods_all = anc["args"], anc["C"], vmeta["methods"]
    counter = StepCounter(out / "cost.json")
    counter.set_limit(("diagnostic",), a.step_limit)
    prov = make_provider(cfg, counter)
    task = make_task(cfg, a.task_id, counter, prov)
    write_manifest(cfg, out / "run_manifest.json", prov, {"args": vars(a), "source": str(src)})

    def restore(t, s):
        restore_to(t, s, cfg["restore_mode"], "diagnostic")

    try:
        with counter.charge("diagnostic"):
            # ---- same pool, same anchors
            bounds = collect_pool(task, prov, task_splits(cfg, task)["train"], sargs["pool_episodes"], C)
            anchors = select_anchors(bounds, sargs["n_anchors"], sargs["seed"])
            old = anc["anchors"]
            ident = [(b["ep"], b["k"], b["split"]) for b in anchors] == [(b["ep"], b["k"], b["split"]) for b in old]
            da0 = [float(np.abs(b["a0"] - S["a0"][i]).max()) for i, b in enumerate(anchors)] if ident else []
            check = {"same_anchors": ident, "a0_max_diff": max(da0) if da0 else None, "a0_tol": a.a0_tol,
                     "per_anchor_a0_diff": da0}
            write_json(out / "reproduction.json", check)
            log(f"[repro] same anchors={ident} max |a0 - stored|={check['a0_max_diff']}")
            if not ident or max(da0) > a.a0_tol:
                log("[repro] FAILED: the pool is not reproduced (GR00T or simulator not deterministic); nothing run")
                return 4
            with open(out / "anchor_snaps.pkl", "wb") as f:
                pickle.dump([{"ep": b["ep"], "k": b["k"], "split": b["split"], "phase": b["phase"]["phase"],
                              "snap": b["snap"], "a0": b["a0"]} for b in anchors], f)

            # ---- branches
            mi_sel = [methods_all.index(m) for m in a.methods]
            E, QP, QV, KEY, rep = [], [], [], [], {}
            for i in vmeta["heldout_idx"]:
                b = anchors[i]
                cont_seed = derive_seed("jacdiag-cont", a.task_id, b["ep"], b["k"])
                nom, _ = fx.execute(task, b["snap"], b["a0"], restore, measure)
                c0 = prov.propose(task.policy_observation(), None, cont_seed).a0_env.copy()
                for x in c0:
                    task.step(x)
                    nom.append(measure(task))
                rows = [r for r, (ii, mi, d, s) in enumerate(S["raw_key"])
                        if ii == i and mi in mi_sel and (i, methods_all[mi], int(d), int(s)) in S["ok"]]
                for n_r, r in enumerate(rows):
                    act = S["raw_act"][r]
                    if n_r == 0:  # reproduction of the saved re-planned branch
                        st, _ = fx.execute(task, b["snap"], act, restore, measure)
                        p = prov.propose(task.policy_observation(), None, cont_seed)
                        for x in p.a0_env:
                            task.step(x)
                            st.append(measure(task))
                        e = np.stack([fx.effect(u, v) for u, v in zip(st, nom)])
                        rep[i] = float(np.abs(e - S["raw_e"][r]).max())
                    st, _ = fx.execute(task, b["snap"], np.concatenate([act, c0]), restore, measure)
                    E.append(np.stack([fx.effect(u, v) for u, v in zip(st, nom)]))
                    QP.append(st[C - 1]["qpos"] - nom[C - 1]["qpos"])
                    QV.append(st[C - 1]["qvel"] - nom[C - 1]["qvel"])
                    KEY.append(r)
                counter.save()
                np.savez_compressed(out / "continuation.npz", e_fixed=np.stack(E), dqpos_C=np.stack(QP), dqvel_C=np.stack(QV),
                                    raw_row=np.array(KEY, int))
                log(f"[cont] anchor {i} phase={b['phase']['phase']} branches={len(rows)} "
                    f"replanned repro max|de|={rep[i]:.1e} steps={counter.steps['diagnostic']}")
            write_json(out / "reproduction.json", dict(check, replanned_max_diff=rep, complete=True,
                                                       methods=list(a.methods)))
    finally:
        counter.save()
        task.close()
    return 0


# ----------------------------------------------------------------------------- analysis


def analyze(src: Path, out: Path) -> str:
    S = load_source(src)
    rp = read_json(out / "reproduction.json") or {}
    anc, vmeta = S["anc"], S["vmeta"]
    C, G, sc = anc["C"], anc["layout"]["groups"], np.array(vmeta["scales"])
    rows_eef = G["eef"]
    z = np.load(out / "continuation.npz")
    raw_row = z["raw_row"]
    key = S["raw_key"][raw_row]
    methods = rp.get("methods") or [vmeta["methods"][m] for m in sorted(set(key[:, 1].tolist()))]
    norm = lambda e: np.linalg.norm(e[:, :, rows_eef] / sc[rows_eef][None, None], axis=2)  # noqa: E731
    Z = {"fixed": norm(z["e_fixed"]), "replanned": norm(S["raw_e"][raw_row])}
    T = Z["fixed"].shape[1]
    qv = np.linalg.norm(z["dqvel_C"], axis=1)
    qp = np.linalg.norm(z["dqpos_C"], axis=1)
    phase = {i: b["phase"]["phase"] for i, b in enumerate(anc["anchors"])}
    ep = {i: b["ep"] for i, b in enumerate(anc["anchors"])}
    mname = [vmeta["methods"][int(m)] for m in key[:, 1]]
    where = {(int(k[0]), mname[r], int(k[2]), int(k[3])): r for r, k in enumerate(key)}

    def anchor_means(vals, m, ph):
        out = {}
        for i in sorted(set(key[:, 0].tolist())):
            if ph != "all" and phase[i] != ph:
                continue
            s = (key[:, 0] == i) & (np.array(mname) == m)
            if s.any():
                out[i] = vals[s].mean(axis=0)
        return out

    L = ["# Fixed vs re-planned continuation", "",
         f"complete: {rp.get('complete', False)}; anchors reproduced: {rp.get('same_anchors')}, "
         f"max |a0 - stored| = {rp.get('a0_max_diff')}; re-planned branch reproduction max |de| per anchor: "
         f"{ {k: float(f'{v:.1e}') for k, v in (rp.get('replanned_max_diff') or {}).items()} }",
         f"branches: {len(raw_row)}. Normalised eef drift, mean over an anchor's draws, then median over anchors.", ""]
    phases = ["all"] + [p for p in ("grasp", "contact", "near", "free", "unknown") if p in set(phase[int(i)] for i in key[:, 0])]
    for ph in phases:
        ids = {int(i) for i in key[:, 0] if ph == "all" or phase[int(i)] == ph}
        L += [f"## phase {ph}: {len(ids)} anchors / {len({ep[i] for i in ids})} episodes", "",
              "| method | drift s=C | fixed s=2C | replanned s=2C | norm dq at C (rad) | norm dqdot at C (rad/s) |",
              "|---|---|---|---|---|---|"]
        for m in methods:
            f = anchor_means(Z["fixed"], m, ph)
            if not f:
                continue
            r = anchor_means(Z["replanned"], m, ph)
            med = lambda d, s: float(np.median([v[s - 1] for v in d.values()]))  # noqa: E731
            L.append(f"| {m} | {med(f, C):.3f} | {med(f, T):.3f} | {med(r, T):.3f} | "
                     f"{np.median(list(anchor_means(qp, m, ph).values())):.2e} | "
                     f"{np.median(list(anchor_means(qv, m, ph).values())):.2e} |")
        L += ["", f"drift per step 1..{T}, fixed continuation (median over anchors; '|' after s=C):", "```"]
        for m in methods:
            f = anchor_means(Z["fixed"], m, ph)
            if f:
                md = np.median(list(f.values()), axis=0)
                L.append(f"{m:<12} " + " ".join(f"{x:5.2f}" + ("|" if t == C else "") for t, x in enumerate(md, 1)))
        L.append("```")
        if "local_eef" in methods:
            L += ["", "local_eef vs method at s=2C: median dlog (<0 = local smaller), local smaller on k/n anchors", "",
                  "| vs | fixed dlog | fixed k/n | replanned dlog | replanned k/n |", "|---|---|---|---|---|"]
            for m in methods:
                if m == "local_eef":
                    continue
                cells = []
                for pr in PROTOCOLS:
                    d = []
                    for i in sorted(ids):
                        common = sorted({(dd, ss) for (ii, mm, dd, ss) in where if ii == i and mm == "local_eef"}
                                        & {(dd, ss) for (ii, mm, dd, ss) in where if ii == i and mm == m})
                        if common:  # only draws usable for both methods
                            lo = np.mean([Z[pr][where[(i, "local_eef", dd, ss)], T - 1] for dd, ss in common])
                            mo = np.mean([Z[pr][where[(i, m, dd, ss)], T - 1] for dd, ss in common])
                            d.append(np.log(max(lo, 1e-12) / max(mo, 1e-12)))
                    cells += [f"{np.median(d):.2f}" if d else "-", f"{sum(x < 0 for x in d)}/{len(d)}"]
                L.append(f"| {m} | " + " | ".join(cells) + " |")
        L.append("")
    return "\n".join(L)


def main():
    p = base_parser(__doc__)
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--methods", nargs="*", default=["dct", "lag", "global_gram", "local_eef"])
    p.add_argument("--a0-tol", type=float, default=1e-4, help="max |a0 - stored a0| for the pool to count as reproduced")
    p.add_argument("--step-limit", type=int, default=12000)
    p.add_argument("--analyze-only", action="store_true")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    cfg = load_config(a.config, a.set)
    from flow_residual_rl.pipeline import runs_dir

    src = runs_dir(cfg) / "diagnostics" / f"jacobian_task{a.task_id}"
    out = runs_dir(cfg) / "diagnostics" / f"continuation_task{a.task_id}"
    out.mkdir(parents=True, exist_ok=True)
    if not a.analyze_only:
        if (out / "continuation.npz").exists() and not a.force:
            sys.exit(f"{out} already has results; use --analyze-only, or --force to recollect")
        from flow_residual_rl.walltime import WallLedger

        _ledger = WallLedger(runs_dir(cfg), "continuation_diag")
        for f in ("continuation.npz", "reproduction.json", "analysis.md"):
            (out / f).unlink(missing_ok=True)
        code = collect(cfg, a, src, out)
        if code:
            sys.exit(code)
    rp = read_json(out / "reproduction.json") or {}
    if not (out / "continuation.npz").exists():
        sys.exit(2)
    md = analyze(src, out)
    (out / "analysis.md").write_text(md)
    print(md)
    if not rp.get("complete"):
        sys.exit(2)


if __name__ == "__main__":
    main()
