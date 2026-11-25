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
and the same anchors are selected (ep, sid, k, elapsed, split must match). Reuse mode (default):
the nominal chunk is the STORED a0 (must match the new proposal within --a0-tol); the corrected
chunk of EVERY branch must reproduce the saved effects of steps 1..C, and one saved re-planned
branch per anchor must be reproduced over 1..2C (--repro-tol); any failure stops the run,
leaves it incomplete and exits 4. --fresh: no comparison with saved effects; both protocols are
re-run from the new snapshots with the same executed corrections (an independent paired
comparison). anchor_snaps.pkl is marked verified only after a complete reuse-mode run.
Also recorded: arm joint position / velocity differences at s=C. The report adds the per-anchor
paired d_i = log(replanned / fixed drift at s=2C).

  python scripts/continuation_diag.py --config configs/pilot.yaml --task-id 0
  python scripts/continuation_diag.py --task-id 0 --analyze-only
  python scripts/continuation_diag.py --config configs/pilot.yaml --task-id 0 --fresh --force
Output: runs/diagnostics/continuation_task<id>/ (exit 2: incomplete, 4: not reproduced)
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
    sargs, C, methods_all, dims = anc["args"], anc["C"], vmeta["methods"], list(anc["dims"])
    counter = StepCounter(out / "cost.json")
    counter.set_limit(("diagnostic",), a.step_limit)
    prov = make_provider(cfg, counter)
    task = make_task(cfg, a.task_id, counter, prov)
    write_manifest(cfg, out / "run_manifest.json", prov, {"args": vars(a), "source": str(src)})

    def restore(t, s):
        restore_to(t, s, cfg["restore_mode"], "diagnostic")

    def snaps(anchors, verified):
        with open(out / "anchor_snaps.pkl", "wb") as f:
            pickle.dump({"verified": verified, "mode": "fresh" if a.fresh else "reuse",
                         "anchors": [{"ep": b["ep"], "sid": b["sid"], "k": b["k"], "elapsed": b["elapsed"],
                                      "split": b["split"], "phase": b["phase"]["phase"], "snap": b["snap"], "a0": b["a0"]}
                                     for b in anchors]}, f)

    def branch(b, act, cont, cont_seed):
        """Chunk `act`, then `cont` (fixed) or a GR00T re-plan (cont=None). States after every step."""
        st, _ = fx.execute(task, b["snap"], act, restore, measure)
        for x in (cont if cont is not None else prov.propose(task.policy_observation(), None, cont_seed).a0_env):
            task.step(x)
            st.append(measure(task))
        return st

    def fail(check, why):
        log(f"[repro] FAILED: {why}")
        write_json(out / "reproduction.json", dict(check, complete=False, failed=why))
        return 4

    try:
        with counter.charge("diagnostic"):
            # ---- same pool, same anchors
            bounds = collect_pool(task, prov, task_splits(cfg, task)["train"], sargs["pool_episodes"], C)
            anchors = select_anchors(bounds, sargs["n_anchors"], sargs["seed"])
            ident_keys = ("ep", "sid", "k", "elapsed", "split")
            ident = [tuple(b[k] for k in ident_keys) for b in anchors] == [tuple(b[k] for k in ident_keys) for b in anc["anchors"]]
            da0 = [float(np.abs(b["a0"] - S["a0"][i]).max()) for i, b in enumerate(anchors)] if ident else []
            check = {"mode": "fresh" if a.fresh else "reuse", "same_anchors": ident,
                     "a0_max_diff": max(da0) if da0 else None, "a0_tol": a.a0_tol, "repro_tol": a.repro_tol,
                     "per_anchor_a0_diff": da0, "methods": list(a.methods)}
            write_json(out / "reproduction.json", dict(check, complete=False))
            log(f"[repro] mode={check['mode']} same anchors (ep, sid, k, elapsed, split)={ident} "
                f"max |a0 - stored|={check['a0_max_diff']}")
            if not ident:
                return fail(check, "different anchors (pool not reproduced)")
            if not a.fresh and max(da0) > a.a0_tol:
                return fail(check, f"GR00T proposals differ from the stored ones by {max(da0):.1e} > {a.a0_tol}; "
                                   "use --fresh for a self-contained comparison")
            snaps(anchors, verified=False)

            # ---- branches
            mi_sel = [methods_all.index(m) for m in a.methods]
            E, ER, QP, QV, KEY, rep, fixed_dev = [], [], [], [], [], {}, {}
            for i in vmeta["heldout_idx"]:
                b = anchors[i]
                cont_seed = derive_seed("jacdiag-cont", a.task_id, b["ep"], b["k"])
                a0 = b["a0"] if a.fresh else S["a0"][i]  # reuse: the stored proposal, exactly as in the validation
                nom, _ = fx.execute(task, b["snap"], a0, restore, measure)
                c0 = prov.propose(task.policy_observation(), None, cont_seed).a0_env.copy()  # nominal continuation
                for x in c0:
                    task.step(x)
                    nom.append(measure(task))
                rows = [r for r, (ii, mi, d, s) in enumerate(S["raw_key"])
                        if ii == i and mi in mi_sel and (i, methods_all[mi], int(d), int(s)) in S["ok"]]
                for n_r, r in enumerate(rows):
                    act = S["raw_act"][r]
                    if a.fresh:  # same executed correction, on top of the new proposal
                        act = a0.copy()
                        act[:, dims] = np.clip(a0[:, dims] + S["raw_act"][r][:, dims] - S["a0"][i][:, dims],
                                               task.ACTION_LOW[dims], task.ACTION_HIGH[dims])
                        ER.append(np.stack([fx.effect(u, v) for u, v in zip(branch(b, act, None, cont_seed), nom)]))
                    elif n_r == 0:  # the saved re-planned branch must be reproduced exactly
                        e = np.stack([fx.effect(u, v) for u, v in zip(branch(b, act, None, cont_seed), nom)])
                        rep[i] = float(np.abs(e - S["raw_e"][r]).max())
                        if rep[i] > a.repro_tol:
                            return fail(dict(check, replanned_max_diff=rep),
                                        f"anchor {i}: re-planned branch differs from the saved one by {rep[i]:.1e}; "
                                        "use --fresh for a self-contained comparison")
                    st = branch(b, act, c0, cont_seed)
                    e = np.stack([fx.effect(u, v) for u, v in zip(st, nom)])
                    if not a.fresh:  # corrected chunk itself: identical to the saved validation branch
                        fixed_dev[i] = max(fixed_dev.get(i, 0.0), float(np.abs(e[:C] - S["raw_e"][r][:C]).max()))
                        if fixed_dev[i] > a.repro_tol:
                            return fail(dict(check, replanned_max_diff=rep, fixed_first_C_max_diff=fixed_dev),
                                        f"anchor {i}: corrected chunk differs from the saved one by {fixed_dev[i]:.1e}")
                    E.append(e)
                    QP.append(st[C - 1]["qpos"] - nom[C - 1]["qpos"])
                    QV.append(st[C - 1]["qvel"] - nom[C - 1]["qvel"])
                    KEY.append(r)
                counter.save()
                extra = {"e_replanned": np.stack(ER)} if a.fresh else {}
                np.savez_compressed(out / "continuation.npz", e_fixed=np.stack(E), dqpos_C=np.stack(QP),
                                    dqvel_C=np.stack(QV), raw_row=np.array(KEY, int), **extra)
                log(f"[cont] anchor {i} phase={b['phase']['phase']} branches={len(rows)} "
                    + ("" if a.fresh else f"repro replanned={rep[i]:.1e} fixed[:C]={fixed_dev.get(i, 0.0):.1e} ")
                    + f"steps={counter.steps['diagnostic']}")
            write_json(out / "reproduction.json", dict(check, replanned_max_diff=rep, fixed_first_C_max_diff=fixed_dev,
                                                       complete=True))
            snaps(anchors, verified=not a.fresh)
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
    # reuse mode: re-planned = the saved validation effects (verified on one branch per anchor);
    # fresh mode: both protocols re-run from the same new snapshots
    Z = {"fixed": norm(z["e_fixed"]), "replanned": norm(z["e_replanned"] if "e_replanned" in z else S["raw_e"][raw_row])}
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
         f"mode: {rp.get('mode')}; complete: {rp.get('complete', False)}"
         + (f"; FAILED: {rp['failed']}" if rp.get("failed") else "")
         + f"; anchors reproduced (ep, sid, k, elapsed, split): {rp.get('same_anchors')}, "
         f"max |a0 - stored| = {rp.get('a0_max_diff')}; tolerance {rp.get('repro_tol')}",
         f"reproduction max |de| per anchor: re-planned branch "
         f"{ {k: float(f'{v:.1e}') for k, v in (rp.get('replanned_max_diff') or {}).items()} }, "
         f"corrected chunk (all branches) { {k: float(f'{v:.1e}') for k, v in (rp.get('fixed_first_C_max_diff') or {}).items()} }",
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
        L += ["", "re-planned vs fixed, per anchor d_i = log(mean drift replanned / mean drift fixed) at s=2C over the same",
              "branches (<0: GR00T re-planning reduces the deviation, >0: increases it)", "",
              "| method | anchors | median d | d<0 | d>0 | min d | max d |", "|---|---|---|---|---|---|---|"]
        for m in methods:
            f, r = anchor_means(Z["fixed"], m, ph), anchor_means(Z["replanned"], m, ph)
            d = [float(np.log(max(r[i][T - 1], 1e-12) / max(f[i][T - 1], 1e-12))) for i in f if i in r]
            if d:
                L.append(f"| {m} | {len(d)} | {np.median(d):.2f} | {sum(x < 0 for x in d)} | {sum(x > 0 for x in d)} | "
                         f"{min(d):.2f} | {max(d):.2f} |")
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
    p.add_argument("--repro-tol", type=float, default=1e-5,
                   help="max |effect - saved effect| (m / rad) of re-executed validation branches (reuse mode)")
    p.add_argument("--fresh", action="store_true",
                   help="do not reuse the saved re-planned effects: run both protocols from the new snapshots "
                        "(same executed corrections added to the new proposals); for when reproduction fails")
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
