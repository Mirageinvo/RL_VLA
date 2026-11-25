"""Physical-effect Jacobian diagnostic for one task (post-Stage-A; no bank, no RL).

1. Pool: a few base-GR00T episodes from TRAIN init states; a snapshot, the proposal and an
   interaction-phase label (grasp / contact / near / free) at every chunk boundary.
2. Anchors: n_anchors boundaries stratified by phase; episodes alternate FIT / HELD-OUT, so
   no held-out anchor shares an episode with a fit anchor.
3. Per anchor: the nominal chunk twice (restore repeatability) and a central-difference
   Jacobian of the physical effect after s = 1..C steps w.r.t. each of the C*n_dims command
   coordinates (+-eta on ONE coordinate of ONE step). On the first few fit anchors again with
   eta/2 (linearity / step-size check).
4. Models on FIT anchors: integrator, fixed controller lag, global mean; OER bases (end /
   shape) from DCT, integrator, lag, global and the anchor's own Jacobian (local oracle).
5. Validation on HELD-OUT anchors, real simulator: random directions inside each method's
   SHAPE subspace at equal planned RMS, both signs; the corrected chunk plus one more GR00T
   chunk (same noise seed as the nominal continuation). Endpoint drift vs intermediate change.
6. analysis.json / analysis.md (re-run offline with --analyze-only).

Every env.step is charged to the `diagnostic` bucket (hard limit --step-limit) and the run is
in the project wall-time ledger. Output: runs/diagnostics/jacobian_task<id>/
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from flow_residual_rl import effects as fx
from flow_residual_rl.config import base_parser, derive_seed, load_config, read_json, write_json

PHASES = ("grasp", "contact", "near", "free")
SHAPE_STEPS = (2, 4, 6)
METHODS = ("dct", "integrator", "lag", "global", "local")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------- collection


def collect_pool(task, prov, ids, n_ep, C):
    bounds = []
    for ep in range(n_ep):
        sid = int(ids[ep % len(ids)])
        task.reset(sid, settle_bucket="diagnostic")
        k = 0
        while task.horizon_left >= C:
            snap = task.snapshot()
            phys = fx.read_physical(task)
            prop = prov.propose(task.policy_observation(), None, derive_seed("jacdiag-pool", task.task_id, ep, k))
            if task.horizon_left >= 2 * C:  # room for the corrected chunk + one continuation chunk
                bounds.append({"ep": ep, "sid": sid, "k": k, "elapsed": int(task.elapsed), "snap": snap,
                               "a0": prop.a0_env.copy(), "phase": fx.phase_label(task, phys)})
            done = False
            for t in range(C):
                _, s = task.step(prop.a0_env[t])
                if s:
                    done = True
                    break
            k += 1
            if done:
                break
        log(f"[pool] ep={ep} state={sid} len={task.elapsed} success={task.success} "
            f"phases={ {p: sum(b['phase']['phase'] == p for b in bounds if b['ep'] == ep) for p in PHASES} }")
    return bounds


def select_anchors(bounds, n, seed):
    rng = np.random.default_rng(seed)
    out = []
    for split, want in (("fit", n - n // 2), ("heldout", n // 2)):
        cand = [b for b in bounds if (b["ep"] % 2 == 0) == (split == "fit")]
        by = {p: [cand[i] for i in rng.permutation(len(cand)) if cand[i]["phase"]["phase"] == p] for p in PHASES}
        got = []
        while len(got) < want and any(by.values()):
            for p in PHASES:
                if by[p] and len(got) < want:
                    got.append(by[p].pop())
        for b in got:
            b["split"] = split
        out += got
    return out


def run_chunk(task, b, actions, restore, cont_seed=None, prov=None):
    """Corrected chunk (+ optionally one GR00T continuation chunk); physical state after each step."""
    states = fx.execute(task, b["snap"], actions, restore, fx.read_physical)
    if cont_seed is not None:
        prop = prov.propose(task.policy_observation(), None, cont_seed)
        for a in prop.a0_env:
            task.step(a)
            states.append(fx.read_physical(task))
    return states


def collect(cfg, a, out):
    from flow_residual_rl.budget import StepCounter
    from flow_residual_rl.pipeline import make_provider, make_task, task_splits, write_manifest
    from flow_residual_rl.restore import restore_to

    counter = StepCounter(out / "cost.json")
    counter.set_limit(("diagnostic",), a.step_limit)
    prov = make_provider(cfg, counter)
    task = make_task(cfg, a.task_id, counter, prov)
    sp = task_splits(cfg, task)
    write_manifest(cfg, out / "run_manifest.json", prov, {"args": vars(a)})
    C, dims = cfg["execution_horizon"], list(cfg["corrected_dims"])
    nd, N = len(dims), cfg["execution_horizon"] * len(cfg["corrected_dims"])

    def restore(t, s):
        restore_to(t, s, cfg["restore_mode"], "diagnostic")

    with counter.charge("diagnostic"):
        bounds = collect_pool(task, prov, sp["train"], a.pool_episodes, C)
        anchors = select_anchors(bounds, a.n_anchors, a.seed)
        log(f"[anchors] {len(anchors)} of {len(bounds)} boundaries: "
            + ", ".join(f"{s}:{[b['phase']['phase'] for b in anchors if b['split'] == s]}" for s in ("fit", "heldout")))
        p0 = fx.read_physical(task)
        layout = fx.feature_layout(len(p0["grip_qpos"]), p0["obj_names"])
        Js, Jh, half_idx = [], [], []
        n_half = 0
        for i, b in enumerate(anchors):
            t0 = time.time()
            n1 = fx.execute(task, b["snap"], b["a0"], restore, fx.read_physical)
            n2 = fx.execute(task, b["snap"], b["a0"], restore, fx.read_physical)
            b["repeat_max"] = float(max(np.abs(fx.effect(x, y)).max() for x, y in zip(n1, n2)))
            J, info = fx.fd_jacobian(task, b["snap"], b["a0"], dims, a.eta, restore, fx.read_physical,
                                     task.ACTION_LOW, task.ACTION_HIGH)
            b["n_clipped_cols"] = info["n_clipped"]
            Js.append(J)
            if b["split"] == "fit" and n_half < a.eta_check_anchors:
                Jh.append(fx.fd_jacobian(task, b["snap"], b["a0"], dims, a.eta / 2, restore, fx.read_physical,
                                         task.ACTION_LOW, task.ACTION_HIGH)[0])
                half_idx.append(i)
                n_half += 1
            counter.save()
            log(f"[jac] anchor {i + 1}/{len(anchors)} {b['split']} phase={b['phase']['phase']} t={b['elapsed']} "
                f"repeat_max={b['repeat_max']:.1e} clipped_cols={b['n_clipped_cols']} "
                f"|J8 eef|={np.linalg.norm(np.nan_to_num(J[-1][layout['groups']['eef']])):.3g} "
                f"wall={time.time() - t0:.0f}s steps={counter.steps['diagnostic']}")
        np.savez_compressed(out / "jacobians.npz", J=np.stack(Js), J_half=np.stack(Jh) if Jh else np.zeros(0),
                            half_idx=np.array(half_idx, int), a0=np.stack([b["a0"] for b in anchors]))
        meta = [{k: v for k, v in b.items() if k not in ("snap", "a0")} for b in anchors]
        write_json(out / "anchors.json", {"anchors": meta, "layout": layout, "args": vars(a), "C": C, "dims": dims,
                                          "restore_mode": cfg["restore_mode"], "epsilon_rms": cfg["epsilon_rms"]})

        # ---- validation on held-out anchors (real interventions)
        fit = [J for J, b in zip(Js, anchors) if b["split"] == "fit" and b["repeat_max"] <= a.repeat_tol]
        sc = fx.group_scales(fit, layout)
        glob = model_bases(fit, sc, layout, C, nd)
        rows = layout["groups"]["eef"]
        rng = np.random.default_rng(a.seed + 1)
        vf = out / "validation.jsonl"
        vf.unlink(missing_ok=True)
        for i, (b, J) in enumerate(zip(anchors, Js)):
            if b["split"] != "heldout" or b["repeat_max"] > a.repeat_tol:
                continue
            cont = derive_seed("jacdiag-cont", a.task_id, b["ep"], b["k"])
            nominal = run_chunk(task, b, b["a0"], restore, cont, prov)
            bases = dict(glob, local=fx.oer_basis([fx.normalise(J, sc)], rows, C, SHAPE_STEPS))
            coefs = rng.standard_normal((a.val_dirs, bases["dct"]["shape"].shape[1]))
            coefs /= np.linalg.norm(coefs, axis=1, keepdims=True)
            for m in METHODS:
                for di, c in enumerate(coefs):
                    delta = (a.val_rms * np.sqrt(N) * (bases[m]["shape"] @ c)).reshape(C, nd)
                    for sign in (1, -1):
                        act = b["a0"].copy()
                        act[:, dims] = np.clip(b["a0"][:, dims] + sign * delta, task.ACTION_LOW[dims], task.ACTION_HIGH[dims])
                        st = run_chunk(task, b, act, restore, cont, prov)
                        e = np.stack([fx.effect(x, y) for x, y in zip(st, nominal)])
                        rec = {"anchor": i, "phase": b["phase"]["phase"], "method": m, "dir": di, "sign": sign,
                               "rms_actual": float(np.sqrt(np.mean((act[:, dims] - b["a0"][:, dims]) ** 2))),
                               **fx.intervention_metrics(e, rows, sc, SHAPE_STEPS, C),
                               "obj_drift_m_end": float(np.abs(e[C - 1][layout["groups"]["obj_pos"]]).max(initial=0)),
                               "obj_drift_m_cont": float(np.abs(e[-1][layout["groups"]["obj_pos"]]).max(initial=0)),
                               "eef_pos_drift_m_end": float(np.linalg.norm(e[C - 1][layout["groups"]["eef_pos"]]))}
                        with open(vf, "a") as f:
                            f.write(json.dumps(rec) + "\n")
            counter.save()
            log(f"[val] anchor {i} phase={b['phase']['phase']} done, steps={counter.steps['diagnostic']}")
    counter.save()
    task.close()


def model_bases(fit, sc, layout, C, nd):
    rows = layout["groups"]["eef"]
    fitn = [fx.normalise(J, sc) for J in fit]
    return {"dct": fx.dct_oer(C, nd),
            "integrator": fx.oer_basis([fx.integrator_J(fx.fit_integrator(fitn, nd), C)], rows, C, SHAPE_STEPS),
            "lag": fx.oer_basis([fx.lag_J(fx.fit_lag(fitn, nd))], rows, C, SHAPE_STEPS),
            "global": fx.oer_basis(fitn, rows, C, SHAPE_STEPS)}


# ----------------------------------------------------------------------------- analysis


def _med(x):
    x = [v for v in x if v is not None and np.isfinite(v)]
    return float(np.median(x)) if x else None


def analyze(out: Path, repeat_tol: float) -> dict:
    meta = read_json(out / "anchors.json")
    z = np.load(out / "jacobians.npz")
    J_all, layout, C = z["J"], meta["layout"], meta["C"]
    nd = len(meta["dims"])
    anchors = meta["anchors"]
    ok = [b["repeat_max"] <= repeat_tol for b in anchors]
    fit_i = [i for i, b in enumerate(anchors) if b["split"] == "fit" and ok[i]]
    ho_i = [i for i, b in enumerate(anchors) if b["split"] == "heldout" and ok[i]]
    fit = [J_all[i] for i in fit_i]
    sc = fx.group_scales(fit, layout)
    G = layout["groups"]
    res = {"n_anchors": len(anchors), "n_fit": len(fit_i), "n_heldout": len(ho_i),
           "excluded_restore": [i for i in range(len(anchors)) if not ok[i]],
           "repeat_max": max(b["repeat_max"] for b in anchors), "scales": {g: float(sc[G[g][0]]) for g in
                                                                            ("eef_pos", "eef_rot", "obj_pos", "obj_rot") if G[g]},
           "phases": {s: [anchors[i]["phase"]["phase"] for i in idx] for s, idx in (("fit", fit_i), ("heldout", ho_i))}}

    # 1. step-size / linearity: J(eta) vs J(eta/2) on the same anchors
    lin = {}
    if z["J_half"].size:
        for g in ("eef_pos", "eef_rot", "obj_pos"):
            if G[g]:
                lin[g] = {f"s{s}": _med([fx.rel_error(z["J_half"][k], J_all[i], G[g], s - 1)
                                         for k, i in enumerate(z["half_idx"])]) for s in (2, 4, 8)}
    res["eta_half_rel_diff"] = lin

    # 2. prediction error of state-independent models on held-out anchors
    fitn = [fx.normalise(J, sc) for J in fit]
    preds = {"integrator": fx.integrator_J(fx.fit_integrator(fitn, nd), C), "lag": fx.lag_J(fx.fit_lag(fitn, nd)),
             "global": np.nanmean(np.stack(fitn), axis=0)}
    err = {}
    for m, Jp in preds.items():
        err[m] = {g: {f"s{s}": _med([fx.rel_error(Jp, fx.normalise(J_all[i], sc), G[g], s - 1) for i in ho_i])
                      for s in (2, 4, 6, 8)} for g in ("eef_pos", "eef_rot", "obj_pos") if G[g]}
    res["heldout_rel_error"] = err

    # 3. subspaces: local bases of every anchor vs the state-independent ones
    rows = G["eef"]
    glob = model_bases(fit, sc, layout, C, nd)
    sub = []
    for i in fit_i + ho_i:
        loc = fx.oer_basis([fx.normalise(J_all[i], sc)], rows, C, SHAPE_STEPS)
        sv = np.linalg.svd(np.nan_to_num(fx.normalise(J_all[i], sc)[C - 1][rows]), compute_uv=False)
        d = {"anchor": i, "split": anchors[i]["split"], "phase": anchors[i]["phase"]["phase"],
             "sv_end_eef": [float(x) for x in sv],
             "obj_resp_end_m": float(np.linalg.norm(np.nan_to_num(J_all[i][C - 1][G["obj_pos"]]))) if G["obj_pos"] else 0.0,
             "eef_resp_end_m": float(np.linalg.norm(np.nan_to_num(J_all[i][C - 1][G["eef_pos"]])))}
        for m, B in glob.items():
            d[f"ov_end_{m}"] = fx.overlap(loc["end"], B["end"])
            d[f"ov_shape_{m}"] = fx.overlap(loc["shape"], B["shape"])
            d[f"max_angle_shape_{m}"] = max(fx.principal_angles_deg(loc["shape"], B["shape"]))
        sub.append(d)
    res["model_basis_overlap"] = {m: {"end_vs_dct": fx.overlap(B["end"], glob["dct"]["end"]),
                                      "shape_vs_dct": fx.overlap(B["shape"], glob["dct"]["shape"])} for m, B in glob.items()}
    res["subspaces_by_phase"] = {}
    for ph in PHASES:
        xs = [d for d in sub if d["phase"] == ph]
        if xs:
            res["subspaces_by_phase"][ph] = {
                "n": len(xs), **{k: _med([d[k] for d in xs]) for k in xs[0] if k.startswith(("ov_", "max_angle"))},
                "obj_resp_end_m": _med([d["obj_resp_end_m"] for d in xs]),
                "eef_resp_end_m": _med([d["eef_resp_end_m"] for d in xs]),
                "sv_end_eef": [float(v) for v in np.median(np.array([d["sv_end_eef"] for d in xs]), axis=0)]}
    res["per_anchor"] = sub

    # 4. validation interventions (held-out, real simulator)
    vf = out / "validation.jsonl"
    if vf.exists():
        recs = [json.loads(l) for l in open(vf)]
        per = {}
        for r in recs:
            r["ratio"] = r["shape_change"] / max(r["endpoint_drift"], 1e-9)
            per.setdefault((r["anchor"], r["method"]), []).append(r)
        keys = ("endpoint_drift", "shape_change", "ratio", "drift_after_continuation", "obj_drift_m_end",
                "obj_drift_m_cont", "eef_pos_drift_m_end", "rms_actual")
        anc = sorted({a for a, _ in per})
        tab = {m: {k: _med([float(np.mean([r[k] for r in per[(i, m)] if k in r])) for i in anc if (i, m) in per])
                   for k in keys} for m in METHODS}
        paired = {}
        for m in METHODS:
            if m == "local":
                continue
            diff = [np.log(np.mean([r["ratio"] for r in per[(i, "local")]])) - np.log(np.mean([r["ratio"] for r in per[(i, m)]]))
                    for i in anc if (i, "local") in per and (i, m) in per]
            paired[m] = {"n_anchors": len(diff), "local_better": int(sum(d > 0 for d in diff)),
                         "median_log_ratio_gain": _med(diff)}
        res["validation"] = {"n_records": len(recs), "n_anchors": len(anc), "median_over_anchors": tab,
                             "local_vs": paired}
    write_json(out / "analysis.json", res)
    (out / "analysis.md").write_text(render_md(res))
    return res


def _f(x, p=3):
    return "—" if x is None else f"{x:.{p}g}"


def render_md(r: dict) -> str:
    L = [f"# Jacobian diagnostic\n",
         f"anchors: {r['n_anchors']} (fit {r['n_fit']}, held-out {r['n_heldout']}; excluded by restore: {r['excluded_restore']}); "
         f"max nominal repeat diff {r['repeat_max']:.1e}\n",
         f"phases: {r['phases']}\n", f"group scales (RMS response at s=C per unit command): {r['scales']}\n",
         "\n## 1. Step size: rel. diff J(eta) vs J(eta/2) (median; small = linear regime)\n"]
    for g, v in r["eta_half_rel_diff"].items():
        L.append(f"- {g}: " + ", ".join(f"{k}={_f(x)}" for k, x in v.items()))
    L.append("\n## 2. Held-out relative error of state-independent models (median over held-out anchors)\n")
    L.append("| model | group | s=2 | s=4 | s=6 | s=8 |\n|---|---|---|---|---|---|")
    for m, gs in r["heldout_rel_error"].items():
        for g, v in gs.items():
            L.append(f"| {m} | {g} | " + " | ".join(_f(v[f's{s}']) for s in (2, 4, 6, 8)) + " |")
    L.append("\nReference: the eta/2 difference above is the measurement noise floor of a single J.\n")
    L.append("## 3. Subspaces (eef features)\n")
    L.append("state-independent bases vs DCT: " + ", ".join(
        f"{m}: end {_f(v['end_vs_dct'])} / shape {_f(v['shape_vs_dct'])}" for m, v in r["model_basis_overlap"].items()))
    L.append("\nLocal (per-anchor) bases vs state-independent ones, median by phase "
             "(overlap 1 = same subspace; random 6-of-48 ~ 0.125):\n")
    L.append("| phase | n | shape vs dct | shape vs integrator | shape vs lag | shape vs global | end vs global | max angle shape vs global (deg) | |J_C eef_pos| m | |J_C obj_pos| m |\n|---|---|---|---|---|---|---|---|---|---|")
    for ph, v in r["subspaces_by_phase"].items():
        L.append(f"| {ph} | {v['n']} | {_f(v.get('ov_shape_dct'))} | {_f(v.get('ov_shape_integrator'))} | {_f(v.get('ov_shape_lag'))} | "
                 f"{_f(v.get('ov_shape_global'))} | {_f(v.get('ov_end_global'))} | {_f(v.get('max_angle_shape_global'))} | "
                 f"{_f(v['eef_resp_end_m'])} | {_f(v['obj_resp_end_m'])} |")
    if "validation" in r:
        v = r["validation"]
        L.append(f"\n## 4. Validation on held-out anchors ({v['n_anchors']} anchors, {v['n_records']} real interventions)\n")
        L.append("Random directions in each method's SHAPE subspace at equal planned RMS. Normalised eef units. "
                 "Wanted: large shape_change with small endpoint_drift.\n")
        L.append("| method | endpoint_drift | shape_change | shape/endpoint | drift after +1 GR00T chunk | obj drift end (m) | obj drift +1 chunk (m) | rms |\n|---|---|---|---|---|---|---|---|")
        for m, t in v["median_over_anchors"].items():
            L.append(f"| {m} | {_f(t['endpoint_drift'])} | {_f(t['shape_change'])} | {_f(t['ratio'])} | "
                     f"{_f(t['drift_after_continuation'])} | {_f(t['obj_drift_m_end'])} | {_f(t['obj_drift_m_cont'])} | {_f(t['rms_actual'])} |")
        L.append("\nLocal oracle vs each method (per anchor, log shape/endpoint ratio): "
                 + "; ".join(f"{m}: better on {p['local_better']}/{p['n_anchors']}, median gain {_f(p['median_log_ratio_gain'])}"
                             for m, p in v["local_vs"].items()))
    return "\n".join(L) + "\n"


def main():
    p = base_parser(__doc__)
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--n-anchors", type=int, default=20)
    p.add_argument("--pool-episodes", type=int, default=6)
    p.add_argument("--eta", type=float, default=0.1, help="per-coordinate command change for finite differences")
    p.add_argument("--eta-check-anchors", type=int, default=4)
    p.add_argument("--val-dirs", type=int, default=3)
    p.add_argument("--val-rms", type=float, default=0.05, help="planned RMS of validation corrections")
    p.add_argument("--repeat-tol", type=float, default=1e-6, help="max nominal repeat difference (m / rad)")
    p.add_argument("--step-limit", type=int, default=45000)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--analyze-only", action="store_true")
    p.add_argument("--force", action="store_true", help="overwrite an existing diagnostic")
    a = p.parse_args()
    cfg = load_config(a.config, a.set)
    from flow_residual_rl.pipeline import runs_dir

    out = runs_dir(cfg) / "diagnostics" / f"jacobian_task{a.task_id}"
    out.mkdir(parents=True, exist_ok=True)
    if not a.analyze_only:
        if (out / "jacobians.npz").exists() and not a.force:
            sys.exit(f"{out} already has results; use --analyze-only, or --force to recollect")
        from flow_residual_rl.walltime import WallLedger

        _ledger = WallLedger(runs_dir(cfg), "jacobian_diag")
        collect(cfg, a, out)
    analyze(out, a.repeat_tol)
    print((out / "analysis.md").read_text())


if __name__ == "__main__":
    main()
