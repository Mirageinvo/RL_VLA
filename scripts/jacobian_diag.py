"""Physical-effect Jacobian diagnostic for one task (post-Stage-A; no bank, no RL).

1. Pool: a few base-GR00T episodes from TRAIN init states; at every chunk boundary a snapshot,
   the proposal (a0, context, proprio), interaction flags / phase (grasp > contact > near > free;
   'unknown' if a check failed) and the physical states of the ORIGINAL execution of the chunk.
2. Anchors: n_anchors boundaries stratified by phase; episodes alternate FIT / HELD-OUT, so no
   held-out anchor shares an episode with a fit anchor.
3. Per anchor: the nominal chunk twice after restore, each compared with the original pool
   execution and with each other (restore check, no extra steps); a finite-difference Jacobian
   of the physical effect after s = 1..C steps w.r.t. each command coordinate (symmetric step,
   flagged one-sided at the bounds, step sizes stored, causality check); on the first few fit
   anchors again with eta/2 (step-size sensitivity: nonlinearity + FD error + noise).
4. Models on FIT anchors: integrator, fixed controller lag, global_mean (prediction) and bases
   from DCT, integrator, lag, global_gram, local_eef (anchor's own J, eef rows) and local_full
   (eef + gripper + task objects). Fit anchors, scales and tolerance are frozen in
   validation_meta.json and reused by the offline analysis.
5. Validation on HELD-OUT anchors, real simulator: per direction ONE common random v in R^N,
   projected onto each method's SHAPE subspace (rotation invariant), equal planned RMS, both
   signs; corrected chunk + one GR00T continuation chunk (same seed as the nominal one).
   Recorded: eef / gripper / task-object / other-object effects, grasp/contact flag changes,
   success step, actual RMS (comparable iff >= 0.95 planned), raw effects and actions.
6. analysis.json / analysis.md, also per phase; --analyze-only re-runs it offline and refuses
   to call an incomplete validation complete (exit code 2).

Every env.step is charged to the `diagnostic` bucket (hard limit --step-limit) and the run is in
the project wall-time ledger. Output: runs/diagnostics/jacobian_task<id>/
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from flow_residual_rl import effects as fx
from flow_residual_rl.config import base_parser, derive_seed, load_config, read_json, write_json

PHASES = ("grasp", "contact", "near", "free", "unknown")
SHAPE_STEPS = (2, 4, 6)
METHODS = ("dct", "integrator", "lag", "global_gram", "local_eef", "local_full")
LOCAL = ("local_eef", "local_full")
SIGNS = (1, -1)
COMPARABLE = 0.95  # actual / planned RMS


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------- collection


def collect_pool(task, prov, ids, n_ep, C):
    bounds, warned = [], False
    for ep in range(n_ep):
        sid = int(ids[ep % len(ids)])
        task.reset(sid, settle_bucket="diagnostic")
        k = 0
        while task.horizon_left >= C:
            snap = task.snapshot()
            phys = fx.read_physical(task)
            ph = fx.phase_label(task, phys)
            if ph["errors"] and not warned:
                log(f"[warn] interaction check failed, phase='unknown': {ph['errors'][:2]}")
                warned = True
            prop = prov.propose(task.policy_observation(), None, derive_seed("jacdiag-pool", task.task_id, ep, k))
            orig, done = [], False
            for t in range(C):
                _, s = task.step(prop.a0_env[t])
                orig.append(fx.read_physical(task))
                if s:
                    done = True
                    break
            if len(orig) == C and task.horizon_left >= C:  # room for one continuation chunk after the anchor chunk
                bounds.append({"ep": ep, "sid": sid, "k": k, "elapsed": int(task.elapsed) - C, "snap": snap,
                               "a0": prop.a0_env.copy(), "context": np.asarray(prop.context, np.float32),
                               "proprio": np.asarray(prop.proprio, np.float32), "phase": ph, "orig": orig})
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


def run_chunk(task, b, actions, restore, cont_seed, prov):
    """Corrected chunk + one GR00T continuation chunk. States after every step, first success
    step (1-based, -1 none; the real policy would have stopped there), flags after C and 2C."""
    states, succ = fx.execute(task, b["snap"], actions, restore, fx.read_physical)
    C = len(actions)
    f_c = fx.interaction_flags(task)
    prop = prov.propose(task.policy_observation(), None, cont_seed)
    for i, a in enumerate(prop.a0_env):
        task.step(a)
        states.append(fx.read_physical(task))
        if succ < 0 and task.success:
            succ = C + i + 1
    return states, succ, f_c, fx.interaction_flags(task)


def model_bases(fitn, layout, C, nd):
    rows = layout["groups"]["eef"]
    return {"dct": fx.dct_oer(C, nd),
            "integrator": fx.oer_basis([fx.integrator_J(fx.fit_integrator(fitn, nd), C)], rows, C, SHAPE_STEPS),
            "lag": fx.oer_basis([fx.lag_J(fx.fit_lag(fitn, nd))], rows, C, SHAPE_STEPS),
            "global_gram": fx.oer_basis(fitn, rows, C, SHAPE_STEPS)}


def local_bases(Jn, layout, C):
    G = layout["groups"]
    return {"local_eef": fx.oer_basis([Jn], G["eef"], C, SHAPE_STEPS),
            "local_full": fx.oer_basis([Jn], G["full"], C, SHAPE_STEPS)}


def collect(cfg, a, out):
    from flow_residual_rl.budget import StepCounter
    from flow_residual_rl.pipeline import make_provider, make_task, task_splits, write_manifest
    from flow_residual_rl.restore import restore_to

    counter = StepCounter(out / "cost.json")
    counter.set_limit(("diagnostic",), a.step_limit)
    prov = make_provider(cfg, counter)
    task = make_task(cfg, a.task_id, counter, prov)
    try:
        _collect(cfg, a, out, counter, prov, task, restore_to, task_splits, write_manifest)
    finally:
        counter.save()
        task.close()


def _collect(cfg, a, out, counter, prov, task, restore_to, task_splits, write_manifest):
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
        layout = fx.feature_layout(len(p0["grip_qpos"]), p0["obj_names"], p0["task_objects"])
        log(f"[layout] objects={layout['obj_names']} task_objects={layout['task_objects']}")
        Js, Jh, half_idx, fd = [], [], [], []
        for i, b in enumerate(anchors):
            t0 = time.time()
            n1, s1 = fx.execute(task, b["snap"], b["a0"], restore, fx.read_physical)
            n2, s2 = fx.execute(task, b["snap"], b["a0"], restore, fx.read_physical)
            b["restore_vs_orig"] = max(fx.max_state_diff(n1, b["orig"]), fx.max_state_diff(n2, b["orig"]))
            b["restore_repeat"] = fx.max_state_diff(n1, n2)
            b["repeat_max"] = max(b["restore_vs_orig"], b["restore_repeat"])
            b["nominal_success_step"] = s1
            J, info = fx.fd_jacobian(task, b["snap"], b["a0"], dims, a.eta, restore, fx.read_physical,
                                     task.ACTION_LOW, task.ACTION_HIGH, n1)
            b.update(fd_one_sided=info["n_one_sided"], fd_unmeasured=info["n_unmeasured"],
                     fd_causality_max=info["causality_max"],
                     fd_branches_with_success=int(sum(x > 0 for x in info["branch_success_steps"])),
                     fd_branches=len(info["branch_success_steps"]))
            Js.append(J)
            fd.append(info)
            if b["split"] == "fit" and len(half_idx) < a.eta_check_anchors:
                Jh.append(fx.fd_jacobian(task, b["snap"], b["a0"], dims, a.eta / 2, restore, fx.read_physical,
                                         task.ACTION_LOW, task.ACTION_HIGH, n1)[0])
                half_idx.append(i)
            counter.save()
            log(f"[jac] anchor {i + 1}/{len(anchors)} {b['split']} phase={b['phase']['phase']} t={b['elapsed']} "
                f"restore_vs_orig={b['restore_vs_orig']:.1e} repeat={b['restore_repeat']:.1e} "
                f"one_sided={b['fd_one_sided']} causality={b['fd_causality_max']:.1e} "
                f"success_branches={b['fd_branches_with_success']}/{b['fd_branches']} "
                f"wall={time.time() - t0:.0f}s steps={counter.steps['diagnostic']}")
        np.savez_compressed(out / "jacobians.npz", J=np.stack(Js), J_half=np.stack(Jh) if Jh else np.zeros(0),
                            half_idx=np.array(half_idx, int), a0=np.stack([b["a0"] for b in anchors]),
                            context=np.stack([b["context"] for b in anchors]),
                            proprio=np.stack([b["proprio"] for b in anchors]),
                            fd_h=np.stack([f["h"] for f in fd]), fd_kind=np.stack([f["kind"] for f in fd]))
        meta = [{k: v for k, v in b.items() if k not in ("snap", "a0", "context", "proprio", "orig")} for b in anchors]
        write_json(out / "anchors.json", {"anchors": meta, "layout": layout, "args": vars(a), "C": C, "dims": dims,
                                          "restore_mode": cfg["restore_mode"], "epsilon_rms": cfg["epsilon_rms"]})

        # ---- validation on held-out anchors (real interventions); frozen choices first
        ok = [b["repeat_max"] <= a.repeat_tol for b in anchors]
        fit_idx = [i for i, b in enumerate(anchors) if b["split"] == "fit" and ok[i]]
        ho_idx = [i for i, b in enumerate(anchors) if b["split"] == "heldout" and ok[i]]
        sc = fx.group_scales([Js[i] for i in fit_idx], layout)
        glob = model_bases([fx.normalise(Js[i], sc) for i in fit_idx], layout, C, nd)
        vmeta = {"fit_idx": fit_idx, "heldout_idx": ho_idx, "scales": sc.tolist(), "repeat_tol": a.repeat_tol,
                 "methods": list(METHODS), "val_dirs": a.val_dirs, "signs": list(SIGNS), "val_rms": a.val_rms,
                 "comparable_min_ratio": COMPARABLE, "complete": False}
        write_json(out / "validation_meta.json", vmeta)
        saved = {f"{m}_{k}": v[k] for m, v in glob.items() for k in ("end", "shape")}
        rng = np.random.default_rng(a.seed + 1)
        vf = out / "validation.jsonl"
        vf.unlink(missing_ok=True)
        raw_e, raw_act, raw_key = [], [], []
        for i in ho_idx:
            b = anchors[i]
            cont = derive_seed("jacdiag-cont", a.task_id, b["ep"], b["k"])
            nom, nom_succ, nf_c, nf_2c = run_chunk(task, b, b["a0"], restore, cont, prov)
            bases = dict(glob, **local_bases(fx.normalise(Js[i], sc), layout, C))
            for m in LOCAL:
                saved.update({f"{m}_{i}_end": bases[m]["end"], f"{m}_{i}_shape": bases[m]["shape"]})
            for di in range(a.val_dirs):
                v = rng.standard_normal(N)  # common to every method
                for m in METHODS:
                    u = fx.project_direction(bases[m]["shape"], v)
                    for sign in SIGNS:
                        rec = {"anchor": i, "ep": b["ep"], "phase": b["phase"]["phase"], "method": m, "dir": di,
                               "sign": sign, "rank_shape": int(bases[m]["shape"].shape[1])}
                        if u is None:
                            rec["skipped"] = "empty shape subspace"
                        else:
                            delta = (a.val_rms * np.sqrt(N) * u).reshape(C, nd)
                            act = b["a0"].copy()
                            act[:, dims] = np.clip(b["a0"][:, dims] + sign * delta, task.ACTION_LOW[dims],
                                                   task.ACTION_HIGH[dims])
                            st, succ, f_c, f_2c = run_chunk(task, b, act, restore, cont, prov)
                            e = np.stack([fx.effect(x, y) for x, y in zip(st, nom)])
                            rms = float(np.sqrt(np.mean((act[:, dims] - b["a0"][:, dims]) ** 2)))
                            rec.update(rms_planned=a.val_rms, rms_actual=rms, comparable=rms >= COMPARABLE * a.val_rms,
                                       success_step=succ, nominal_success_step=nom_succ,
                                       **fx.intervention_metrics(e, layout, sc, SHAPE_STEPS, C),
                                       changed_end=fx.flag_changes(f_c, nf_c), changed_cont=fx.flag_changes(f_2c, nf_2c))
                            raw_e.append(e)
                            raw_act.append(act)
                            raw_key.append([i, METHODS.index(m), di, sign])
                        with open(vf, "a") as f:
                            f.write(json.dumps(rec) + "\n")
            counter.save()
            if raw_e:
                np.savez_compressed(out / "validation_raw.npz", effects=np.stack(raw_e), actions=np.stack(raw_act),
                                    key=np.array(raw_key, int))
            log(f"[val] anchor {i} phase={b['phase']['phase']} done, steps={counter.steps['diagnostic']}")
        np.savez_compressed(out / "validation_bases.npz", **saved)
        write_json(out / "validation_meta.json", dict(vmeta, complete=True))


# ----------------------------------------------------------------------------- analysis


def _med(x):
    x = [v for v in x if v is not None and np.isfinite(v)]
    return float(np.median(x)) if x else None


def _validation(recs, vmeta, anchors):
    ho = set(vmeta["heldout_idx"])
    stray = sorted({r["anchor"] for r in recs if r["anchor"] not in ho})
    recs = [r for r in recs if r["anchor"] in ho]
    expected = {(i, m, d, s) for i in ho for m in vmeta["methods"] for d in range(vmeta["val_dirs"]) for s in vmeta["signs"]}
    present = {(r["anchor"], r["method"], r["dir"], r["sign"]) for r in recs}
    missing = sorted(expected - present)
    used = [r for r in recs if "skipped" not in r and r.get("comparable")]
    per = {}
    for r in used:
        per.setdefault((r["anchor"], r["method"]), []).append(r)
    keys = ("endpoint_drift", "shape_change", "drift_after_continuation", "obj_pos_task_absmax_end",
            "obj_pos_task_absmax_cont", "obj_pos_other_absmax_end", "obj_pos_other_absmax_cont", "grip_absmax_end",
            "eef_pos_m_end", "rms_actual")

    def table(idx):
        t = {}
        for m in vmeta["methods"]:
            rows = {k: _med([float(np.mean([r[k] for r in per[(i, m)]])) for i in idx if (i, m) in per]) for k in keys}
            rs = [r for i in idx for r in per.get((i, m), [])]
            rows["frac_grasp_changed_cont"] = float(np.mean([bool(r["changed_cont"]["grasp"]) for r in rs])) if rs else None
            rows["frac_contact_changed_cont"] = float(np.mean([bool(r["changed_cont"]["contact"]) for r in rs])) if rs else None
            rows["frac_success_in_window"] = float(np.mean([r["success_step"] > 0 for r in rs])) if rs else None
            rows["n_anchors"] = sum((i, m) in per for i in idx)
            t[m] = rows
        return t

    def paired(idx):
        out = {}
        for loc in LOCAL:
            for m in vmeta["methods"]:
                if m in LOCAL:
                    continue
                de, ds, dom = [], [], 0
                for i in idx:
                    if (i, loc) in per and (i, m) in per:
                        e_l, e_m = (np.mean([r["endpoint_drift"] for r in per[(i, x)]]) for x in (loc, m))
                        s_l, s_m = (np.mean([r["shape_change"] for r in per[(i, x)]]) for x in (loc, m))
                        de.append(float(np.log(max(e_l, 1e-12) / max(e_m, 1e-12))))
                        ds.append(float(np.log(max(s_l, 1e-12) / max(s_m, 1e-12))))
                        dom += int(s_l >= s_m and e_l <= e_m)
                out[f"{loc}_vs_{m}"] = {"n_anchors": len(de), "median_dlog_endpoint": _med(de),
                                        "median_dlog_shape": _med(ds), "local_dominates": dom}
        return out

    idx_all = sorted(ho)
    by_phase = {}
    for ph in PHASES:
        idx = [i for i in idx_all if anchors[i]["phase"]["phase"] == ph]
        if idx:
            by_phase[ph] = {"n_anchors": len(idx), "n_episodes": len({anchors[i]["ep"] for i in idx}),
                            "table": table(idx), "paired": paired(idx)}
    return {"complete": bool(vmeta.get("complete")) and not missing and not stray, "n_records": len(recs),
            "n_used": len(used), "n_noncomparable": sum(1 for r in recs if "skipped" not in r and not r.get("comparable")),
            "n_skipped": sum(1 for r in recs if "skipped" in r), "missing": [list(x) for x in missing[:20]],
            "n_missing": len(missing), "stray_anchors": stray, "overall": {"table": table(idx_all), "paired": paired(idx_all)},
            "by_phase": by_phase}


def analyze(out: Path, repeat_tol: float) -> dict:
    meta = read_json(out / "anchors.json")
    z = np.load(out / "jacobians.npz")
    J_all, layout, C = z["J"], meta["layout"], meta["C"]
    nd = len(meta["dims"])
    anchors = meta["anchors"]
    G = layout["groups"]
    vmeta = read_json(out / "validation_meta.json")
    if vmeta is not None:  # the choices the validated bases were actually built with
        fit_i, ho_i, sc, tol = vmeta["fit_idx"], vmeta["heldout_idx"], np.array(vmeta["scales"]), vmeta["repeat_tol"]
    else:
        tol = repeat_tol
        fit_i = [i for i, b in enumerate(anchors) if b["split"] == "fit" and b["repeat_max"] <= tol]
        ho_i = [i for i, b in enumerate(anchors) if b["split"] == "heldout" and b["repeat_max"] <= tol]
        sc = fx.group_scales([J_all[i] for i in fit_i], layout)
    fitn = [fx.normalise(J_all[i], sc) for i in fit_i]
    hon = {i: fx.normalise(J_all[i], sc) for i in ho_i}
    eps = lambda idx: {ph: {"anchors": sum(anchors[i]["phase"]["phase"] == ph for i in idx),  # noqa: E731
                            "episodes": len({anchors[i]["ep"] for i in idx if anchors[i]["phase"]["phase"] == ph})}
                       for ph in PHASES}
    res = {"n_anchors": len(anchors), "n_fit": len(fit_i), "n_heldout": len(ho_i), "repeat_tol": tol,
           "repeat_tol_note": None if repeat_tol == tol else f"validation used repeat_tol={tol}; CLI value {repeat_tol} ignored",
           "excluded_restore": [i for i, b in enumerate(anchors) if b["repeat_max"] > tol],
           "restore_vs_orig_max": max(b["restore_vs_orig"] for b in anchors),
           "restore_repeat_max": max(b["restore_repeat"] for b in anchors),
           "fd_causality_max": max(b["fd_causality_max"] for b in anchors),
           "fd_one_sided_cols": sum(b["fd_one_sided"] for b in anchors),
           "fd_unmeasured_cols": sum(b["fd_unmeasured"] for b in anchors),
           "fd_branches_with_success": sum(b["fd_branches_with_success"] for b in anchors),
           "fd_branches": sum(b["fd_branches"] for b in anchors),
           "anchors_nominal_success": sum(b["nominal_success_step"] > 0 for b in anchors),
           "phase_errors": sorted({e for b in anchors for e in b["phase"]["errors"]})[:5],
           "phases": {"fit": eps(fit_i), "heldout": eps(ho_i)},
           "scales": {g: float(sc[G[g][0]]) for g in fx.GROUPS if G.get(g)},
           "task_objects": layout["task_objects"]}

    # 1. step-size sensitivity: J(eta) vs J(eta/2) on the same anchors
    res["eta_half_rel_diff"] = {g: {f"s{s}": _med([fx.rel_error(z["J_half"][k], J_all[i], G[g], s - 1)
                                                   for k, i in enumerate(z["half_idx"])]) for s in (2, 4, 8)}
                                for g in ("eef_pos", "eef_rot", "obj_pos_task") if G[g] and z["J_half"].size}

    # 2. prediction error of state-independent models on held-out anchors (pooled + absolute)
    preds = {"integrator": fx.integrator_J(fx.fit_integrator(fitn, nd), C), "lag": fx.lag_J(fx.fit_lag(fitn, nd)),
             "global_mean": np.nanmean(np.stack(fitn), axis=0)}
    groups = [g for g in ("eef_pos", "eef_rot", "grip", "obj_pos_task", "obj_pos_other") if G[g]]
    res["heldout_error"] = {m: {g: {f"s{s}": fx.pooled_errors(Jp, list(hon.values()), G[g], s - 1) for s in (2, 4, 6, 8)}
                                for g in groups} for m, Jp in preds.items()}
    res["heldout_error_by_phase"] = {
        ph: {g: fx.pooled_errors(preds["global_mean"], [hon[i] for i in ho_i if anchors[i]["phase"]["phase"] == ph],
                                 G[g], C - 1) for g in groups}
        for ph in PHASES if any(anchors[i]["phase"]["phase"] == ph for i in ho_i)}

    # 3. subspaces: local bases of every anchor vs the state-independent ones
    glob = model_bases(fitn, layout, C, nd)
    res["model_bases"] = {m: {"rank_end": B["rank_end"], "rank_shape": B["rank_shape"],
                              "end_vs_dct": fx.overlap(B["end"], glob["dct"]["end"]),
                              "shape_vs_dct": fx.overlap(B["shape"], glob["dct"]["shape"])} for m, B in glob.items()}
    sub = []
    for i in fit_i + ho_i:
        Jn = fx.normalise(J_all[i], sc)
        loc = fx.oer_basis([Jn], G["eef"], C, SHAPE_STEPS)
        d = {"anchor": i, "split": anchors[i]["split"], "phase": anchors[i]["phase"]["phase"],
             "rank_end": loc["rank_end"], "rank_shape": loc["rank_shape"],
             "sv_end_eef": [float(x) for x in np.linalg.svd(np.nan_to_num(Jn[C - 1][G["eef"]]), compute_uv=False)],
             "eef_resp_end_m": float(np.linalg.norm(np.nan_to_num(J_all[i][C - 1][G["eef_pos"]]))),
             "task_obj_resp_end_m": float(np.linalg.norm(np.nan_to_num(J_all[i][C - 1][G["obj_pos_task"]]))),
             "other_obj_resp_end_m": float(np.linalg.norm(np.nan_to_num(J_all[i][C - 1][G["obj_pos_other"]])))}
        for m, B in glob.items():
            d[f"ov_end_{m}"] = fx.overlap(loc["end"], B["end"])
            d[f"ov_shape_{m}"] = fx.overlap(loc["shape"], B["shape"])
            ang = fx.principal_angles_deg(loc["shape"], B["shape"])
            d[f"max_angle_shape_{m}"] = max(ang) if ang else None
        sub.append(d)
    res["subspaces_by_phase"] = {}
    for ph in PHASES:
        xs = [d for d in sub if d["phase"] == ph]
        if xs:
            res["subspaces_by_phase"][ph] = {
                "n": len(xs), "n_episodes": len({anchors[d["anchor"]]["ep"] for d in xs}),
                **{k: _med([d[k] for d in xs]) for k in xs[0] if k.startswith(("ov_", "max_angle", "rank_"))},
                **{k: _med([d[k] for d in xs]) for k in ("eef_resp_end_m", "task_obj_resp_end_m", "other_obj_resp_end_m")},
                "sv_end_eef": [float(v) for v in np.median(np.array([d["sv_end_eef"] for d in xs]), axis=0)]}
    res["per_anchor"] = sub

    # 4. validation interventions (held-out, real simulator)
    vf = out / "validation.jsonl"
    if vmeta is not None:
        recs = [json.loads(l) for l in open(vf)] if vf.exists() else []
        res["validation"] = _validation(recs, vmeta, anchors)
    write_json(out / "analysis.json", res)
    (out / "analysis.md").write_text(render_md(res))
    return res


def _f(x, p=3):
    if x is None:
        return "—"
    if isinstance(x, bool):
        return str(x)
    return f"{x:.{p}g}"


def _val_table(t):
    L = ["| method | n | endpoint_drift | shape_change | eef drift +1 GR00T chunk | task obj m (end / +1) | other obj m (end / +1) "
         "| grasp chg | contact chg | success in window | rms |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m, v in t.items():
        L.append(f"| {m} | {v['n_anchors']} | {_f(v['endpoint_drift'])} | {_f(v['shape_change'])} | {_f(v['drift_after_continuation'])} | "
                 f"{_f(v['obj_pos_task_absmax_end'])} / {_f(v['obj_pos_task_absmax_cont'])} | "
                 f"{_f(v['obj_pos_other_absmax_end'])} / {_f(v['obj_pos_other_absmax_cont'])} | "
                 f"{_f(v['frac_grasp_changed_cont'])} | {_f(v['frac_contact_changed_cont'])} | {_f(v['frac_success_in_window'])} | "
                 f"{_f(v['rms_actual'])} |")
    return L


def _paired(p):
    return ["| comparison | n | median dlog endpoint (<0 better) | median dlog shape (>0 better) | local dominates |",
            "|---|---|---|---|---|"] + [f"| {k} | {v['n_anchors']} | {_f(v['median_dlog_endpoint'])} | "
                                        f"{_f(v['median_dlog_shape'])} | {v['local_dominates']} |" for k, v in p.items()]


def render_md(r: dict) -> str:
    L = ["# Jacobian diagnostic\n"]
    v = r.get("validation")
    if v is not None and not v["complete"]:
        L.append(f"**INCOMPLETE VALIDATION: {v['n_missing']} missing intervention(s), stray anchors {v['stray_anchors']}. "
                 f"Not a result.**\n")
    L += [f"anchors: {r['n_anchors']} (fit {r['n_fit']}, held-out {r['n_heldout']}); excluded by restore (tol {r['repeat_tol']}): "
          f"{r['excluded_restore']}. {r['repeat_tol_note'] or ''}",
          f"restore vs original pool execution: max {r['restore_vs_orig_max']:.1e}; restore repeat: max {r['restore_repeat_max']:.1e}",
          f"finite differences: one-sided columns {r['fd_one_sided_cols']}, unmeasured {r['fd_unmeasured_cols']}, "
          f"causality max |J_s[t>=s]| {r['fd_causality_max']:.1e}; branches reaching success inside the chunk "
          f"{r['fd_branches_with_success']}/{r['fd_branches']}; anchors whose nominal chunk succeeds {r['anchors_nominal_success']}",
          f"phases (anchors / independent episodes): {r['phases']}",
          f"interaction-check errors: {r['phase_errors'] or 'none'}",
          f"task objects: {r['task_objects']}; group scales (RMS response at s=C per unit command): {r['scales']}\n",
          "## 1. Step-size sensitivity: rel. diff J(eta) vs J(eta/2)\n",
          "Mixes nonlinearity, finite-difference error and noise; small = the linear model is meaningful at this amplitude.\n"]
    for g, x in r["eta_half_rel_diff"].items():
        L.append(f"- {g}: " + ", ".join(f"{k}={_f(y)}" for k, y in x.items()))
    L.append("\n## 2. Held-out error of state-independent models (pooled rel. error | abs RMS error / true RMS, raw units)\n")
    L.append("| model | group | s=2 | s=4 | s=6 | s=8 |\n|---|---|---|---|---|---|")
    for m, gs in r["heldout_error"].items():
        for g, x in gs.items():
            L.append(f"| {m} | {g} | " + " | ".join(
                f"{_f(x[f's{s}']['pooled_rel'])} ({_f(x[f's{s}']['abs_rms'], 2)}/{_f(x[f's{s}']['true_rms'], 2)})"
                for s in (2, 4, 6, 8)) + " |")
    L.append("\nglobal_mean at s=C by phase (pooled rel. error):\n")
    for ph, gs in r["heldout_error_by_phase"].items():
        L.append(f"- {ph}: " + ", ".join(f"{g}={_f(x['pooled_rel'])}" for g, x in gs.items()))
    L.append("\n## 3. Subspaces (eef features)\n")
    L.append("state-independent bases: " + "; ".join(
        f"{m}: rank end/shape {x['rank_end']}/{x['rank_shape']}, overlap with DCT end {_f(x['end_vs_dct'])} / shape {_f(x['shape_vs_dct'])}"
        for m, x in r["model_bases"].items()))
    L.append("\nLocal (per-anchor) eef bases vs state-independent ones, median by phase "
             "(overlap 1 = same subspace; random 6-of-48 ~ 0.125):\n")
    L.append("| phase | anchors / eps | local rank end/shape | shape vs dct | vs integrator | vs lag | vs global_gram | end vs global_gram "
             "| max angle shape vs global (deg) | J_C eef_pos (m) | task obj (m) | other obj (m) |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
    for ph, x in r["subspaces_by_phase"].items():
        L.append(f"| {ph} | {x['n']} / {x['n_episodes']} | {_f(x['rank_end'])}/{_f(x['rank_shape'])} | {_f(x.get('ov_shape_dct'))} | "
                 f"{_f(x.get('ov_shape_integrator'))} | {_f(x.get('ov_shape_lag'))} | {_f(x.get('ov_shape_global_gram'))} | "
                 f"{_f(x.get('ov_end_global_gram'))} | {_f(x.get('max_angle_shape_global_gram'))} | {_f(x['eef_resp_end_m'])} | "
                 f"{_f(x['task_obj_resp_end_m'])} | {_f(x['other_obj_resp_end_m'])} |")
    if v is not None:
        L.append(f"\n## 4. Validation on held-out anchors ({v['n_records']} records, {v['n_used']} used, "
                 f"{v['n_noncomparable']} not comparable after clipping, {v['n_skipped']} skipped for an empty subspace)\n")
        L.append("One common random direction per draw, projected onto each method's SHAPE subspace; equal planned RMS. "
                 "Normalised eef units. Wanted: large shape_change AND small endpoint_drift — read both, not their ratio.\n")
        L.append("### all held-out anchors\n")
        L += _val_table(v["overall"]["table"]) + [""] + _paired(v["overall"]["paired"])
        for ph, x in v["by_phase"].items():
            L.append(f"\n### phase {ph}: {x['n_anchors']} anchors from {x['n_episodes']} episode(s)\n")
            L += _val_table(x["table"]) + [""] + _paired(x["paired"])
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
    p.add_argument("--repeat-tol", type=float, default=1e-6, help="max restore difference (m / rad), vs original and repeat")
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
        for f in ("validation_meta.json", "validation.jsonl", "validation_raw.npz", "validation_bases.npz"):
            (out / f).unlink(missing_ok=True)
        collect(cfg, a, out)
    r = analyze(out, a.repeat_tol)
    print((out / "analysis.md").read_text())
    if "validation" not in r or not r["validation"]["complete"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
