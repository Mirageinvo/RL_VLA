"""Last geometry test: handoff-aware ("tail") OER vs fixed bases, both continuation protocols.

Endpoint OER (keep the eef pose at s=C) was refuted by continuation_diag.py: with the integrating
controller the deviation reappears after C even under a fixed continuation. Tail OER constrains
the corrections d (48-dim, normalised eef effects) by
    zero command sum  S d = 0,   end pose  J_C d = 0,   last-step change  (J_C - J_{C-1}) d = 0
and keeps the 6 directions of the null space with the largest intermediate response
sum_{h in 2,4,6} |J_h d|^2 (effects.tail_basis). Methods, all rank 6:
    dct          DCT mode-1 shape basis (zero sum)                       analytic control
    lag_tail     tail basis of the fitted controller-lag model            fixed
    global_tail  tail basis of the mean normalised J of the fit anchors   fixed
    local_tail   tail basis of the anchor's own J                         state-conditioned oracle

Held-out anchors from the VERIFIED snapshots of continuation_diag.py (no pool, no Jacobians
re-measured), nominal chunk = stored a0. Per direction one common random v projected on every
method's basis, equal planned RMS, both signs; every corrected chunk is run with the nominal GR00T
continuation replayed (fixed) and with GR00T re-planning (replanned); the two corrected chunks
must coincide (checked). Recorded per branch and protocol: eef drift at C and 2C, mean / max drift
over C+1..2C, shape change, share of the executed (clipped) correction in the command-sum
directions (|S d| / |d|, 0 for zero-sum), task-object displacement,
grasp / contact changes, success.

Pre-registered decision (overall, re-planned protocol, paired over common draws; best fixed =
fixed method with the lowest median drift at 2C):
  1. local_tail vs best fixed: median dlog drift(2C) <= log 0.75      (>= 25% smaller)
  2. local_tail vs best fixed: median dlog shape change >= log 0.90    (<= 10% less shape)
  3. criterion 1 holds in >= 2 independent episodes (episode median of anchor dlog)
  4. median dlog mean drift over C+1..2C <= 0 and of max drift <= log 1.10
All four -> continue_state_conditioned; otherwise stop_state_conditioned_oer.

  python scripts/tail_diag.py --config configs/pilot.yaml --task-id 0
  python scripts/tail_diag.py --task-id 0 --analyze-only
Output: runs/diagnostics/tail_task<id>/ (exit 2 incomplete, 3 insufficient signal, 4 snapshots unusable)
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

METHODS = ("dct", "lag_tail", "global_tail", "local_tail")
FIXED = ("dct", "lag_tail", "global_tail")
PROTOCOLS = ("fixed", "replanned")
PHASES = ("grasp", "contact", "near", "free", "unknown")
SHAPE_STEPS = (2, 4, 6)
SIGNS = (1, -1)
COMPARABLE = 0.95
CRIT = {"drift_ratio": 0.75, "shape_ratio": 0.90, "min_episodes": 2, "max_cont_ratio": 1.10}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def bases(Jn_fit, Jn_anchor, rows, C, nd):
    """Normalised Jacobians -> {method: tail/dct basis dict}."""
    return {"dct": fx.dct_oer(C, nd),
            "lag_tail": fx.tail_basis(fx.lag_J(fx.fit_lag(Jn_fit, nd)), rows, SHAPE_STEPS, nd),
            "global_tail": fx.tail_basis(np.nanmean(np.stack(Jn_fit), axis=0), rows, SHAPE_STEPS, nd),
            "local_tail": fx.tail_basis(Jn_anchor, rows, SHAPE_STEPS, nd)}


def branch_metrics(e, layout, sc, C):
    rows = layout["groups"]["eef"]
    z = np.linalg.norm(e[:, rows] / sc[rows][None], axis=1)
    G = layout["groups"]
    return {"drift_C": float(z[C - 1]), "drift_2C": float(z[-1]), "drift_cont_mean": float(z[C:].mean()),
            "drift_cont_max": float(z[C:].max()),
            "shape_change": float(np.sqrt(sum(np.sum((e[s - 1, rows] / sc[rows]) ** 2) for s in SHAPE_STEPS))),
            "obj_task_m_2C": float(np.abs(e[-1, G["obj_pos_task"]]).max(initial=0.0))}


# ----------------------------------------------------------------------------- collection


def collect(cfg, a, src, snaps_path, out):
    from flow_residual_rl.budget import StepCounter
    from flow_residual_rl.pipeline import make_provider, make_task, write_manifest
    from flow_residual_rl.restore import restore_to

    from continuation_diag import measure

    anc, vmeta = read_json(src / "anchors.json"), read_json(src / "validation_meta.json")
    jz = np.load(src / "jacobians.npz")
    with open(snaps_path, "rb") as f:
        snaps = pickle.load(f)
    keys = ("ep", "sid", "k", "elapsed", "split")
    ok = (isinstance(snaps, dict) and snaps.get("verified") is True and snaps.get("mode") == "reuse"
          and [tuple(s[k] for k in keys) for s in snaps["anchors"]] == [tuple(b[k] for k in keys) for b in anc["anchors"]]
          and all(np.array_equal(s["a0"], jz["a0"][i]) for i, s in enumerate(snaps["anchors"])))
    if not ok:
        log(f"[tail] {snaps_path}: snapshots missing, unverified or not matching {src}")
        return 4
    C, dims, layout = anc["C"], list(anc["dims"]), anc["layout"]
    nd, N = len(dims), anc["C"] * len(anc["dims"])
    rows, sc = layout["groups"]["eef"], np.array(vmeta["scales"])
    Jn = [fx.normalise(J, sc) for J in jz["J"]]
    Jn_fit = [Jn[i] for i in vmeta["fit_idx"]]
    ho = list(vmeta["heldout_idx"])

    # all bases first: frozen before any intervention
    B, ranks, saved = {}, {}, {}
    for i in ho:
        B[i] = bases(Jn_fit, Jn[i], rows, C, nd)
        for m in METHODS:
            ranks[f"{m}_{i}"] = {"shape": int(B[i][m]["rank_shape"]), "constraints": int(B[i][m].get("rank_constraints", nd))}
            saved[f"{m}_{i}_shape"] = B[i][m]["shape"]
    rank_ok = all(r["shape"] == 6 for r in ranks.values())
    if not rank_ok:
        log(f"[warn] not every basis has rank 6: { {k: v for k, v in ranks.items() if v['shape'] != 6} }")
    np.savez_compressed(out / "tail_bases.npz", **saved)
    meta = {"source": str(src), "snapshots": str(snaps_path), "heldout_idx": ho, "methods": list(METHODS),
            "val_dirs": a.val_dirs, "signs": list(SIGNS), "val_rms": a.val_rms, "seed": a.seed, "C": C,
            "ranks": ranks, "rank_ok": rank_ok, "criteria": CRIT, "complete": False}
    write_json(out / "tail_meta.json", meta)

    counter = StepCounter(out / "cost.json")
    counter.set_limit(("diagnostic",), a.step_limit)
    prov = make_provider(cfg, counter)
    task = make_task(cfg, a.task_id, counter, prov)
    write_manifest(cfg, out / "run_manifest.json", prov, {"args": vars(a)})

    def restore(t, s):
        restore_to(t, s, cfg["restore_mode"], "diagnostic")

    def run(snap, act, cont, seed):
        st, _ = fx.execute(task, snap, act, restore, measure)
        f_c = fx.interaction_flags(task)
        succ = -1
        for j, x in enumerate(cont if cont is not None else prov.propose(task.policy_observation(), None, seed).a0_env):
            task.step(x)
            st.append(measure(task))
            if succ < 0 and task.success:
                succ = C + j + 1
        return st, succ, f_c, fx.interaction_flags(task)

    rng = np.random.default_rng(a.seed)
    vf = out / "tail.jsonl"
    vf.unlink(missing_ok=True)
    EF, ER, ACT, KEY = [], [], [], []
    try:
        with counter.charge("diagnostic"):
            for i in ho:
                s = snaps["anchors"][i]
                a0 = jz["a0"][i]
                seed = derive_seed("jacdiag-cont", a.task_id, s["ep"], s["k"])
                nom_st, _ = fx.execute(task, s["snap"], a0, restore, measure)
                c0 = prov.propose(task.policy_observation(), None, seed).a0_env.copy()
                nom, nom_succ, _, nf2 = run(s["snap"], a0, c0, seed)
                if fx.max_state_diff(nom[:C], nom_st) > 0:
                    log(f"[warn] anchor {i}: nominal chunk not repeatable")
                for di in range(a.val_dirs):
                    v = rng.standard_normal(N)  # common to every method
                    for mi, m in enumerate(METHODS):
                        u = fx.project_direction(B[i][m]["shape"], v)
                        for sign in SIGNS:
                            rec = {"anchor": i, "ep": s["ep"], "phase": s["phase"], "method": m, "dir": di, "sign": sign,
                                   "rank_shape": int(B[i][m]["rank_shape"])}
                            if u is None:
                                rec["skipped"] = "empty shape subspace"
                            else:
                                d = (a.val_rms * np.sqrt(N) * u).reshape(C, nd)
                                act = a0.copy()
                                act[:, dims] = np.clip(a0[:, dims] + sign * d, task.ACTION_LOW[dims], task.ACTION_HIGH[dims])
                                ex = act[:, dims] - a0[:, dims]
                                rms = float(np.sqrt(np.mean(ex ** 2)))
                                stf, sf, _, ff2 = run(s["snap"], act, c0, seed)
                                str_, sr, _, fr2 = run(s["snap"], act, None, seed)
                                ef = np.stack([fx.effect(x, y) for x, y in zip(stf, nom)])
                                er = np.stack([fx.effect(x, y) for x, y in zip(str_, nom)])
                                rec.update(rms_planned=a.val_rms, rms_actual=rms, comparable=rms >= COMPARABLE * a.val_rms,
                                           exec_sum_frac=float(np.linalg.norm(ex.sum(axis=0)) / np.sqrt(C) / max(rms * np.sqrt(N), 1e-12)),
                                           chunk_repeat=float(np.abs(ef[:C] - er[:C]).max()), nominal_success_step=nom_succ)
                                for p, e, sc_, f2 in (("fixed", ef, sf, ff2), ("replanned", er, sr, fr2)):
                                    rec[p] = dict(branch_metrics(e, layout, sc, C), success_step=sc_,
                                                  changed=fx.flag_changes(f2, nf2))
                                EF.append(ef)
                                ER.append(er)
                                ACT.append(act)
                                KEY.append([i, mi, di, sign])
                            with open(vf, "a") as f:
                                f.write(json.dumps(rec) + "\n")
                counter.save()
                np.savez_compressed(out / "tail_raw.npz", effects_fixed=np.stack(EF), effects_replanned=np.stack(ER),
                                    actions=np.stack(ACT), key=np.array(KEY, int))
                log(f"[tail] anchor {i} phase={s['phase']} done steps={counter.steps['diagnostic']}")
        write_json(out / "tail_meta.json", dict(meta, complete=True))
    finally:
        counter.save()
        task.close()
    return 0


# ----------------------------------------------------------------------------- analysis


def _med(x):
    x = [v for v in x if v is not None and np.isfinite(v)]
    return float(np.median(x)) if x else None


def analyze(out: Path) -> dict:
    meta = read_json(out / "tail_meta.json")
    recs = [json.loads(x) for x in (out / "tail.jsonl").read_text().splitlines() if x.strip()]
    ho = set(meta["heldout_idx"])
    expected = {(i, m, d, s) for i in ho for m in meta["methods"] for d in range(meta["val_dirs"]) for s in meta["signs"]}
    seen = [(r["anchor"], r["method"], r["dir"], r["sign"]) for r in recs]
    complete = bool(meta.get("complete")) and set(seen) == expected and len(seen) == len(set(seen))
    used = {(r["anchor"], r["method"], r["dir"], r["sign"]): r for r in recs if "skipped" not in r and r.get("comparable")}
    ep = {r["anchor"]: r["ep"] for r in recs}
    phase = {r["anchor"]: r["phase"] for r in recs}
    chunk_rep = max((r.get("chunk_repeat", 0.0) for r in used.values()), default=0.0)

    def table(idx, p):
        t = {}
        for m in meta["methods"]:
            per = {}
            for (i, mm, d, s), r in used.items():
                if mm == m and i in idx:
                    per.setdefault(i, []).append(r)
            row = {k: _med([np.mean([r[p][k] for r in rs]) for rs in per.values()])
                   for k in ("drift_C", "drift_2C", "drift_cont_mean", "drift_cont_max", "shape_change", "obj_task_m_2C")}
            row["exec_sum_frac"] = _med([np.mean([r["exec_sum_frac"] for r in rs]) for rs in per.values()])
            rs = [r for v in per.values() for r in v]
            row["grasp_changed"] = float(np.mean([bool(r[p]["changed"]["grasp"]) for r in rs])) if rs else None
            row["success_in_window"] = float(np.mean([r[p]["success_step"] > 0 for r in rs])) if rs else None
            row["n_anchors"] = len(per)
            t[m] = row
        return t

    def paired(idx, ref, p, k):
        """per anchor: log(mean local_tail / mean ref) over common draws."""
        out = {}
        for i in sorted(idx):
            common = [(d, s) for d in range(meta["val_dirs"]) for s in meta["signs"]
                      if (i, "local_tail", d, s) in used and (i, ref, d, s) in used]
            if common:
                lo = np.mean([used[(i, "local_tail", d, s)][p][k] for d, s in common])
                rf = np.mean([used[(i, ref, d, s)][p][k] for d, s in common])
                out[i] = float(np.log(max(lo, 1e-12) / max(rf, 1e-12)))
        return out

    def paired_summary(idx, p):
        res = {}
        for ref in FIXED:
            res[ref] = {}
            for k in ("drift_C", "drift_2C", "drift_cont_mean", "drift_cont_max", "shape_change"):
                d = paired(idx, ref, p, k)
                res[ref][k] = {"median_dlog": _med(list(d.values())), "local_smaller": sum(x < 0 for x in d.values()),
                               "n_anchors": len(d), "n_episodes": len({ep[i] for i in d})}
        return res

    groups = {"all": sorted(ho)}
    for ph in PHASES:
        idx = [i for i in sorted(ho) if phase.get(i) == ph]
        if idx:
            groups[ph] = idx
    res = {g: {"n_anchors": len(idx), "n_episodes": len({ep[i] for i in idx if i in ep}),
               "tables": {p: table(idx, p) for p in PROTOCOLS}, "paired": {p: paired_summary(idx, p) for p in PROTOCOLS}}
           for g, idx in groups.items()}

    # ---- pre-registered decision (overall, re-planned)
    tab = res["all"]["tables"]["replanned"]
    cand = [m for m in FIXED if tab[m]["drift_2C"] is not None]
    dec = {"criteria": CRIT, "protocol": "replanned", "complete": complete, "rank_ok": meta.get("rank_ok"),
           "sufficient": bool(used) and bool(cand)}
    if dec["sufficient"]:
        best = min(cand, key=lambda m: tab[m]["drift_2C"])
        idx = groups["all"]
        d2 = paired(idx, best, "replanned", "drift_2C")
        dsh = paired(idx, best, "replanned", "shape_change")
        dmean = paired(idx, best, "replanned", "drift_cont_mean")
        dmax = paired(idx, best, "replanned", "drift_cont_max")
        by_ep = {}
        for i, x in d2.items():
            by_ep.setdefault(ep[i], []).append(x)
        ep_ok = sorted(e for e, xs in by_ep.items() if np.median(xs) <= np.log(CRIT["drift_ratio"]))
        c = {"1_drift_2C": (_med(list(d2.values())) or 0) <= np.log(CRIT["drift_ratio"]),
             "2_shape": (_med(list(dsh.values())) if dsh else -np.inf) >= np.log(CRIT["shape_ratio"]),
             "3_episodes": len(ep_ok) >= CRIT["min_episodes"],
             "4_continuation": ((_med(list(dmean.values())) if dmean else np.inf) <= 0
                                and (_med(list(dmax.values())) if dmax else np.inf) <= np.log(CRIT["max_cont_ratio"]))}
        if not d2:
            c["1_drift_2C"] = False
        dec.update(best_fixed=best, median_dlog_drift_2C=_med(list(d2.values())), median_dlog_shape=_med(list(dsh.values())),
                   median_dlog_cont_mean=_med(list(dmean.values())), median_dlog_cont_max=_med(list(dmax.values())),
                   episodes_meeting_1=ep_ok, n_episodes=len(by_ep), checks={k: bool(v) for k, v in c.items()},
                   decision="continue_state_conditioned" if all(c.values()) else "stop_state_conditioned_oer")
        gl = paired(idx, "global_tail", "replanned", "drift_2C")
        dec["local_vs_global_tail_median_dlog_2C"] = _med(list(gl.values()))
    r = {"complete": complete, "n_records": len(recs), "n_used": len(used), "chunk_repeat_max": chunk_rep,
         "groups": res, "decision": dec, "ranks_ok": meta.get("rank_ok")}
    write_json(out / "analysis.json", r)
    (out / "analysis.md").write_text(render_md(r, meta))
    write_json(out / "decision.json", dec)
    return r


def _f(x, p=3):
    return "-" if x is None else f"{x:.{p}f}"


def render_md(r, meta) -> str:
    dec = r["decision"]
    L = ["# Tail OER diagnostic", ""]
    if not r["complete"]:
        L += ["**INCOMPLETE: missing / duplicate records or the run did not finish. Not a result.**", ""]
    if not dec.get("sufficient"):
        L += ["**INSUFFICIENT SIGNAL: no comparable interventions.**", ""]
    if not meta.get("rank_ok"):
        L += [f"**WARNING: not every basis has rank 6** ({ {k: v for k, v in meta['ranks'].items() if v['shape'] != 6} })", ""]
    L += [f"records {r['n_records']}, comparable {r['n_used']}; corrected chunk identical in both protocols: "
          f"max |de| = {r['chunk_repeat_max']:.1e}. Normalised eef drift; mean over an anchor's draws, then median over anchors.",
          f"constraint ranks (sum + pose + last-step change): "
          f"{sorted({v['constraints'] for k, v in meta['ranks'].items() if not k.startswith('dct')})}", ""]
    if dec.get("sufficient"):
        L += ["## Decision (pre-registered, overall, re-planned protocol)", "",
              f"best fixed method: **{dec['best_fixed']}**; local_tail vs it (paired median dlog): drift 2C "
              f"{_f(dec['median_dlog_drift_2C'], 2)} (need <= {np.log(CRIT['drift_ratio']):.2f}), shape "
              f"{_f(dec['median_dlog_shape'], 2)} (need >= {np.log(CRIT['shape_ratio']):.2f}), mean drift C+1..2C "
              f"{_f(dec['median_dlog_cont_mean'], 2)} (need <= 0), max {_f(dec['median_dlog_cont_max'], 2)} "
              f"(need <= {np.log(CRIT['max_cont_ratio']):.2f}); episodes meeting 1: {dec['episodes_meeting_1']} of "
              f"{dec['n_episodes']} (need >= {CRIT['min_episodes']})",
              f"checks: {dec['checks']}", f"local_tail vs global_tail, median dlog drift 2C: "
              f"{_f(dec['local_vs_global_tail_median_dlog_2C'], 2)}", "", f"**decision: {dec['decision']}**", ""]
    for g, x in r["groups"].items():
        L += [f"## {g}: {x['n_anchors']} anchors / {x['n_episodes']} episodes", ""]
        for p in PROTOCOLS:
            L += [f"### {p} continuation", "",
                  "| method | n | drift C | drift 2C | mean C+1..2C | max C+1..2C | shape | exec sum frac | task obj m 2C | grasp chg | success |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
            for m, t in x["tables"][p].items():
                L.append(f"| {m} | {t['n_anchors']} | {_f(t['drift_C'])} | {_f(t['drift_2C'])} | {_f(t['drift_cont_mean'])} | "
                         f"{_f(t['drift_cont_max'])} | {_f(t['shape_change'])} | {_f(t['exec_sum_frac'])} | "
                         f"{_f(t['obj_task_m_2C'], 4)} | {_f(t['grasp_changed'], 2)} | {_f(t['success_in_window'], 2)} |")
            L += ["", "local_tail vs fixed method, paired median dlog (local smaller on k/n anchors):", "",
                  "| vs | drift C | drift 2C | mean C+1..2C | max C+1..2C | shape (>0 = local more) | anchors / episodes |",
                  "|---|---|---|---|---|---|---|"]
            for ref, v in x["paired"][p].items():
                cell = lambda k: f"{_f(v[k]['median_dlog'], 2)} ({v[k]['local_smaller']}/{v[k]['n_anchors']})"  # noqa: E731
                L.append(f"| {ref} | {cell('drift_C')} | {cell('drift_2C')} | {cell('drift_cont_mean')} | "
                         f"{cell('drift_cont_max')} | {cell('shape_change')} | {v['drift_2C']['n_anchors']} / "
                         f"{v['drift_2C']['n_episodes']} |")
            L.append("")
    return "\n".join(L) + "\n"


def main():
    p = base_parser(__doc__)
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--val-dirs", type=int, default=3)
    p.add_argument("--val-rms", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--step-limit", type=int, default=12000)
    p.add_argument("--analyze-only", action="store_true")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    cfg = load_config(a.config, a.set)
    from flow_residual_rl.pipeline import runs_dir

    d = runs_dir(cfg) / "diagnostics"
    src, out = d / f"jacobian_task{a.task_id}", d / f"tail_task{a.task_id}"
    out.mkdir(parents=True, exist_ok=True)
    if not a.analyze_only:
        if (out / "tail_raw.npz").exists() and not a.force:
            sys.exit(f"{out} already has results; use --analyze-only, or --force to recollect")
        from flow_residual_rl.walltime import WallLedger

        _ledger = WallLedger(runs_dir(cfg), "tail_diag")
        for f in ("tail_raw.npz", "tail.jsonl", "tail_meta.json", "analysis.json", "analysis.md", "decision.json"):
            (out / f).unlink(missing_ok=True)
        code = collect(cfg, a, src, d / f"continuation_task{a.task_id}" / "anchor_snaps.pkl", out)
        if code:
            sys.exit(code)
    r = analyze(out)
    print((out / "analysis.md").read_text())
    if not r["complete"]:
        sys.exit(2)
    if not r["decision"].get("sufficient"):
        sys.exit(3)


if __name__ == "__main__":
    main()
