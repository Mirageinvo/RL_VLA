"""Offline follow-up on a finished Jacobian diagnostic (no simulator, no GR00T).

Question: methods with a small endpoint drift at s=C (lag, global_gram, local_*) show a LARGER
eef drift after the GR00T continuation chunk than dct/integrator. Why, and how well does the
local linear model predict the real corrections?

Only comparable interventions (actual RMS >= 0.95 planned, not skipped). Aggregation as in the
main report: mean over the draws of an anchor first, then median (or mean, for profiles) over
anchors. Everything overall and per phase.

  1. lag kernel ||K_l|| on eef rows (normalised), l = 1..C;
  2. correction energy per chunk step; first-2 vs last-2 share, compared with dct as the
     control (DCT shape modes already put ~41% of their energy in each end pair);
  3. normalised eef drift at every step 1..2C (C+1.. = GR00T continuation);
     handoff proxy ||z_C - z_{C-1}|| (change of the effect over the last chunk step);
  4. local_eef vs each method on common draws: median dlog drift at s=C and s=2C (<0 = local
     smaller) and the number of anchors where local is smaller;
  5. linear prediction of the executed corrections: e_hat_s = J_s(anchor) @ delta_actual and
     with the global mean J of the fit anchors; rel. error ||e_hat - e|| / ||e|| per group at
     s = C/2 and s = C.

  python scripts/jac_followup.py [--dir runs/diagnostics/jacobian_task0]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from flow_residual_rl import effects as fx

PHASES = ("all", "grasp", "contact", "near", "free", "unknown")


def load(d: Path) -> dict:
    anc = json.loads((d / "anchors.json").read_text())
    vmeta = json.loads((d / "validation_meta.json").read_text())
    jz = np.load(d / "jacobians.npz")
    raw = np.load(d / "validation_raw.npz")
    recs = [json.loads(x) for x in (d / "validation.jsonl").read_text().splitlines() if x.strip()]
    ok = {(r["anchor"], r["method"], r["dir"], r["sign"]) for r in recs if "skipped" not in r and r.get("comparable")}
    methods = vmeta["methods"]
    keep = np.array([(int(i), methods[int(m)], int(di), int(s)) in ok for i, m, di, s in raw["key"]], bool)
    return {"anc": anc, "vmeta": vmeta, "J": jz["J"], "a0": jz["a0"], "methods": methods,
            "eff": raw["effects"][keep], "act": raw["actions"][keep], "key": raw["key"][keep],
            "phase": {i: b["phase"]["phase"] for i, b in enumerate(anc["anchors"])},
            "ep": {i: b["ep"] for i, b in enumerate(anc["anchors"])}}


def per_anchor(D, values, mi, phase):
    """values [R, ...] -> {anchor: mean over that anchor's draws} for method index mi and phase."""
    out = {}
    for i in sorted({int(k) for k in D["key"][:, 0]}):
        if phase != "all" and D["phase"][i] != phase:
            continue
        s = (D["key"][:, 0] == i) & (D["key"][:, 1] == mi)
        if s.any():
            out[i] = values[s].mean(axis=0)
    return out


def fmt(xs, w=5, p=2):
    return " ".join(f"{x:{w}.{p}f}" for x in xs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="runs/diagnostics/jacobian_task0")
    a = p.parse_args()
    D = load(Path(a.dir))
    anc, vmeta, methods = D["anc"], D["vmeta"], D["methods"]
    C, dims = anc["C"], anc["dims"]
    nd = len(dims)
    G = anc["layout"]["groups"]
    rows = G["eef"]
    sc = np.array(vmeta["scales"])
    eff, act, key = D["eff"], D["act"], D["key"]
    T = eff.shape[1]
    phases = [ph for ph in PHASES if ph == "all" or any(D["phase"][int(i)] == ph for i in key[:, 0])]
    print(f"comparable interventions: {len(key)}; held-out anchors: {len(set(key[:, 0].tolist()))}")

    # 1. lag kernel
    fitn = [fx.normalise(D["J"][i], sc) for i in vmeta["fit_idx"]]
    K = fx.fit_lag(fitn, nd)
    print("\n1. lag kernel ||K_l|| (eef rows, normalised), l=1..C:", fmt([np.linalg.norm(K[l][rows]) for l in range(C)]))

    z = np.linalg.norm(eff[:, :, rows] / sc[rows][None, None], axis=2)  # [R, T]
    dlt = act[:, :, dims] - D["a0"][key[:, 0]][:, :, dims]  # [R, C, nd] executed correction
    en = np.sum(dlt ** 2, axis=2)
    en = en / np.maximum(en.sum(axis=1, keepdims=True), 1e-30)
    hand =np.linalg.norm((eff[:, C - 1, rows] - eff[:, C - 2, rows]) / sc[rows][None], axis=1)

    for ph in phases:
        n_a = len({int(i) for i in key[:, 0] if ph == "all" or D["phase"][int(i)] == ph})
        n_e = len({D["ep"][int(i)] for i in key[:, 0] if ph == "all" or D["phase"][int(i)] == ph})
        print(f"\n================ phase {ph}: {n_a} anchors / {n_e} episodes")

        print("\n2. correction energy per chunk step (mean over anchors); end pairs vs dct")
        print(f"{'method':<12} {fmt(range(1, C + 1), 5, 0)}  first2 last2 last2/first2")
        for mi, m in enumerate(methods):
            pa = per_anchor(D, en, mi, ph)
            if not pa:
                continue
            e = np.mean(list(pa.values()), axis=0)
            print(f"{m:<12} {fmt(e)}  {e[:2].sum():6.2f} {e[-2:].sum():5.2f} {e[-2:].sum() / max(e[:2].sum(), 1e-12):11.2f}")

        print(f"\n3. normalised eef drift per step 1..{T} (median over anchors; '|' after s=C), handoff ||dz_C||")
        print(f"{'method':<12} " + " ".join(f"{t:>5}" + (" " if t == C else "") for t in range(1, T + 1)) + "  handoff")
        for mi, m in enumerate(methods):
            pa = per_anchor(D, z, mi, ph)
            if not pa:
                continue
            med = np.median(list(pa.values()), axis=0)
            hm = np.median(list(per_anchor(D, hand, mi, ph).values()))
            print(f"{m:<12} " + " ".join(f"{x:5.2f}" + ("|" if t == C else "") for t, x in enumerate(med, 1)) + f"  {hm:7.2f}")

        print("\n4. local_eef vs method, common draws: median dlog drift (<0 = local smaller), local smaller on k/n anchors")
        print(f"{'vs':<12} {'dlog s=C':>9} {'k/n':>6} {'dlog s=2C':>10} {'k/n':>6}")
        li = methods.index("local_eef")
        for mi, m in enumerate(methods):
            if m == "local_eef":
                continue
            dC, d2, n = [], [], 0
            for i in sorted({int(k) for k in key[:, 0]}):
                if ph != "all" and D["phase"][i] != ph:
                    continue
                sel = lambda x: {(int(k[2]), int(k[3])): r for r, k in enumerate(key)
                                 if int(k[0]) == i and int(k[1]) == x}
                L, M = sel(li), sel(mi)
                common = sorted(set(L) & set(M))
                if not common:
                    continue
                for s, out in ((C, dC), (T, d2)):
                    zl = np.mean([z[L[c], s - 1] for c in common])
                    zm = np.mean([z[M[c], s - 1] for c in common])
                    out.append(np.log(max(zl, 1e-12) / max(zm, 1e-12)))
                n += 1
            if n:
                print(f"{m:<12} {np.median(dC):9.2f} {sum(x < 0 for x in dC):>3}/{n:<2} "
                      f"{np.median(d2):10.2f} {sum(x < 0 for x in d2):>3}/{n:<2}")

        print("\n5. linear prediction of executed corrections, rel. error ||J delta - e|| / ||e|| (median over anchors)")
        groups = [g for g in ("eef_pos", "eef_rot", "obj_pos_task") if G.get(g)]
        Jbar = np.nanmean(np.stack([D["J"][i] for i in vmeta["fit_idx"]]), axis=0)
        steps = sorted({max(1, C // 2), C})
        print(f"{'method':<12} " + " ".join(f"{g + ' s=' + str(s) + ' loc/glob':>26}" for g in groups for s in steps))
        for mi, m in enumerate(methods):
            sel = key[:, 1] == mi
            if not sel.any():
                continue
            cells = []
            for g in groups:
                r = G[g]
                for s in steps:
                    el, eg = [], []
                    for row in np.where(sel)[0]:
                        i = int(key[row, 0])
                        if ph != "all" and D["phase"][i] != ph:
                            continue
                        d = dlt[row].reshape(-1)
                        e = eff[row, s - 1, r]
                        ne = np.linalg.norm(e)
                        el.append((i, np.linalg.norm(D["J"][i][s - 1][r] @ d - e) / max(ne, 1e-12) if ne > 0 else np.nan))
                        eg.append((i, np.linalg.norm(Jbar[s - 1][r] @ d - e) / max(ne, 1e-12) if ne > 0 else np.nan))
                    agg = []
                    for lst in (el, eg):
                        by = {}
                        for i, v in lst:
                            by.setdefault(i, []).append(v)
                        vals = [np.nanmean(v) for v in by.values() if np.isfinite(v).any()]
                        agg.append(np.median(vals) if vals else np.nan)
                    cells.append(f"{agg[0]:12.2f} / {agg[1]:<11.2f}")
            print(f"{m:<12} " + " ".join(cells))
        tiny = (np.abs(eff[:, C - 1, G["obj_pos_task"]]).max(axis=1) < 1e-6).mean() if G.get("obj_pos_task") else 0
        if ph == "all" and tiny:
            print(f"   note: {tiny:.0%} of interventions move no task object at s=C (obj_pos_task error undefined there, skipped)")


if __name__ == "__main__":
    main()
