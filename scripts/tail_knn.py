"""Offline check: can the local tail basis be predicted from the observation? (no simulator)

Predictor (the one simple candidate agreed with the reviewer): for a held-out anchor, the k=3
nearest FIT anchors in (proprio, nominal chunk a0), features standardised on the fit anchors;
basis = effects.tail_basis_mean over their normalised Jacobians. The held-out anchor's own
Jacobian is used ONLY for scoring.

Scores per basis B (6 orthonormal command-space columns), on the held-out anchor's normalised J,
eef rows, mean over the basis directions:
    pose   = |J_C B|_F / sqrt(6)                  end-pose response at C        (smaller = better)
    vel    = |(J_C - J_C-1) B|_F / sqrt(6)        last-step change              (smaller = better)
    shape  = sqrt(sum_h |J_h B|_F^2) / sqrt(6)    intermediate response h=2,4,6 (larger  = more shape)
    sum    = |S B|_F                              command-sum leak (0 for every tail basis)
Compared with global_tail (mean-Gram over all fit anchors) and lag_tail (fitted lag model);
local_tail (the anchor's own J) is the oracle reference.

Pre-registered pass (vs EACH control separately): knn_tail and the control have rank 6 (6 columns,
numerically independent) at EVERY held-out anchor (the per-direction scores would otherwise reward a
smaller basis), and in >= 2 held-out episodes the episode median of dlog(knn / control) is < 0 for
pose AND for vel, and the episode median shape ratio is >= 0.9.
Only then is knn_tail added to the 2C test on the saved snapshots (~2k sim steps).

  python scripts/tail_knn.py [--dir runs/diagnostics/jacobian_task0] [--k 3]
Writes <dir>/tail_knn.json and tail_knn.md.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from flow_residual_rl import effects as fx

SHAPE_STEPS = (2, 4, 6)
CONTROLS = ("global_tail", "lag_tail")
REQUIRED_RANK = 6  # every compared basis must span 6 directions at every held-out anchor
METHODS = ("knn_tail",) + CONTROLS + ("local_tail",)


def scores(J, B, rows, C, nd):
    Jr = np.nan_to_num(J[:, rows, :])
    k = max(B.shape[1], 1)
    return {"pose": float(np.linalg.norm(Jr[C - 1] @ B) / np.sqrt(k)),
            "vel": float(np.linalg.norm((Jr[C - 1] - Jr[C - 2]) @ B) / np.sqrt(k)),
            "shape": float(np.sqrt(sum(np.linalg.norm(Jr[h - 1] @ B) ** 2 for h in SHAPE_STEPS)) / np.sqrt(k)),
            "sum": float(np.linalg.norm(fx.sum_operator(C, nd) @ B)), "rank": int(B.shape[1]),
            "numerical_rank": int(np.linalg.matrix_rank(B, tol=1e-6)) if B.size else 0}


def features(z, idx):
    return np.concatenate([z["proprio"][idx].reshape(len(idx), -1), z["a0"][idx].reshape(len(idx), -1)], axis=1)


def run(d: Path, k: int) -> dict:
    anc = json.loads((d / "anchors.json").read_text())
    vmeta = json.loads((d / "validation_meta.json").read_text())
    z = np.load(d / "jacobians.npz")
    C, nd = anc["C"], len(anc["dims"])
    rows, sc = anc["layout"]["groups"]["eef"], np.array(vmeta["scales"])
    Jn = [fx.normalise(J, sc) for J in z["J"]]
    fit, ho = list(vmeta["fit_idx"]), list(vmeta["heldout_idx"])
    Xf = features(z, fit)
    mu, sd = Xf.mean(0), Xf.std(0)
    sd[sd < 1e-6] = 1.0  # constant features carry no distance
    Xf = (Xf - mu) / sd
    fixed = {"global_tail": fx.tail_basis_mean([Jn[i] for i in fit], rows, SHAPE_STEPS, nd)["shape"],
             "lag_tail": fx.tail_basis(fx.lag_J(fx.fit_lag([Jn[i] for i in fit], nd)), rows, SHAPE_STEPS, nd)["shape"]}
    per = []
    for i in ho:
        x = (features(z, [i]) - mu) / sd
        nn = [fit[j] for j in np.argsort(np.linalg.norm(Xf - x, axis=1))[:k]]
        B = {"knn_tail": fx.tail_basis_mean([Jn[j] for j in nn], rows, SHAPE_STEPS, nd)["shape"], **fixed,
             "local_tail": fx.tail_basis(Jn[i], rows, SHAPE_STEPS, nd)["shape"]}
        b = anc["anchors"][i]
        per.append({"anchor": i, "ep": b["ep"], "phase": b["phase"]["phase"], "neighbours": nn,
                    "neighbour_phases": [anc["anchors"][j]["phase"]["phase"] for j in nn],
                    "scores": {m: scores(Jn[i], B[m], rows, C, nd) for m in METHODS}})

    def dl(r, ctrl, key):
        return float(np.log(max(r["scores"]["knn_tail"][key], 1e-12) / max(r["scores"][ctrl][key], 1e-12)))

    eps = sorted({r["ep"] for r in per})
    verdict = {}
    for ctrl in CONTROLS:
        by_ep = {}
        for e in eps:
            rs = [r for r in per if r["ep"] == e]
            by_ep[e] = {"n_anchors": len(rs), "dlog_pose": float(np.median([dl(r, ctrl, "pose") for r in rs])),
                        "dlog_vel": float(np.median([dl(r, ctrl, "vel") for r in rs])),
                        "shape_ratio": float(np.median([np.exp(dl(r, ctrl, "shape")) for r in rs]))}
        ok = [e for e, v in by_ep.items() if v["dlog_pose"] < 0 and v["dlog_vel"] < 0 and v["shape_ratio"] >= 0.9]
        bad_rank = [{"anchor": r["anchor"], "method": m, "rank": r["scores"][m]["rank"],
                     "numerical_rank": r["scores"][m]["numerical_rank"]}
                    for r in per for m in ("knn_tail", ctrl)
                    if min(r["scores"][m]["rank"], r["scores"][m]["numerical_rank"]) != REQUIRED_RANK]
        verdict[ctrl] = {"by_episode": by_ep, "episodes_passing": ok, "rank_ok": not bad_rank, "rank_failures": bad_rank,
                         "pass": len(ok) >= 2 and not bad_rank,
                         "overall_median_dlog_pose": float(np.median([dl(r, ctrl, "pose") for r in per])),
                         "overall_median_dlog_vel": float(np.median([dl(r, ctrl, "vel") for r in per])),
                         "overall_median_shape_ratio": float(np.median([np.exp(dl(r, ctrl, "shape")) for r in per]))}
    out = {"k": k, "features": "proprio + a0 (standardised on fit anchors)", "fit_idx": fit, "heldout_idx": ho,
           "per_anchor": per, "verdict": verdict, "pass": all(v["pass"] for v in verdict.values()),
           "medians": {m: {key: float(np.median([r["scores"][m][key] for r in per])) for key in ("pose", "vel", "shape", "sum")}
                       for m in METHODS}}
    (d / "tail_knn.json").write_text(json.dumps(out, indent=1))
    (d / "tail_knn.md").write_text(render(out))
    return out


def render(o) -> str:
    L = ["# Tail basis from 3 nearest fit anchors (offline)", "",
         f"k={o['k']}, features: {o['features']}; held-out anchors {o['heldout_idx']}", "",
         "## medians over held-out anchors (normalised eef response per basis direction)", "",
         "| method | pose at C (lower) | last-step change (lower) | shape (higher) | sum leak |", "|---|---|---|---|---|"]
    for m, v in o["medians"].items():
        L.append(f"| {m} | {v['pose']:.4f} | {v['vel']:.4f} | {v['shape']:.3f} | {v['sum']:.1e} |")
    for ctrl, v in o["verdict"].items():
        L += ["", f"## knn_tail vs {ctrl}: {'PASS' if v['pass'] else 'fail'} (episodes passing {v['episodes_passing']}, "
                  f"rank {REQUIRED_RANK} everywhere: {'yes' if v['rank_ok'] else 'NO, ' + str(len(v['rank_failures'])) + ' failures'})", "",
              f"overall median dlog pose {v['overall_median_dlog_pose']:.2f}, vel {v['overall_median_dlog_vel']:.2f}, "
              f"shape ratio {v['overall_median_shape_ratio']:.2f}", "",
              "| episode | anchors | median dlog pose | median dlog vel | median shape ratio |", "|---|---|---|---|---|"]
        for e, x in v["by_episode"].items():
            L.append(f"| {e} | {x['n_anchors']} | {x['dlog_pose']:.2f} | {x['dlog_vel']:.2f} | {x['shape_ratio']:.2f} |")
    L += ["", "## per anchor", "", "| anchor | ep | phase | neighbours (phases) | knn pose / vel / shape | global pose / vel / shape |",
          "|---|---|---|---|---|---|"]
    for r in o["per_anchor"]:
        s = r["scores"]
        L.append(f"| {r['anchor']} | {r['ep']} | {r['phase']} | {r['neighbours']} ({', '.join(r['neighbour_phases'])}) | "
                 f"{s['knn_tail']['pose']:.4f} / {s['knn_tail']['vel']:.4f} / {s['knn_tail']['shape']:.3f} | "
                 f"{s['global_tail']['pose']:.4f} / {s['global_tail']['vel']:.4f} / {s['global_tail']['shape']:.3f} |")
    L += ["", f"**overall: {'PASS: add knn_tail to the 2C test' if o['pass'] else 'FAIL: do not extend the 2C test'}**", ""]
    return "\n".join(L)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", default="runs/diagnostics/jacobian_task0")
    p.add_argument("--k", type=int, default=3)
    a = p.parse_args()
    o = run(Path(a.dir), a.k)
    print((Path(a.dir) / "tail_knn.md").read_text())
    return 0 if o["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
