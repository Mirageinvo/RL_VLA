"""Offline follow-up on a finished Jacobian diagnostic (no simulator, no GR00T).

Question: methods with a small endpoint drift at s=C (lag, global_gram, local_*) show a LARGER
eef drift after the GR00T continuation chunk than dct/integrator. Is the endpoint at s=C
only "hidden" by the controller lag (correction energy pushed to the last chunk steps, whose
effect arrives after C), or does it come from GR00T reacting to the perturbation?

Prints, from the saved files only:
  1. lag kernel: ||K_l|| on eef rows (normalised) for l = 1..C -> how delayed the response is;
  2. per method: share of correction energy in each chunk step t = 1..C;
  3. per method: median normalised eef drift at every step 1..2C (C+1.. = GR00T continuation);
  4. per anchor: local_eef vs each method, sign of the per-anchor mean difference of the
     endpoint drift (s=C) and of the drift after continuation (s=2C).

  python scripts/jac_followup.py [--dir runs/diagnostics/jacobian_task0]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from flow_residual_rl import effects as fx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="runs/diagnostics/jacobian_task0")
    a = p.parse_args()
    out = Path(a.dir)
    anc = json.loads((out / "anchors.json").read_text())
    vmeta = json.loads((out / "validation_meta.json").read_text())
    jz = np.load(out / "jacobians.npz")
    raw = np.load(out / "validation_raw.npz")
    C, dims = anc["C"], anc["dims"]
    nd = len(dims)
    rows = anc["layout"]["groups"]["eef"]
    sc = np.array(vmeta["scales"])
    methods = vmeta["methods"]
    eff, act, key = raw["effects"], raw["actions"], raw["key"]
    a0 = jz["a0"]
    recs = {(r["anchor"], r["method"], r["dir"], r["sign"]): r
            for r in map(json.loads, (out / "validation.jsonl").read_text().splitlines()) if "skipped" not in r}

    # 1. lag kernel from the fit anchors
    fitn = [fx.normalise(jz["J"][i], sc) for i in vmeta["fit_idx"]]
    K = fx.fit_lag(fitn, nd)
    kn = [float(np.linalg.norm(K[l][rows])) for l in range(C)]
    print("1. lag kernel ||K_l|| (eef rows, normalised), l=1..C:", " ".join(f"{x:.2f}" for x in kn))

    # 2. energy per chunk step, 3. drift time course
    z = np.linalg.norm(eff[:, :, rows] / sc[rows][None, None], axis=2)  # [R, T]
    dlt = act[:, :, dims] - a0[key[:, 0]][:, :, dims]  # [R, C, nd]
    en = np.sum(dlt ** 2, axis=2)
    en = en / np.maximum(en.sum(axis=1, keepdims=True), 1e-30)
    T = z.shape[1]
    print(f"\n2. share of correction energy per chunk step t=1..{C}")
    print(f"{'method':<12} " + " ".join(f"{t:>5}" for t in range(1, C + 1)) + "   last2")
    for mi, m in enumerate(methods):
        s = key[:, 1] == mi
        e = en[s].mean(axis=0)
        print(f"{m:<12} " + " ".join(f"{x:5.2f}" for x in e) + f"   {e[-2:].sum():5.2f}")
    print(f"\n3. median normalised eef drift at steps 1..{T} (| marks s=C; after it GR00T continues)")
    print(f"{'method':<12} " + " ".join(f"{t:>5}" for t in range(1, T + 1)))
    for mi, m in enumerate(methods):
        s = key[:, 1] == mi
        med = np.median(z[s], axis=0)
        print(f"{m:<12} " + " ".join(f"{x:5.2f}" + ("|" if t == C else "") for t, x in enumerate(med, 1)))

    # 4. per-anchor sign test, local_eef vs others, over common draws
    ho = sorted({int(k) for k in key[:, 0]})
    print("\n4. per held-out anchor: local_eef better (lower mean over common draws) on how many anchors")
    print(f"{'vs':<12} {'endpoint s=C':>14} {'after cont s=2C':>16}")
    for m in methods:
        if m == "local_eef":
            continue
        wins_e = wins_c = n = 0
        for i in ho:
            common = [(d, s) for (ii, mm, d, s) in recs if ii == i and mm == m
                      and (i, "local_eef", d, s) in recs
                      and recs[(i, m, d, s)]["comparable"] and recs[(i, "local_eef", d, s)]["comparable"]]
            if not common:
                continue
            n += 1
            f = lambda x, k: np.mean([recs[(i, x, d, s)][k] for d, s in common])
            wins_e += f("local_eef", "endpoint_drift") < f(m, "endpoint_drift")
            wins_c += f("local_eef", "drift_after_continuation") < f(m, "drift_after_continuation")
        print(f"{m:<12} {wins_e:>9}/{n:<4} {wins_c:>11}/{n:<4}")


if __name__ == "__main__":
    main()
