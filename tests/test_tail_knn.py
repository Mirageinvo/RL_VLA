"""scripts/tail_knn.py on synthetic Jacobians whose controller lag depends on the state."""
import json

import numpy as np

from flow_residual_rl import effects as fx

from test_continuation import _load
from test_effects import C, LAYOUT, ND, LinTask, jac


def _write(d, alphas, proprio, fit, ho, eps):
    d.mkdir(parents=True)
    Js = np.stack([jac(LinTask(alpha=al))[0] for al in alphas])
    n = len(alphas)
    np.savez_compressed(d / "jacobians.npz", J=Js, a0=np.zeros((n, C, 7)), proprio=np.array(proprio, float))
    anchors = [{"ep": eps[i], "phase": {"phase": "free"}} for i in range(n)]
    (d / "anchors.json").write_text(json.dumps({"anchors": anchors, "layout": LAYOUT, "C": C, "dims": list(range(ND))}))
    sc = fx.group_scales([Js[i] for i in fit], LAYOUT)
    (d / "validation_meta.json").write_text(json.dumps({"fit_idx": fit, "heldout_idx": ho, "scales": sc.tolist()}))


def test_knn_recovers_state_dependent_lag(tmp_path):
    tk = _load("tail_knn")
    # fit: three anchors for each of three controller lags (2 lags would be covered exactly by the
    # 12 suppressed mean-Gram directions); held-out: two episodes with different lags
    alphas = [0.1] * 3 + [0.5] * 3 + [0.9] * 3 + [0.1, 0.9, 0.5, 0.1]
    proprio = [[float(k)] * 8 for k in (0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 2, 1, 0)]
    eps = [0, 0, 0, 2, 2, 2, 4, 4, 4, 1, 1, 3, 3]
    _write(tmp_path / "d", alphas, proprio, list(range(9)), [9, 10, 11, 12], eps)
    o = tk.run(tmp_path / "d", 3)
    assert all(r["scores"][m]["sum"] < 1e-9 for r in o["per_anchor"] for m in tk.METHODS)
    for r in o["per_anchor"]:  # the right controller class is found
        assert {alphas[j] for j in r["neighbours"]} == {alphas[r["anchor"]]}
        assert r["scores"]["knn_tail"]["pose"] < 1e-6 < r["scores"]["global_tail"]["pose"]
    assert o["verdict"]["global_tail"]["pass"] and o["verdict"]["global_tail"]["episodes_passing"] == [1, 3]
    assert "knn_tail vs global_tail: PASS" in (tmp_path / "d" / "tail_knn.md").read_text()


def test_knn_without_state_dependence_does_not_pass(tmp_path):
    tk = _load("tail_knn")
    alphas = [0.5] * 10
    proprio = [[float(i)] * 8 for i in range(10)]
    _write(tmp_path / "d", alphas, proprio, list(range(6)), [6, 7, 8, 9], [0, 0, 2, 2, 4, 4, 1, 1, 3, 3])
    o = tk.run(tmp_path / "d", 3)
    assert not o["pass"]  # identical dynamics everywhere: knn cannot beat the global basis


def test_rank_deficient_knn_basis_cannot_pass(tmp_path, monkeypatch):
    tk = _load("tail_knn")
    alphas = [0.1] * 3 + [0.5] * 3 + [0.9] * 3 + [0.1, 0.9, 0.5, 0.1]
    proprio = [[float(k)] * 8 for k in (0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 2, 1, 0)]
    eps = [0, 0, 0, 2, 2, 2, 4, 4, 4, 1, 1, 3, 3]
    _write(tmp_path / "d", alphas, proprio, list(range(9)), [9, 10, 11, 12], eps)
    orig = fx.tail_basis_mean

    def one_column(Js, *a, **kw):  # the k=3 neighbour basis keeps a single direction
        out = orig(Js, *a, **kw)
        return dict(out, shape=out["shape"][:, :1]) if len(Js) == 3 else out

    monkeypatch.setattr(tk.fx, "tail_basis_mean", one_column)
    o = tk.run(tmp_path / "d", 3)
    for ctrl in tk.CONTROLS:
        v = o["verdict"][ctrl]
        assert not v["rank_ok"] and not v["pass"]
        assert {f["method"] for f in v["rank_failures"]} == {"knn_tail"} and len(v["rank_failures"]) == 4
    assert not o["pass"]
