"""Jacobian-diagnostic math on synthetic systems with a known answer (no simulator)."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from flow_residual_rl import effects as fx

C, ND = 8, 6
OBJS = ["basket_1", "ketchup_1"]
LAYOUT = fx.feature_layout(2, OBJS, ["basket_1"])


def rodrigues(v):
    th = np.linalg.norm(v)
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


@pytest.mark.parametrize("angle", [1e-8, 1e-3, 0.5, 2.0, 3.1])
def test_rotvec_inverts_rodrigues(angle):
    rng = np.random.default_rng(0)
    for _ in range(20):
        ax = rng.standard_normal(3)
        v = angle * ax / np.linalg.norm(ax)
        np.testing.assert_allclose(fx.rotvec(rodrigues(v)), v, atol=1e-6)


def _phys(pos=np.zeros(3), R=np.eye(3)):
    return {"eef_pos": pos, "eef_mat": R, "grip_qpos": np.zeros(2), "obj_pos": np.zeros((2, 3)),
            "obj_mat": np.stack([np.eye(3)] * 2), "obj_names": OBJS}


def test_relative_rotation_is_frame_consistent():
    """Same world-frame increment on two different base orientations -> same effect
    (axis-angle subtraction would not give this)."""
    inc = np.array([0.01, -0.02, 0.005])
    for base in (np.zeros(3), np.array([0.3, 2.5, -1.0])):
        R0 = rodrigues(base)
        np.testing.assert_allclose(fx.effect(_phys(R=rodrigues(inc) @ R0), _phys(R=R0))[3:6], inc, atol=1e-9)


def test_layout_separates_task_objects():
    g = LAYOUT["groups"]
    assert LAYOUT["task_objects"] == ["basket_1"] and len(g["obj_pos_task"]) == 3 and len(g["obj_pos_other"]) == 3
    assert set(g["full"]) == set(g["eef"] + g["grip"] + g["obj_pos_task"] + g["obj_rot_task"])


class LinTask:
    """6-dof point with a first-order command filter v <- alpha v + (1-alpha) g(x) f(a); pos += 0.02 v[:3];
    world-frame rotation by 0.1 v[3:]. alpha=0: pure integrator. f = identity unless `quad`."""

    ACTION_LOW, ACTION_HIGH = -np.ones(7), np.ones(7)

    def __init__(self, alpha=0.0, gain_fn=None, quad=False):
        self.alpha, self.gain_fn, self.quad = alpha, gain_fn or (lambda x: np.ones(6)), quad
        self.pos, self.R, self.v, self.success = np.zeros(3), np.eye(3), np.zeros(6), False

    def step(self, a):
        a = np.asarray(a, float)[:6]
        a = a ** 2 if self.quad else a
        self.v = self.alpha * self.v + (1 - self.alpha) * self.gain_fn(self.pos) * a
        self.pos = self.pos + 0.02 * self.v[:3]
        self.R = rodrigues(0.1 * self.v[3:]) @ self.R
        return {}, False

    def snapshot(self):
        return (self.pos.copy(), self.R.copy(), self.v.copy())

    def restore(self, s):
        self.pos, self.R, self.v = s[0].copy(), s[1].copy(), s[2].copy()


def measure(t):
    return _phys(t.pos.copy(), t.R.copy())


def jac(task, start=None, eta=1e-3, a0=None):
    if start is not None:
        task.pos = np.asarray(start, float)
    snap = task.snapshot()
    if a0 is None:
        a0 = np.zeros((C, 7))
        a0[:, :3] = 0.1  # translation only: a rotating nominal chunk makes the rotation Jacobian non-diagonal (correctly)
    restore = lambda t, s: t.restore(s)  # noqa: E731
    nominal, _ = fx.execute(task, snap, a0, restore, measure)
    return fx.fd_jacobian(task, snap, a0, list(range(ND)), eta, restore, measure, task.ACTION_LOW, task.ACTION_HIGH, nominal)


def test_integrator_jacobian_and_bases_match_dct():
    J, info = jac(LinTask(alpha=0.0))
    assert info["causality_max"] == 0.0 and info["n_one_sided"] == 0
    S = fx.fit_integrator([J], ND)
    np.testing.assert_allclose(S[:3, :3], 0.02 * np.eye(3), atol=1e-6)
    np.testing.assert_allclose(S[3:6, 3:6], 0.1 * np.eye(3), atol=1e-4)
    assert fx.rel_error(fx.integrator_J(S, C), J, LAYOUT["groups"]["eef"], C - 1) < 1e-3
    sc = fx.group_scales([J], LAYOUT)
    b = fx.oer_basis([fx.normalise(J, sc)], LAYOUT["groups"]["eef"], C, (2, 4, 6))
    d = fx.dct_oer(C, ND)
    assert b["rank_end"] == 6 and b["rank_shape"] == 6
    assert fx.overlap(b["end"], d["end"]) > 0.999
    assert fx.overlap(b["shape"], d["shape"]) > 0.9
    assert np.abs(b["end"].T @ b["shape"]).max() < 1e-8


def test_lag_model_beats_integrator_on_filtered_dynamics():
    Js = [jac(LinTask(alpha=0.6))[0] for _ in range(2)]
    rows = LAYOUT["groups"]["eef_pos"]
    e_int = fx.rel_error(fx.integrator_J(fx.fit_integrator(Js, ND), C), Js[0], rows, 3)
    e_lag = fx.rel_error(fx.lag_J(fx.fit_lag(Js, ND)), Js[0], rows, 3)
    assert e_lag < 1e-3 < e_int


def test_state_dependent_gain_shows_up_only_locally():
    gain = lambda x: np.array([1, 1, 1, 1, 1, 1.0]) if x[0] < 0.5 else np.array([1, 0.1, 1, 1, 1, 1.0])  # noqa: E731
    Ja, Jb = jac(LinTask(gain_fn=gain), start=[0, 0, 0])[0], jac(LinTask(gain_fn=gain), start=[1, 0, 0])[0]
    glob = np.nanmean(np.stack([Ja, Jb]), axis=0)
    rows = LAYOUT["groups"]["eef_pos"]
    assert fx.rel_error(glob, Jb, rows, C - 1) > 0.1 and fx.rel_error(Jb, Jb, rows, C - 1) == 0


def test_clipped_step_stays_symmetric_on_nonlinear_dynamics():
    """Review: a0=0.95, eta=0.1, f(a)=a^2: the asymmetric (+0.05, -0.10) difference gave 1.85 instead of 1.90."""
    a0 = np.zeros((C, 7))
    a0[0, 0] = 0.95
    J, info = jac(LinTask(quad=True), eta=0.1, a0=a0)
    assert info["kind"][0] == 0 and np.isclose(info["h"][0], 0.05)  # symmetric, step shrunk to the room
    np.testing.assert_allclose(J[C - 1][0, 0] / 0.02, 1.90, rtol=1e-9)  # d(a^2)/da at 0.95, exact for a quadratic


def test_one_sided_difference_at_the_bound_is_flagged():
    a0 = np.zeros((C, 7))
    a0[0, 0] = 1.0
    J, info = jac(LinTask(), eta=0.1, a0=a0)
    assert info["kind"][0] == -1 and np.isclose(info["h"][0], 0.1) and info["n_one_sided"] == 1
    np.testing.assert_allclose(J[C - 1][0, 0], 0.02, rtol=1e-9)
    assert info["causality_max"] == 0.0  # actions after step s never change the state at s


def test_zero_response_errors_are_not_nan():
    J = np.zeros((C, 3, 48))
    P = np.zeros_like(J)
    P[-1, 0, 0] = 1.0
    assert fx.rel_error(P, J, [0, 1, 2], C - 1) == float("inf") and fx.rel_error(J, J, [0, 1, 2], C - 1) == 0.0
    e = fx.pooled_errors(P, [J, J], [0, 1, 2], C - 1)
    assert e["pooled_rel"] == float("inf") and e["abs_rms"] > 0


def test_degenerate_jacobian_gives_reduced_rank_and_orthogonal_bases():
    zero = np.zeros((C, 6, 48))
    b = fx.oer_basis([zero], list(range(6)), C, (2, 4, 6))
    assert b["rank_end"] == 0 and b["rank_shape"] == 0 and b["end"].shape == (48, 0)
    one = np.zeros((C, 6, 48))
    one[:, 0, 0] = 1.0  # a single responsive direction
    b = fx.oer_basis([one], list(range(6)), C, (2, 4, 6))
    assert b["rank_end"] == 1 and b["rank_shape"] == 0
    assert fx.overlap(b["end"], b["shape"]) is None
    J, _ = jac(LinTask())
    b = fx.oer_basis([J], LAYOUT["groups"]["eef"], C, (2, 4, 6))
    assert np.abs(b["end"].T @ b["shape"]).max() < 1e-10


def test_projected_direction_is_basis_rotation_invariant():
    rng = np.random.default_rng(0)
    B = np.linalg.qr(rng.standard_normal((48, 6)))[0]
    Rot = np.linalg.qr(rng.standard_normal((6, 6)))[0]
    v = rng.standard_normal(48)
    np.testing.assert_allclose(fx.project_direction(B, v), fx.project_direction(B @ Rot, v), atol=1e-12)
    assert fx.project_direction(np.zeros((48, 0)), v) is None


def _load_script():
    p = Path(__file__).resolve().parents[1] / "scripts" / "jacobian_diag.py"
    spec = importlib.util.spec_from_file_location("jacobian_diag", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _synthetic_out(tmp_path, jd, repeat=(0.0, 0.0, 0.0, 0.0), drop=0, stray=False):
    Js = np.stack([jac(LinTask(alpha=al))[0] for al in (0.0, 0.2, 0.0, 0.2)])
    np.savez_compressed(tmp_path / "jacobians.npz", J=Js, J_half=Js[:1], half_idx=np.array([0]), a0=np.zeros((4, C, 7)))
    anchors = [{"ep": i, "split": "fit" if i % 2 == 0 else "heldout", "repeat_max": repeat[i], "restore_vs_orig": repeat[i],
                "restore_repeat": 0.0, "fd_causality_max": 0.0, "fd_one_sided": 0, "fd_unmeasured": 0,
                "fd_branches_with_success": 0, "fd_branches": 96, "nominal_success_step": -1,
                "phase": {"phase": "free" if i < 2 else "contact", "errors": []}, "elapsed": 8 * i} for i in range(4)]
    (tmp_path / "anchors.json").write_text(json.dumps({"anchors": anchors, "layout": LAYOUT, "C": C, "dims": list(range(ND))}))
    ho = [i for i in (1, 3) if repeat[i] <= 1e-6]
    sc = fx.group_scales([Js[i] for i in (0, 2)], LAYOUT)
    vmeta = {"fit_idx": [0, 2], "heldout_idx": ho, "scales": sc.tolist(), "repeat_tol": 1e-6, "methods": list(jd.METHODS),
             "val_dirs": 1, "signs": [1, -1], "val_rms": 0.05, "complete": True}
    (tmp_path / "validation_meta.json").write_text(json.dumps(vmeta))
    recs = []
    for i in ho + ([1] if stray else []):
        for m in jd.METHODS:
            for s in (1, -1):
                recs.append({"anchor": i, "ep": i, "phase": anchors[i]["phase"]["phase"], "method": m, "dir": 0, "sign": s,
                             "rank_shape": 6, "rms_planned": 0.05, "rms_actual": 0.05, "comparable": True, "success_step": -1,
                             "endpoint_drift": 0.1 if m in jd.LOCAL else 0.2, "shape_change": 1.0,
                             "drift_after_continuation": 0.3, "obj_pos_task_absmax_end": 0.0, "obj_pos_task_absmax_cont": 0.0,
                             "obj_pos_other_absmax_end": 0.0, "obj_pos_other_absmax_cont": 0.0, "grip_absmax_end": 0.0,
                             "eef_pos_m_end": 0.001, "changed_end": {"grasp": [], "contact": []},
                             "changed_cont": {"grasp": [], "contact": ["basket_1"] if m == "dct" else []}})
    with open(tmp_path / "validation.jsonl", "w") as f:
        for r in recs[drop:]:
            f.write(json.dumps(r) + "\n")


def test_offline_analysis_complete(tmp_path):
    jd = _load_script()
    _synthetic_out(tmp_path, jd)
    r = jd.analyze(tmp_path, 1e-6)
    v = r["validation"]
    assert v["complete"] and v["n_missing"] == 0 and set(v["by_phase"]) == {"free", "contact"}
    assert r["model_bases"]["integrator"]["end_vs_dct"] > 0.999
    assert r["eta_half_rel_diff"]["eef_pos"]["s8"] == 0.0
    assert v["overall"]["paired"]["local_eef_vs_dct"]["local_dominates"] == 2
    assert v["overall"]["table"]["dct"]["frac_contact_changed_cont"] == 1.0
    assert r["heldout_error"]["global_mean"]["obj_pos_task"]["s8"]["pooled_rel"] == 0.0  # static objects: 0, not NaN
    assert (tmp_path / "analysis.md").read_text().startswith("# Jacobian diagnostic")


def test_offline_analysis_rejects_excluded_anchor_records(tmp_path):
    """Review: a held-out anchor that failed restore still entered the oracle comparison."""
    jd = _load_script()
    _synthetic_out(tmp_path, jd, repeat=(0.0, 1.0, 0.0, 0.0), stray=True)
    v = jd.analyze(tmp_path, 1e-6)["validation"]
    assert v["stray_anchors"] == [1] and not v["complete"]
    assert all(x["n_anchors"] <= 1 for x in v["overall"]["table"].values())


def test_offline_analysis_flags_incomplete_validation(tmp_path):
    jd = _load_script()
    _synthetic_out(tmp_path, jd, drop=1)
    r = jd.analyze(tmp_path, 1e-6)
    assert not r["validation"]["complete"] and r["validation"]["n_missing"] == 1
    assert "INCOMPLETE VALIDATION" in (tmp_path / "analysis.md").read_text()


def test_flag_changes_ignore_failed_checks():
    a = {"grasp": {"x": None, "y": True}, "contact": {"x": False, "y": True}}
    b = {"grasp": {"x": False, "y": False}, "contact": {"x": False, "y": None}}
    c = fx.flag_changes(a, b)
    assert c["grasp"] == ["y"] and c["contact"] == [] and c["unavailable"] == ["contact:y", "grasp:x"]


def test_pooled_errors_without_measurements_is_no_estimate():
    J = np.full((C, 3, 48), np.nan)
    assert fx.pooled_errors(np.zeros((C, 3, 48)), [J], [0, 1, 2], C - 1)["pooled_rel"] is None


def _recs(tmp_path):
    return [json.loads(l) for l in open(tmp_path / "validation.jsonl")]


def _write(tmp_path, recs):
    with open(tmp_path / "validation.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def test_paired_comparison_uses_common_draws_only(tmp_path):
    """Review counterexample: identical effects on common draws, some draws excluded only for the oracle."""
    jd = _load_script()
    _synthetic_out(tmp_path, jd)
    vm = json.loads((tmp_path / "validation_meta.json").read_text())
    vm["val_dirs"] = 2
    (tmp_path / "validation_meta.json").write_text(json.dumps(vm))
    recs = []
    for r in _recs(tmp_path):
        for d, (e, s_) in enumerate(((0.2, 1.0), (1.0, 0.1))):  # draw 1 is bad for everybody
            recs.append(dict(r, dir=d, endpoint_drift=e, shape_change=s_))
    for r in recs:
        if r["method"] in jd.LOCAL and r["dir"] == 1:
            r["comparable"] = False  # clipped only for the oracle
    _write(tmp_path, recs)
    p = jd.analyze(tmp_path, 1e-6)["validation"]["overall"]["paired"]["local_eef_vs_dct"]
    assert p["median_dlog_endpoint"] == 0.0 and p["median_dlog_shape"] == 0.0
    assert p["local_dominates"] == 0 and p["ties"] == p["n_anchors"] == 2


def test_duplicates_and_insufficient_signal(tmp_path):
    jd = _load_script()
    _synthetic_out(tmp_path, jd)
    recs = _recs(tmp_path)
    _write(tmp_path, recs + recs[:1])
    v = jd.analyze(tmp_path, 1e-6)["validation"]
    assert not v["complete"] and v["duplicates"] == 1
    _write(tmp_path, [dict(r, comparable=False) for r in recs])
    v = jd.analyze(tmp_path, 1e-6)["validation"]
    assert v["complete"] and not v["sufficient"] and v["n_used"] == 0
    assert "INSUFFICIENT SIGNAL" in (tmp_path / "analysis.md").read_text()


def test_no_valid_heldout_anchor_is_insufficient(tmp_path):
    jd = _load_script()
    _synthetic_out(tmp_path, jd, repeat=(0.0, 1.0, 0.0, 1.0))
    v = jd.analyze(tmp_path, 1e-6)["validation"]
    assert not v["sufficient"]
