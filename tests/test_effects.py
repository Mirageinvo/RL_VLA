"""Jacobian-diagnostic math on synthetic systems with a known answer (no simulator)."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from flow_residual_rl import effects as fx

C, ND = 8, 6


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


def test_relative_rotation_is_frame_consistent():
    """Same world-frame increment applied to two different base orientations -> same effect,
    which axis-angle subtraction would not give."""
    inc = np.array([0.01, -0.02, 0.005])
    for base in (np.zeros(3), np.array([0.3, 2.5, -1.0])):
        R0 = rodrigues(base)
        p = {"eef_pos": np.zeros(3), "eef_mat": rodrigues(inc) @ R0, "grip_qpos": np.zeros(2),
             "obj_pos": np.zeros((0, 3)), "obj_mat": np.zeros((0, 3, 3))}
        q = dict(p, eef_mat=R0)
        np.testing.assert_allclose(fx.effect(p, q)[3:6], inc, atol=1e-9)


class LinTask:
    """6-dof point with a first-order command filter v <- alpha v + (1-alpha) g(x) a; pos += 0.02 v[:3];
    world-frame rotation by 0.1 v[3:]. alpha=0: pure integrator. gain_fn(x) makes it state dependent."""

    ACTION_LOW, ACTION_HIGH = -np.ones(7), np.ones(7)

    def __init__(self, alpha=0.0, gain_fn=None):
        self.alpha, self.gain_fn = alpha, gain_fn or (lambda x: np.ones(6))
        self.pos, self.R, self.v = np.zeros(3), np.eye(3), np.zeros(6)

    def step(self, a):
        a = np.asarray(a, float)[:6]
        self.v = self.alpha * self.v + (1 - self.alpha) * self.gain_fn(self.pos) * a
        self.pos = self.pos + 0.02 * self.v[:3]
        self.R = rodrigues(0.1 * self.v[3:]) @ self.R
        return {}, False

    def snapshot(self):
        return (self.pos.copy(), self.R.copy(), self.v.copy())

    def restore(self, s):
        self.pos, self.R, self.v = s[0].copy(), s[1].copy(), s[2].copy()


def measure(t):
    return {"eef_pos": t.pos.copy(), "eef_mat": t.R.copy(), "grip_qpos": np.zeros(2),
            "obj_pos": np.zeros((0, 3)), "obj_mat": np.zeros((0, 3, 3))}


def jac(task, start=None, eta=1e-3):
    if start is not None:
        task.pos = np.asarray(start, float)
    snap = task.snapshot()
    a0 = np.zeros((C, 7))
    a0[:, :3] = 0.1  # translation only: a rotating nominal chunk makes the rotation Jacobian non-diagonal (correctly)
    J, info = fx.fd_jacobian(task, snap, a0, list(range(ND)), eta, lambda t, s: t.restore(s), measure,
                             task.ACTION_LOW, task.ACTION_HIGH)
    return J


LAYOUT = fx.feature_layout(2, [])


def test_integrator_jacobian_and_bases_match_dct():
    task = LinTask(alpha=0.0)
    J = jac(task)
    S = fx.fit_integrator([J], ND)
    np.testing.assert_allclose(S[:3, :3], 0.02 * np.eye(3), atol=1e-6)          # position gain
    np.testing.assert_allclose(S[3:6, 3:6], 0.1 * np.eye(3), atol=1e-4)         # rotation gain (small angles)
    assert fx.rel_error(fx.integrator_J(S, C), J, LAYOUT["groups"]["eef"], C - 1) < 1e-3
    sc = fx.group_scales([J], LAYOUT)
    b = fx.oer_basis([fx.normalise(J, sc)], LAYOUT["groups"]["eef"], C, (2, 4, 6))
    d = fx.dct_oer(C, ND)
    assert fx.overlap(b["end"], d["end"]) > 0.999                      # endpoint = constant mode per coordinate
    assert fx.overlap(b["shape"], d["shape"]) > 0.9                    # shape ~ first non-constant DCT mode
    assert np.abs(b["end"].T @ b["shape"]).max() < 1e-8                # shape lies in the endpoint null space
    assert np.linalg.norm(np.nan_to_num(fx.normalise(J, sc)[C - 1][LAYOUT["groups"]["eef"]]) @ b["shape"]) < 1e-6


def test_lag_model_beats_integrator_on_filtered_dynamics():
    Js = [jac(LinTask(alpha=0.6)) for _ in range(2)]
    rows = LAYOUT["groups"]["eef_pos"]
    e_int = fx.rel_error(fx.integrator_J(fx.fit_integrator(Js, ND), C), Js[0], rows, 3)
    e_lag = fx.rel_error(fx.lag_J(fx.fit_lag(Js, ND)), Js[0], rows, 3)
    assert e_lag < 1e-3 < e_int


def test_state_dependent_gain_shows_up_only_locally():
    gain = lambda x: np.array([1, 1, 1, 1, 1, 1.0]) if x[0] < 0.5 else np.array([1, 0.1, 1, 1, 1, 1.0])
    Ja, Jb = jac(LinTask(gain_fn=gain), start=[0, 0, 0]), jac(LinTask(gain_fn=gain), start=[1, 0, 0])
    glob = np.nanmean(np.stack([Ja, Jb]), axis=0)
    rows = LAYOUT["groups"]["eef_pos"]
    assert fx.rel_error(glob, Jb, rows, C - 1) > 0.1 and fx.rel_error(Jb, Jb, rows, C - 1) == 0


def test_clipped_coordinate_uses_actual_change():
    task = LinTask()
    snap = task.snapshot()
    a0 = np.zeros((C, 7))
    a0[0, 0] = 0.95  # +eta clipped to 1.0
    J, info = fx.fd_jacobian(task, snap, a0, list(range(ND)), 0.1, lambda t, s: t.restore(s), measure,
                             task.ACTION_LOW, task.ACTION_HIGH)
    assert info["n_clipped"] == 1 and np.isclose(info["actual"][0, 0], 0.05)
    np.testing.assert_allclose(J[C - 1][0, 0], 0.02, rtol=1e-6)  # still the true gain


def _load_script():
    p = Path(__file__).resolve().parents[1] / "scripts" / "jacobian_diag.py"
    spec = importlib.util.spec_from_file_location("jacobian_diag", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_offline_analysis_end_to_end(tmp_path):
    """analyze() on synthetic integrator Jacobians + validation records: runs and reports DCT overlap."""
    jd = _load_script()
    Js = np.stack([jac(LinTask(alpha=al)) for al in (0.0, 0.2, 0.0, 0.2)])
    np.savez_compressed(tmp_path / "jacobians.npz", J=Js, J_half=Js[:1], half_idx=np.array([0]), a0=np.zeros((4, C, 7)))
    anchors = [{"ep": i, "split": "fit" if i % 2 == 0 else "heldout", "repeat_max": 0.0,
                "phase": {"phase": "free" if i < 2 else "near"}, "elapsed": 8 * i} for i in range(4)]
    (tmp_path / "anchors.json").write_text(json.dumps({"anchors": anchors, "layout": LAYOUT, "C": C,
                                                       "dims": list(range(ND))}))
    with open(tmp_path / "validation.jsonl", "w") as f:
        for i in (1, 3):
            for m in jd.METHODS:
                f.write(json.dumps({"anchor": i, "phase": "x", "method": m, "dir": 0, "sign": 1, "rms_actual": 0.05,
                                    "endpoint_drift": 0.1 if m == "local" else 0.2, "shape_change": 1.0,
                                    "drift_after_continuation": 0.3, "obj_drift_m_end": 0.0, "obj_drift_m_cont": 0.0,
                                    "eef_pos_drift_m_end": 0.001}) + "\n")
    r = jd.analyze(tmp_path, 1e-6)
    assert r["model_basis_overlap"]["integrator"]["end_vs_dct"] > 0.999
    assert r["eta_half_rel_diff"]["eef_pos"]["s8"] == 0.0
    assert r["validation"]["local_vs"]["dct"]["local_better"] == 2
    assert (tmp_path / "analysis.md").read_text().startswith("# Jacobian diagnostic")
