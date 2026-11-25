"""scripts/continuation_diag.py on a fake simulator / policy (no LIBERO, no GR00T)."""
import contextlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from flow_residual_rl import effects as fx

from test_effects import LAYOUT, LinTask, _phys

C, ND = 8, 6
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
METHODS = ["dct", "integrator", "lag", "global_gram", "local_eef", "local_full"]


def _load(name):
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    sys.modules[name] = m
    return m


class FakeTask(LinTask):
    task_id = 0

    def policy_observation(self):
        return self.pos.copy()

    def close(self):
        pass


class Prop:
    def __init__(self, a):
        self.a0_env = a


class FakeProvider:
    """reactive: the continuation chunk depends on the observed position (GR00T-like re-planning)."""

    def __init__(self, reactive):
        self.reactive = reactive

    def propose(self, obs, _ctx, seed):
        a = np.full((C, 7), 0.1)
        if self.reactive:
            a[:, :3] -= 5.0 * np.asarray(obs)[None]
        return Prop(a)


class FakeCounter:
    def __init__(self, *_):
        self.steps = {"diagnostic": 0}

    def set_limit(self, *_):
        pass

    @contextlib.contextmanager
    def charge(self, _):
        yield

    def save(self):
        pass


def _measure(t):
    p = _phys(t.pos.copy(), t.R.copy())
    p["qpos"], p["qvel"] = t.pos.copy(), t.v.copy()
    return p


def _anchors(task):
    out = []
    for i in range(4):
        task.restore((np.array([0.01 * i, 0, 0]), np.eye(3), np.zeros(6)))
        out.append({"ep": i, "sid": 10 + i, "k": 0, "elapsed": 8 * i, "split": "fit" if i % 2 == 0 else "heldout",
                    "snap": task.snapshot(),
                    "a0": np.full((C, 7), 0.1 + 0.01 * i), "phase": {"phase": "free" if i < 2 else "contact"}})
    return out


def _source(src, task, prov, a0_shift=0.0):
    """A finished jacobian_diag output whose re-planned branches were produced by the same fakes."""
    src.mkdir(parents=True)
    anchors = _anchors(task)
    restore = lambda t, s: t.restore(s)  # noqa: E731
    E, A, K, recs = [], [], [], []
    rng = np.random.default_rng(0)
    for i in (1, 3):
        b = anchors[i]

        def run(act):
            st, _ = fx.execute(task, b["snap"], act, restore, _measure)
            for x in prov.propose(task.policy_observation(), None, 0).a0_env:
                task.step(x)
                st.append(_measure(task))
            return st

        nom = run(b["a0"])
        for mi, m in enumerate(METHODS):
            for s in (1, -1):
                act = b["a0"].copy()
                act[:, :ND] += s * 0.05 * rng.standard_normal((C, ND))
                E.append(np.stack([fx.effect(u, v) for u, v in zip(run(act), nom)]))
                A.append(act)
                K.append([i, mi, 0, s])
                recs.append({"anchor": i, "method": m, "dir": 0, "sign": s, "comparable": True})
    meta = [{k: v for k, v in b.items() if k not in ("snap", "a0")} for b in anchors]
    (src / "anchors.json").write_text(json.dumps({"anchors": meta, "layout": LAYOUT, "C": C, "dims": list(range(ND)),
                                                  "args": {"pool_episodes": 4, "n_anchors": 4, "seed": 11}}))
    (src / "validation_meta.json").write_text(json.dumps(
        {"fit_idx": [0, 2], "heldout_idx": [1, 3], "scales": [1.0] * len(LAYOUT["names"]), "methods": METHODS,
         "val_dirs": 1, "signs": [1, -1], "complete": True}))
    np.savez_compressed(src / "jacobians.npz", a0=np.stack([b["a0"] for b in anchors]) + a0_shift)
    np.savez_compressed(src / "validation_raw.npz", effects=np.stack(E), actions=np.stack(A), key=np.array(K))
    (src / "validation.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")


def _run(tmp_path, monkeypatch, reactive, a0_shift=0.0, fresh=False, tamper=None, pool_shift=0):
    cd = _load("continuation_diag")
    jd = sys.modules["jacobian_diag"] if "jacobian_diag" in sys.modules else _load("jacobian_diag")
    import flow_residual_rl.budget as budget
    import flow_residual_rl.pipeline as pipeline
    import flow_residual_rl.restore as rs

    task, prov = FakeTask(alpha=0.6), FakeProvider(reactive)
    src, out = tmp_path / "jacobian_task0", tmp_path / "continuation_task0"
    _source(src, task, prov, a0_shift)
    if tamper:
        raw = dict(np.load(src / "validation_raw.npz"))
        tamper(raw["effects"])
        np.savez_compressed(src / "validation_raw.npz", **raw)

    def pool(*_a, **_k):
        out = _anchors(task)
        out[1]["elapsed"] += pool_shift
        return out

    out.mkdir()
    monkeypatch.setattr(jd, "collect_pool", pool)
    monkeypatch.setattr(jd, "select_anchors", lambda bounds, n, seed: bounds)
    monkeypatch.setattr(cd, "measure", _measure)
    monkeypatch.setattr(budget, "StepCounter", FakeCounter)
    monkeypatch.setattr(pipeline, "make_provider", lambda cfg, counter: prov)
    monkeypatch.setattr(pipeline, "make_task", lambda cfg, tid, counter, p: task)
    monkeypatch.setattr(pipeline, "task_splits", lambda cfg, t: {"train": [0]})
    monkeypatch.setattr(pipeline, "write_manifest", lambda *a, **k: None)
    monkeypatch.setattr(rs, "restore_to", lambda t, s, mode, bucket: t.restore(s))

    class A:
        methods, a0_tol, repro_tol, step_limit, task_id = ["dct", "lag", "global_gram", "local_eef"], 1e-4, 1e-5, 10000, 0

    A.fresh = fresh

    code = cd.collect({"restore_mode": "snapshot"}, A, src, out)
    return cd, code, src, out


def test_open_loop_policy_fixed_equals_replanned(tmp_path, monkeypatch):
    cd, code, src, out = _run(tmp_path, monkeypatch, reactive=False)
    assert code == 0
    rp = json.loads((out / "reproduction.json").read_text())
    assert rp["complete"] and rp["same_anchors"] and rp["a0_max_diff"] == 0.0
    assert all(v == 0.0 for v in rp["replanned_max_diff"].values())
    z = np.load(out / "continuation.npz")
    raw = np.load(src / "validation_raw.npz")
    assert len(z["raw_row"]) == 2 * 4 * 2  # 2 held-out anchors x 4 methods x 2 signs
    # policy ignores the observation -> replaying c0 is exactly what re-planning does
    np.testing.assert_allclose(z["e_fixed"], raw["effects"][z["raw_row"]], atol=1e-12)
    assert np.abs(z["dqvel_C"]).max() > 0  # different velocity at the handoff is recorded
    md = cd.analyze(src, out)
    assert "## phase all: 2 anchors" in md and "## phase contact" in md and "| local_eef |" in md


def test_reactive_policy_separates_protocols(tmp_path, monkeypatch):
    cd, code, src, out = _run(tmp_path, monkeypatch, reactive=True)
    assert code == 0
    z = np.load(out / "continuation.npz")
    raw = np.load(src / "validation_raw.npz")
    e_f, e_r = z["e_fixed"], raw["effects"][z["raw_row"]]
    np.testing.assert_allclose(e_f[:, :C], e_r[:, :C], atol=1e-12)  # same corrected chunk
    assert np.abs(e_f[:, C:] - e_r[:, C:]).max() > 1e-6  # continuation differs only through re-planning


def test_pool_not_reproduced_runs_nothing(tmp_path, monkeypatch):
    _, code, _, out = _run(tmp_path, monkeypatch, reactive=False, a0_shift=1e-3)
    assert code == 4
    assert not (out / "continuation.npz").exists()
    assert json.loads((out / "reproduction.json").read_text())["a0_max_diff"] == pytest.approx(1e-3)


def _snaps(out):
    import pickle

    with open(out / "anchor_snaps.pkl", "rb") as f:
        return pickle.load(f)


def test_reuse_run_is_verified_and_reports_paired_protocol_difference(tmp_path, monkeypatch):
    cd, code, src, out = _run(tmp_path, monkeypatch, reactive=True)
    rp = json.loads((out / "reproduction.json").read_text())
    assert code == 0 and rp["complete"] and rp["mode"] == "reuse"
    assert all(v == 0.0 for v in rp["fixed_first_C_max_diff"].values())
    assert _snaps(out)["verified"] is True
    md = cd.analyze(src, out)
    assert "re-planned vs fixed, per anchor d_i" in md and "| local_eef | 2 |" in md


def test_saved_replanned_continuation_mismatch_blocks(tmp_path, monkeypatch):
    """Review: a 0.05 mismatch of the saved re-planned branch was only logged (exit 0, complete)."""

    def tamper(e):
        e[:, C:] += 0.05

    _, code, _, out = _run(tmp_path, monkeypatch, reactive=True, tamper=tamper)
    rp = json.loads((out / "reproduction.json").read_text())
    assert code == 4 and rp["complete"] is False and "re-planned branch differs" in rp["failed"]
    assert _snaps(out)["verified"] is False


def test_corrected_chunk_mismatch_in_any_branch_blocks(tmp_path, monkeypatch):
    def tamper(e):
        e[12 + 2 * METHODS.index("local_eef") + 1, 2] += 1e-3  # anchor 3, local_eef, sign -1: not the checked re-planned one

    _, code, _, out = _run(tmp_path, monkeypatch, reactive=True, tamper=tamper)
    rp = json.loads((out / "reproduction.json").read_text())
    assert code == 4 and "corrected chunk differs" in rp["failed"]


def test_anchor_identity_includes_elapsed(tmp_path, monkeypatch):
    _, code, _, out = _run(tmp_path, monkeypatch, reactive=False, pool_shift=8)
    rp = json.loads((out / "reproduction.json").read_text())
    assert code == 4 and rp["same_anchors"] is False and not (out / "anchor_snaps.pkl").exists()


def test_fresh_mode_reruns_both_protocols(tmp_path, monkeypatch):
    def tamper(e):
        e += 0.05  # saved effects are not used in fresh mode

    cd, code, src, out = _run(tmp_path, monkeypatch, reactive=False, a0_shift=1e-3, fresh=True, tamper=tamper)
    rp = json.loads((out / "reproduction.json").read_text())
    assert code == 0 and rp["complete"] and rp["mode"] == "fresh"
    z = np.load(out / "continuation.npz")
    np.testing.assert_allclose(z["e_fixed"], z["e_replanned"], atol=1e-12)  # open-loop policy: protocols coincide
    assert _snaps(out)["verified"] is False
    assert "mode: fresh" in cd.analyze(src, out)
