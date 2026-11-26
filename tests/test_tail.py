"""scripts/tail_diag.py on a fake simulator / policy (no LIBERO, no GR00T)."""
import json
import pickle
import sys

import numpy as np

from flow_residual_rl import effects as fx

from test_continuation import C, ND, FakeCounter, FakeProvider, FakeTask, _load, _measure
from test_effects import LAYOUT

GAIN = lambda x: np.ones(6) * (1.0 + 5.0 * x[0])  # noqa: E731  state-dependent gain (nonlinear dynamics)


def _setup(tmp_path, monkeypatch, verified=True, reactive=False):
    td = _load("tail_diag")
    cd = sys.modules["continuation_diag"] if "continuation_diag" in sys.modules else _load("continuation_diag")
    import flow_residual_rl.budget as budget
    import flow_residual_rl.pipeline as pipeline
    import flow_residual_rl.restore as rs

    task, prov = FakeTask(alpha=0.6, gain_fn=GAIN), FakeProvider(reactive)
    restore = lambda t, s: t.restore(s)  # noqa: E731
    src, cont, out = tmp_path / "jacobian_task0", tmp_path / "continuation_task0", tmp_path / "tail_task0"
    for d in (src, cont, out):
        d.mkdir()
    anchors, Js, a0s, snaps = [], [], [], []
    for i in range(6):
        task.restore((np.array([0.01 * i, 0, 0]), np.eye(3), np.zeros(6)))
        snap = task.snapshot()
        a0 = np.full((C, 7), 0.1)
        nominal, _ = fx.execute(task, snap, a0, restore, _measure)
        J, _ = fx.fd_jacobian(task, snap, a0, list(range(ND)), 1e-3, restore, _measure, task.ACTION_LOW,
                              task.ACTION_HIGH, nominal)
        b = {"ep": i, "sid": 10 + i, "k": 0, "elapsed": 8 * i, "split": "fit" if i % 2 == 0 else "heldout",
             "phase": {"phase": "free" if i < 4 else "contact"}}
        anchors.append(b)
        Js.append(J)
        a0s.append(a0)
        snaps.append(dict({k: b[k] for k in ("ep", "sid", "k", "elapsed", "split")}, phase=b["phase"]["phase"],
                          snap=snap, a0=a0))
    sc = fx.group_scales([Js[i] for i in (0, 2, 4)], LAYOUT)
    (src / "anchors.json").write_text(json.dumps({"anchors": anchors, "layout": LAYOUT, "C": C, "dims": list(range(ND))}))
    (src / "validation_meta.json").write_text(json.dumps({"fit_idx": [0, 2, 4], "heldout_idx": [1, 3, 5],
                                                          "scales": sc.tolist(), "complete": True}))
    np.savez_compressed(src / "jacobians.npz", J=np.stack(Js), a0=np.stack(a0s))
    with open(cont / "anchor_snaps.pkl", "wb") as f:
        pickle.dump({"verified": verified, "mode": "reuse", "anchors": snaps}, f)
    monkeypatch.setattr(cd, "measure", _measure)
    monkeypatch.setattr(fx, "interaction_flags", lambda t: {"grasp": {"basket_1": False}, "contact": {"basket_1": False},
                                                             "errors": []})
    monkeypatch.setattr(budget, "StepCounter", FakeCounter)
    monkeypatch.setattr(pipeline, "make_provider", lambda cfg, counter: prov)
    monkeypatch.setattr(pipeline, "make_task", lambda cfg, tid, counter, p: task)
    monkeypatch.setattr(pipeline, "write_manifest", lambda *a, **k: None)
    monkeypatch.setattr(rs, "restore_to", lambda t, s, mode, bucket: t.restore(s))

    class A:
        task_id, val_dirs, val_rms, seed, step_limit = 0, 2, 0.05, 23, 100000

    return td, td.collect({"restore_mode": "snapshot"}, A, src, cont / "anchor_snaps.pkl", out), out


def test_tail_diag_end_to_end(tmp_path, monkeypatch):
    td, code, out = _setup(tmp_path, monkeypatch)
    assert code == 0
    r = td.analyze(out)
    assert r["complete"] and r["ranks_ok"] and r["chunk_repeat_max"] == 0.0
    assert r["n_used"] == 3 * 4 * 2 * 2
    t = r["groups"]["all"]["tables"]
    # local tail = the true constraints of the (linear-in-commands) anchor: no handoff deviation
    # (LinTask has the same temporal response at every state, so local == global here: only the
    # handoff property is tested, not a state-conditioned advantage)
    for m in ("lag_tail", "global_tail", "local_tail"):
        assert t["fixed"][m]["drift_2C"] < 0.1 * t["fixed"]["dct"]["drift_2C"]
    assert t["fixed"]["local_tail"]["exec_sum_frac"] < 1e-6 and t["fixed"]["dct"]["exec_sum_frac"] < 1e-6
    np.testing.assert_allclose(t["fixed"]["dct"]["drift_2C"], t["replanned"]["dct"]["drift_2C"])  # open-loop policy
    dec = json.loads((out / "decision.json").read_text())
    assert dec["decision"] in ("continue_state_conditioned", "stop_state_conditioned_oer")
    assert set(dec["checks"]) == {"1_drift_2C", "2_shape", "3_episodes", "4_continuation"}
    assert "**decision:" in (out / "analysis.md").read_text()


def test_unverified_snapshots_are_refused(tmp_path, monkeypatch):
    _, code, out = _setup(tmp_path, monkeypatch, verified=False)
    assert code == 4 and not (out / "tail.jsonl").exists()


def test_missing_record_is_incomplete(tmp_path, monkeypatch):
    td, code, out = _setup(tmp_path, monkeypatch)
    lines = (out / "tail.jsonl").read_text().splitlines()
    (out / "tail.jsonl").write_text("\n".join(lines[1:]) + "\n")
    r = td.analyze(out)
    assert not r["complete"] and "INCOMPLETE" in (out / "analysis.md").read_text()
