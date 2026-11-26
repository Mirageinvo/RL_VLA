"""scripts/tail_diag.py on a fake simulator / policy (no LIBERO, no GR00T)."""
import json
import pickle
import sys

import numpy as np

from flow_residual_rl import effects as fx

from test_continuation import C, ND, FakeCounter, FakeProvider, FakeTask, _load, _measure
from test_effects import LAYOUT

GAIN = lambda x: np.ones(6) * (1.0 + 5.0 * x[0])  # noqa: E731  state-dependent gain (nonlinear dynamics)


class SuccessTask(FakeTask):
    """success from the 3rd step after a restore on (inside the corrected chunk)."""

    def restore(self, s):
        super().restore(s)
        self.n, self.success = 0, False

    def step(self, a):
        out = super().step(a)
        self.n = getattr(self, "n", 0) + 1
        self.success = self.n >= 3
        return out


def _setup(tmp_path, monkeypatch, verified=True, reactive=False, task_cls=FakeTask, a0_value=0.1):
    td = _load("tail_diag")
    cd = sys.modules["continuation_diag"] if "continuation_diag" in sys.modules else _load("continuation_diag")
    import flow_residual_rl.budget as budget
    import flow_residual_rl.pipeline as pipeline
    import flow_residual_rl.restore as rs

    task, prov = task_cls(alpha=0.6, gain_fn=GAIN), FakeProvider(reactive)
    restore = lambda t, s: t.restore(s)  # noqa: E731
    src, cont, out = tmp_path / "jacobian_task0", tmp_path / "continuation_task0", tmp_path / "tail_task0"
    for d in (src, cont, out):
        d.mkdir(parents=True)
    anchors, Js, a0s, snaps = [], [], [], []
    for i in range(6):
        task.restore((np.array([0.01 * i, 0, 0]), np.eye(3), np.zeros(6)))
        snap = task.snapshot()
        a0 = np.full((C, 7), a0_value)
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
        task_id, val_dirs, val_rms, seed, step_limit, repro_tol = 0, 2, 0.05, 23, 100000, 1e-5

    return td, td.collect({"restore_mode": "snapshot"}, A, src, cont / "anchor_snaps.pkl", out), out


def test_tail_diag_end_to_end(tmp_path, monkeypatch):
    td, code, out = _setup(tmp_path, monkeypatch)
    assert code == 0
    r = td.analyze(out)
    assert r["complete"] and r["ranks_ok"] and r["chunk_repeat_max"] == 0.0 and r["exit_code"] == 0
    assert r["n_used"] == 3 * 4 * 2 * 2 and r["n_common_branches"] == 3 * 2 * 2
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
    assert dec["valid"] and set(dec["checks"]) == {"1_drift_2C", "2_shape", "3_episodes", "4_continuation"}
    assert "**decision:" in (out / "analysis.md").read_text()


def test_unverified_snapshots_are_refused(tmp_path, monkeypatch):
    _, code, out = _setup(tmp_path, monkeypatch, verified=False)
    assert code == 4 and not (out / "tail.jsonl").exists()


def test_missing_record_is_incomplete(tmp_path, monkeypatch):
    td, code, out = _setup(tmp_path, monkeypatch)
    lines = (out / "tail.jsonl").read_text().splitlines()
    (out / "tail.jsonl").write_text("\n".join(lines[1:]) + "\n")
    r = td.analyze(out)
    assert not r["complete"] and "INVALID RUN" in (out / "analysis.md").read_text()


def _rewrite(out, fn):
    recs = [json.loads(x) for x in (out / "tail.jsonl").read_text().splitlines()]
    recs = fn(recs)
    (out / "tail.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")


def test_tables_and_control_use_only_common_branches(tmp_path, monkeypatch):
    """Review: the control was picked on each method's own branches; a branch unusable for one
    method must leave every method's statistics."""
    td, _, out = _setup(tmp_path, monkeypatch)

    def drop(recs):
        for r in recs:
            if r["method"] == "global_tail" and r["anchor"] == 1 and r["dir"] == 0 and r["sign"] == 1:
                r["comparable"] = False
            if r["method"] == "dct" and r["anchor"] == 1 and r["dir"] == 0 and r["sign"] == 1:
                r["replanned"]["drift_2C"] = 1e-9  # would make dct the control if it were counted
        return recs

    _rewrite(out, drop)
    r = td.analyze(out)
    assert r["n_common_branches"] == 3 * 2 * 2 - 1
    t = r["groups"]["all"]["tables"]["replanned"]
    assert len({t[m]["n_branches"] for m in t}) == 1
    assert t["dct"]["drift_2C"] > 1e-6  # the excluded dct branch does not enter


def test_no_decision_for_invalid_runs(tmp_path, monkeypatch):
    td, _, out = _setup(tmp_path, monkeypatch)
    _rewrite(out, lambda recs: recs[1:])  # missing record
    r = td.analyze(out)
    assert r["exit_code"] == 2 and r["decision"]["decision"] is None and "INVALID RUN" in (out / "analysis.md").read_text()

    td2, _, out2 = _setup(tmp_path / "b", monkeypatch)
    _rewrite(out2, lambda recs: [dict(r, chunk_repeat=0.05) if "chunk_repeat" in r else r for r in recs])
    r = td2.analyze(out2)
    assert r["exit_code"] == 4 and r["decision"]["decision"] is None

    td3, _, out3 = _setup(tmp_path / "c", monkeypatch)
    m3 = json.loads((out3 / "tail_meta.json").read_text())
    m3["rank_ok"] = False
    (out3 / "tail_meta.json").write_text(json.dumps(m3))
    r = td3.analyze(out3)
    assert r["exit_code"] == 3 and r["decision"]["decision"] is None

    td4, _, out4 = _setup(tmp_path / "d", monkeypatch)
    _rewrite(out4, lambda recs: [dict(r, comparable=False) if r["ep"] != 1 and "skipped" not in r else r for r in recs])
    r = td4.analyze(out4)
    assert r["exit_code"] == 3 and r["decision"]["decision"] is None and r["decision"]["common_episodes"] == [1]


def test_common_scale_avoids_clipping():
    a0 = np.array([[0.98, 0.0], [0.0, -0.99]])
    d1, d2 = np.array([[0.1, 0.0], [0.0, 0.0]]), np.array([[0.0, 0.0], [0.0, 0.05]])
    lam = td_mod().common_scale(a0, [d1, d2], -np.ones(2), np.ones(2))
    assert np.isclose(lam, 0.2)  # 0.98 + 0.2 * 0.1 = 1.0, and -0.99 - 0.2 * 0.05 = -1.0
    for d in (d1, d2):
        for x in (a0 + lam * d, a0 - lam * d):
            assert np.all(np.abs(x) <= 1 + 1e-12)


def td_mod():
    return sys.modules["tail_diag"] if "tail_diag" in sys.modules else _load("tail_diag")


def test_sets_below_common_scale_are_excluded_and_sums_stay_zero(tmp_path, monkeypatch):
    _, _, out = _setup(tmp_path, monkeypatch, a0_value=0.97)  # little room: most sets need scale < 0.95
    recs = [json.loads(x) for x in (out / "tail.jsonl").read_text().splitlines()]
    sk = [r for r in recs if r.get("skipped") == "common scale below threshold"]
    ok = [r for r in recs if "skipped" not in r]
    assert sk and all(r["scale"] < 0.95 for r in sk)
    by_set = {}
    for r in recs:
        by_set.setdefault((r["anchor"], r["dir"]), set()).add("skipped" in r)
    assert all(len(v) == 1 for v in by_set.values())  # a set is used or excluded as a whole
    for r in ok:
        assert abs(r["rms_actual"] - 0.05 * r["scale"]) < 1e-9
        assert r["exec_sum_frac"] < 1e-9  # no clipping: every method keeps its zero command sum


def test_success_inside_the_corrected_chunk_is_recorded(tmp_path, monkeypatch):
    _, _, out = _setup(tmp_path, monkeypatch, task_cls=SuccessTask)
    recs = [json.loads(x) for x in (out / "tail.jsonl").read_text().splitlines() if "skipped" not in x]
    assert recs and all(r["fixed"]["success_step"] == 3 and r["replanned"]["success_step"] == 3 for r in recs)
    assert all(r["nominal_success_step"] == 3 for r in recs)
