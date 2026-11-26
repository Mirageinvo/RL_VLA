"""Fixed-basis RL pilot path on the CPU fakes (plumbing only, not an experiment)."""
import json
import sys

import numpy as np
import pytest
import yaml

from fakes import FakeProvider, FakeTask
from flow_residual_rl import effects as fx
from flow_residual_rl.basis import dct_basis
from flow_residual_rl.budget import StepCounter
from flow_residual_rl.config import splits

from test_continuation import _load
from test_effects import _load_script, _synthetic_out
from test_review_regressions import pilot_cfg


def _diag(tmp_path):
    d = tmp_path / "diag"
    d.mkdir()
    _synthetic_out(d, _load_script())
    return d


def test_tail12_basis(tmp_path):
    from flow_residual_rl.pipeline import basis_for_arm, task_dir
    from flow_residual_rl.tail_basis import build_tail12

    cfg = pilot_cfg(tmp_path)
    meta = build_tail12(_diag(tmp_path), task_dir(cfg, 3) / "basis" / "tail12")
    B = basis_for_arm(cfg, "tail12", 3)
    assert B.shape == (48, 12) and meta["orthonormality_err"] < 1e-5
    np.testing.assert_allclose(B[:, :6], dct_basis(8, 6, 1), atol=1e-6)  # mean-command channel, as in dct12
    assert np.abs(fx.sum_operator(8, 6) @ B[:, 6:]).max() < 1e-5  # shape part: zero command sum
    with pytest.raises(FileNotFoundError):
        basis_for_arm(cfg, "tail12", 8)  # never silently replaced by something else


def test_nominal_normalizer_and_validation_base_eval(tmp_path, monkeypatch):
    from flow_residual_rl import stages
    from flow_residual_rl.pipeline import load_normalizer

    cfg = pilot_cfg(tmp_path)
    tasks = []

    def mk(cfg_, tid, counter, prov):
        tasks.append(FakeTask(counter, episode_cap=64, goal=0.3, task_id=tid))
        return tasks[-1]

    monkeypatch.setattr(stages, "make_task", mk)
    with pytest.raises(FileNotFoundError):
        load_normalizer(cfg, 3)
    prov = FakeProvider(8, drift=0.3)
    d = stages.fit_nominal_normalizer(cfg, 3, prov, 3)
    assert d["n_episodes"] == 3 and d["n"] > 3 and d["cost"]["steps"]["obs_norm"] > 0
    norm = load_normalizer(cfg, 3)
    assert norm.mean.shape == (128 + 8 + 8 * 7 + 1,)
    n_tasks = len(tasks)
    assert stages.fit_nominal_normalizer(cfg, 3, prov, 3) == json.loads(
        (stages.task_dir(cfg, 3) / "obs_norm" / "obs_norm.json").read_text())  # cached
    assert len(tasks) == n_tasks
    # validation-split base evaluation: own cache, validation starts only
    counter = StepCounter(None)
    task = FakeTask(counter, episode_cap=64, goal=0.3, task_id=3)
    r = stages.eval_base(cfg, 3, prov, task, 4, counter, split="validation")
    sp = splits(cfg, task.n_init_states)
    recs = [json.loads(x) for x in (stages.base_eval_dir(cfg, 3, "validation") / "episodes.jsonl").read_text().splitlines()]
    assert r["split"] == "validation" and {x["init_state_id"] for x in recs} <= set(sp["validation"])
    assert not (stages.base_eval_dir(cfg, 3, "evaluation") / "episodes.jsonl").exists()


def test_paired_counts_and_schedule_check():
    rfp = _load("run_fixed_pilot")
    base = [{"init_state_id": i, "key": f"k{i}", "success": s} for i, s in enumerate([1, 1, 0, 0])]
    arm = [{"init_state_id": i, "key": f"k{i}", "success": s} for i, s in enumerate([1, 0, 1, 0])]
    p = rfp.paired(base, arm)
    assert (p["rescued"], p["harmed"], p["ref_failures"], p["ref_successes"]) == (1, 1, 2, 2)
    with pytest.raises(ValueError):
        rfp.paired(base, [dict(arm[0], key="other")] + arm[1:])


def test_run_fixed_pilot_end_to_end(tmp_path, monkeypatch):
    from flow_residual_rl import pipeline, stages

    rfp = _load("run_fixed_pilot")
    cfg = pilot_cfg(tmp_path)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    prov = FakeProvider(8, drift=0.3)
    mk = lambda cfg_, tid, counter, p: FakeTask(counter, episode_cap=64, goal=0.3, task_id=tid)  # noqa: E731
    for mod in (rfp, stages, pipeline):
        if hasattr(mod, "make_task"):
            monkeypatch.setattr(mod, "make_task", mk)
    monkeypatch.setattr(rfp, "make_provider", lambda cfg_: prov)
    monkeypatch.setattr(rfp, "write_manifest", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["run_fixed_pilot.py", "--config", str(cfg_path), "--task-id", "3", "--steps", "200",
                                      "--block-steps", "100", "--eval-episodes", "3", "--obs-norm-episodes", "2",
                                      "--tail-source", str(_diag(tmp_path))])
    rfp.main()
    tdir = pipeline.task_dir(cfg, 3)
    s = json.loads((tdir / "fixed_pilot_summary.json").read_text())
    assert set(s["arms"]) == {"dct12", "tail12"} and s["base"]["n"] == 3
    for v in s["arms"].values():
        assert v["n"] == 3 and v["rescued"] <= v["ref_failures"] and v["harmed"] <= v["ref_successes"]
    assert "tail12_vs_dct12" in s and (tdir / "obs_norm" / "obs_norm.json").exists()
    for arm in ("dct12", "tail12"):
        st = json.loads((stages.cell_dir(cfg, 3, arm, cfg["training_seeds"][0]) / "status.json").read_text())
        assert st["online_steps"] >= 200
    np.testing.assert_allclose(np.load(stages.cell_dir(cfg, 3, "tail12", cfg["training_seeds"][0]) / "basis.npy")[:, :6],
                               dct_basis(8, 6, 1), atol=1e-6)
