"""Regressions for the bf3fe64 review (items 1-7). CPU fakes only, not experiments."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from fakes import FakeProvider, FakeTask
from flow_residual_rl import bank as bk
from flow_residual_rl.budget import StepCounter
from flow_residual_rl.config import splits, write_json
from flow_residual_rl.envs.chunk_env import ObsNormalizer

ROOT = Path(__file__).resolve().parents[1]


def pilot_cfg(tmp_path, **kw):
    cfg = yaml.safe_load((ROOT / "configs" / "pilot.yaml").read_text())
    cfg.update(runs_dir=str(tmp_path / "runs"), sac_hidden=[32, 32], sac_buffer_size=2000, sac_learning_starts_chunks=4,
               sac_batch_size=8, policy_checkpoint_every=100, episode_cap=64)
    cfg.update(kw)
    return cfg


def load_smoke():
    spec = importlib.util.spec_from_file_location("smoke_script", ROOT / "scripts" / "smoke.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------- 1. init states
def test_init_states_loader_handles_numpy_pickle(tmp_path):
    from flow_residual_rl.envs.libero_task import load_init_states_file

    arr = np.random.default_rng(0).standard_normal((50, 123))
    f = tmp_path / "x.pruned_init"
    torch.save(arr, f)  # what LIBERO ships: a pickled numpy array
    with pytest.raises(Exception):
        torch.load(f)  # torch >= 2.6 default weights_only=True refuses it (the reported blocker)
    out = load_init_states_file(f)
    assert out.shape == (50, 123) and np.array_equal(out, arr)


def test_real_libero_task0_init_states():
    pytest.importorskip("libero.libero")
    from libero.libero import benchmark

    from flow_residual_rl.envs.libero_task import load_task_init_states

    bm = benchmark.get_benchmark_dict()["libero_10"]()
    s = load_task_init_states(bm, 0)
    assert s.shape[0] == 50 and s.ndim == 2


# ---------------------------------------------------------------- 2. restore check
def _check3(tmp_path, mode, bias):
    smoke = load_smoke()
    cfg = pilot_cfg(tmp_path, restore_mode=mode)
    counter = StepCounter(None)
    task = FakeTask(counter, episode_cap=720, goal=10.0)
    task.restore_bias = bias
    with counter.charge("smoke"):
        res = smoke.check3(cfg, task, FakeProvider(8, drift=0.3), {"train": [0, 1, 2]})
    return res, task


def test_restore_check_catches_repeatable_wrong_restore(tmp_path):
    res, task = _check3(tmp_path, "snapshot", 0.1)
    assert res["pass"] is False and task.n_restore > 0
    a = res["anchors"][0]
    assert a["state_repeat"]["ok"] and not a["state_vs_original"]["ok"]  # repeatable, but not the original


def test_restore_check_passes_exact_snapshot(tmp_path):
    res, _ = _check3(tmp_path, "snapshot", 0.0)
    assert res["pass"] is True


def test_restore_check_uses_replay_path_when_configured(tmp_path):
    res, task = _check3(tmp_path, "replay", 0.1)  # bias only affects snapshot restore
    assert task.n_replay == 10 and task.n_restore == 0 and res["pass"] is True and res["restore_env_steps"] > 0


def test_bank_rejects_groups_failing_restore_and_aborts(tmp_path):
    cfg = pilot_cfg(tmp_path, episode_cap=96, bank_pool_max_steps=400, bank_branch_horizon=48, bank_budget_steps=6000,
                    bank_max_restore_failures=3, restore_mode="snapshot")
    counter = StepCounter(None)
    task = FakeTask(counter, episode_cap=96, goal=10.0)
    prov = FakeProvider(8, drift=0.25)
    sp = splits(cfg, task.n_init_states)
    _, bounds = bk.collect_pool(task, prov, cfg, sp, counter, log=lambda *a: None)
    task.restore_bias = 0.01
    rejected = []
    with pytest.raises(bk.RestoreCheckFailed):
        bk.collect_groups(task, prov, cfg, bounds, bk.select_anchors(bounds, cfg, 12, seed=0), counter, tmp_path,
                          log=lambda *a: None, rejected=rejected)
    assert len(rejected) == 3


# ---------------------------------------------------------------- 3. smoke verdict / exit code
def test_smoke_overall_verdict():
    smoke = load_smoke()
    ok = {"check1": {"pass": True}, "check2": {"pass": True}, "check3": {"pass": True},
          "check4_5": {"pass4": True, "pass5": True}}
    assert smoke.summarize(ok)["overall_pass"]
    assert not smoke.summarize(dict(ok, check4_5={"pass4": False, "pass5": True}))["overall_pass"]
    assert not smoke.check_passed({"pass4": False, "pass5": True})
    assert not smoke.summarize(dict(ok, check3={"pass": False}))["overall_pass"]
    assert not smoke.summarize({k: v for k, v in ok.items() if k != "check2"})["overall_pass"]
    stale = {k: dict(v, fingerprint="old") for k, v in ok.items()}
    assert not smoke.summarize(stale, "new")["overall_pass"] and smoke.summarize(stale, "old")["overall_pass"]


# ---------------------------------------------------------------- 4. durable cost / bank retry
def test_counter_flushes_without_explicit_save(tmp_path):
    c = StepCounter(tmp_path / "cost.json", flush_every=50)
    with c.charge("online_train"):
        for _ in range(120):
            c.on_env_step()
    # process "dies" here without save()
    c2 = StepCounter(tmp_path / "cost.json")
    assert c2.steps["online_train"] == 100  # flushed at 50 and 100; at most flush_every-1 lost on a hard kill
    c2.reconcile("online_train", 64, "aborted_online", "test")
    assert c2.steps["online_train"] == 64 and c2.steps["aborted_online"] == 36
    assert c2.total() == StepCounter(tmp_path / "cost.json").total()  # reconcile never deletes steps


def test_bank_retry_keeps_previous_spend_in_limit(tmp_path):
    cfg = pilot_cfg(tmp_path, episode_cap=96, bank_pool_max_steps=160, bank_branch_horizon=24, settle_steps=0)
    counter = StepCounter(None)
    counter.steps["aborted_bank"] = 150  # crashed attempt
    counter.set_limit(bk.BANK_LIMIT_BUCKETS, 200)
    task = FakeTask(counter, episode_cap=96, goal=10.0)
    prov = FakeProvider(8, drift=0.25)
    sp = splits(cfg, task.n_init_states)
    _, bounds = bk.collect_pool(task, prov, cfg, sp, counter, log=lambda *a: None)
    bk.collect_groups(task, prov, cfg, bounds, bk.select_anchors(bounds, cfg, 8, seed=0), counter, tmp_path,
                      log=lambda *a: None)
    assert counter.total(bk.BANK_LIMIT_BUCKETS) <= 200
    assert counter.to_dict()["bank_total"] == counter.total(bk.BANK_LIMIT_BUCKETS)


# ---------------------------------------------------------------- 5/7. cell resume contract + transitions
def _cell_setup(tmp_path, **kw):
    from flow_residual_rl.pipeline import task_dir

    cfg = pilot_cfg(tmp_path, **kw)
    out = task_dir(cfg, 0) / "bank"
    out.mkdir(parents=True)
    write_json(out / "obs_norm.json", ObsNormalizer.identity(128 + 8 + 8 * 7 + 1).to_dict())
    counter = StepCounter(None)
    task = FakeTask(counter, episode_cap=64, goal=0.3)
    return cfg, task, FakeProvider(8, drift=0.3), splits(cfg, task.n_init_states)


def test_block_keeps_every_executed_transition_and_resume_contract(tmp_path):
    from flow_residual_rl.fingerprint import ContractMismatch
    from flow_residual_rl.stages import cell_dir, eval_cell, run_cell_block
    from flow_residual_rl.train_sac import ResidualSAC, model_dir

    cfg, task, prov, sp = _cell_setup(tmp_path)
    log = lambda *a: None  # noqa: E731
    st = run_cell_block(cfg, task, prov, "random12", 61, 200, sp, log)
    b = st["blocks"][-1]
    assert b["replay_added"] == b["chunks_this_block"] == b["chunks"]  # last transition is stored
    st = run_cell_block(cfg, task, prov, "random12", 61, 400, sp, log)
    cdir = cell_dir(cfg, 0, "random12", 61)
    m = ResidualSAC.load(model_dir(cdir, st) / "model.zip")
    m.load_replay_buffer(model_dir(cdir, st) / "replay.pkl")
    assert m.replay_buffer.size() == m.num_timesteps == sum(x["chunks_this_block"] for x in st["blocks"])
    assert len(list((cdir / "gens").iterdir())) == 1  # only the committed generation is kept
    cost = json.loads((cdir / "cost.json").read_text())
    assert cost["steps"]["online_train"] == st["online_steps"]
    # eval from the exact checkpoint, with the actual step count recorded
    s = eval_cell(cfg, task, prov, "random12", 61, 2, 200, sp, log)
    assert 200 <= s["policy_actual_online_steps"] < 208
    # changing the action contract must refuse to resume the old replay / model
    with pytest.raises(ContractMismatch):
        run_cell_block(dict(cfg, epsilon_rms=0.2), task, prov, "random12", 61, 600, sp, log)
    # growing the online target is allowed
    run_cell_block(cfg, task, prov, "random12", 61, 500, sp, log)


def test_crashed_block_is_booked_as_aborted(tmp_path, monkeypatch):
    from flow_residual_rl import train_sac
    from flow_residual_rl.stages import cell_dir, run_cell_block

    cfg, task, prov, sp = _cell_setup(tmp_path)
    log = lambda *a: None  # noqa: E731
    run_cell_block(cfg, task, prov, "dct12", 61, 200, sp, log)
    orig = train_sac.ResidualSAC._store_transition
    calls = {"n": 0}

    def boom(self, *a, **k):
        calls["n"] += 1
        if calls["n"] == 10:
            raise KeyboardInterrupt
        return orig(self, *a, **k)

    monkeypatch.setattr(train_sac.ResidualSAC, "_store_transition", boom)
    with pytest.raises(KeyboardInterrupt):
        run_cell_block(cfg, task, prov, "dct12", 61, 400, sp, log)
    monkeypatch.setattr(train_sac.ResidualSAC, "_store_transition", orig)
    cdir = cell_dir(cfg, 0, "dct12", 61)
    spent = json.loads((cdir / "cost.json").read_text())["steps"]["online_train"]
    assert spent > 200  # the crashed steps reached disk (finally: save)
    st = run_cell_block(cfg, task, prov, "dct12", 61, 400, sp, log)
    cost = json.loads((cdir / "cost.json").read_text())["steps"]
    assert cost["aborted_online"] == spent - 200 and cost["online_train"] == st["online_steps"]
    # the crashed block's episode records were dropped: no episode appears twice in the training curve
    recs = [json.loads(l) for l in (cdir / "train_episodes.jsonl").read_text().splitlines() if l.strip()]
    keys = [r["key"] for r in recs]
    assert len(keys) == len(set(keys)) and st["dropped_uncommitted_episode_records"][0] >= 1
    assert st["episodes_logged"] == len(recs) and max(r["online_steps_end"] for r in recs) <= st["online_steps"]


def test_report_prefers_larger_eval_at_same_step(tmp_path):
    from flow_residual_rl.report import load_cells

    cdir = tmp_path / "libero_10_task0" / "cells" / "outcome12_seed61"
    (cdir / "evals").mkdir(parents=True)
    for n in (50, 100):
        eps = [{"episode": e, "key": f"eval/0/{e}", "success": True, "init_state_id": 0} for e in range(n)]
        write_json(cdir / "evals" / f"eval_0060000_n{n}.json",
                   {"summary": {"online_steps": 60000, "n_episodes": n, "policy_ckpt": "p.pt"}, "episodes": eps})
    (c,) = load_cells(tmp_path)
    assert c["evals"][60000]["summary"]["n_episodes"] == 100


def test_weights_identity_detects_replaced_weights(tmp_path):
    import os

    from flow_residual_rl.pipeline import weights_identity

    ck = tmp_path / "ckpt"
    ck.mkdir()
    (ck / "model.safetensors").write_bytes(b"a" * 1000)
    cache = tmp_path / "cache.json"
    h1 = weights_identity(ck, cache)
    assert weights_identity(ck, cache) == h1  # cached
    (ck / "model.safetensors").write_bytes(b"b" * 1000)  # same size
    os.utime(ck / "model.safetensors", ns=(1, 1))
    assert weights_identity(ck, cache) != h1


def test_fingerprint_covers_weights_and_projection(tmp_path):
    from flow_residual_rl import fingerprint as fpr

    ck = tmp_path / "ckpt"
    ck.mkdir()
    (ck / "model.safetensors").write_bytes(b"a" * 10)
    cfg = pilot_cfg(tmp_path, checkpoint_dir=str(ck))

    class P(FakeProvider):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def projection_digest(self):
            return fpr.sha_array(np.full((4, 2), self.m, np.float32))

    f1 = fpr.common(cfg, 0, P(1.0))
    assert fpr.common(cfg, 0, P(2.0))["context_projection"] != f1["context_projection"]
    (ck / "model.safetensors").write_bytes(b"bbbbbbbbbbbb")
    assert fpr.common(cfg, 0, P(1.0))["checkpoint"] != f1["checkpoint"]


def test_project_wall_ledger_counts_all_commands(tmp_path):
    from flow_residual_rl.walltime import WallLedger, project_hours

    write_json(tmp_path / "wall" / "collect_bank_1_1.json",  # a crashed earlier command: counted to its last beat
               {"command": "collect_bank", "pid": 1, "start": 0.0, "last": 7200.0, "finished": False})
    led = WallLedger(tmp_path, "run_pilot_A", heartbeat_s=3600)
    h = led.project_hours()
    led.close()
    assert 2.0 <= h < 2.01 and 2.0 <= project_hours(tmp_path) < 2.01


# ---------------------------------------------------------------- 6. Stage-A decision
def _cell(arm, sr, at=60000, n=50):
    eps = [{"episode": e, "key": f"eval/0/{e}", "init_state_id": e % 10, "success": e < sr * n} for e in range(n)]
    return {"task_id": 0, "arm": arm, "seed": 61, "status": {"online_steps": at},
            "evals": {at: {"summary": {"success_rate": sr, "n_episodes": n}, "episodes": eps}},
            "train_eps": [], "base_eval": [dict(e, success=False) for e in eps], "bank_steps": 0}


def test_stage_a_decision_requires_all_arms():
    from flow_residual_rl.report import decide

    req = ("outcome12", "random12", "dct12", "full48")
    d = decide([_cell("outcome12", 0.7), _cell("random12", 0.5)], 0, 61, 60000, req, 50)
    assert d["decision"] == "partial" and any("dct12" in m for m in d["missing"])
    d = decide([_cell("outcome12", 0.7)], 0, 61, 60000, req, 50)  # used to crash on max([])
    assert d["decision"] == "partial"
    d = decide([_cell(a, s) for a, s in zip(req, (0.7, 0.5, 0.5, 0.5))], 0, 61, 60000, req, 50)
    assert d["decision"] == "continue"
    short = [_cell(a, s, n=(25 if a == "dct12" else 50)) for a, s in zip(req, (0.7, 0.5, 0.5, 0.5))]
    assert decide(short, 0, 61, 60000, req, 50)["decision"] == "partial"
