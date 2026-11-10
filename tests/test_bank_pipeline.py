"""Bank -> basis pipeline on the fake env (CPU plumbing only, not an experiment)."""
import numpy as np
import yaml

from fakes import FakeProvider, FakeTask
from flow_residual_rl import bank as bk
from flow_residual_rl.budget import StepCounter
from flow_residual_rl.config import splits, write_json
from flow_residual_rl.envs.chunk_env import ObsNormalizer


def test_bank_and_bases(tmp_path, monkeypatch):
    cfg = yaml.safe_load(open("configs/pilot.yaml"))
    cfg.update(runs_dir=str(tmp_path), episode_cap=96, bank_pool_max_steps=800, bank_budget_steps=6000,
               bank_branch_horizon=48, bank_anchor_target=24, basis_train_updates=200, basis_min_discordant_train=1,
               basis_min_discordant_val=1)
    for mode in ("snapshot", "replay"):
        cfg["restore_mode"] = mode
        counter = StepCounter(None)
        counter.set_limit(bk.BANK_BUCKETS, cfg["bank_budget_steps"])
        task = FakeTask(counter, episode_cap=96, goal=0.35, y_gate=-0.002)
        prov = FakeProvider(8, drift=0.25)
        sp = splits(cfg, task.n_init_states)
        eps, bounds = bk.collect_pool(task, prov, cfg, sp, counter, log=lambda *a: None)
        assert counter.steps["bank_pool"] <= 800 and {b.split for b in bounds} == {"train", "validation"}
        anchors = bk.select_anchors(bounds, cfg, 24, seed=0)
        assert len(set(anchors)) == len(anchors)
        groups = bk.collect_groups(task, prov, cfg, bounds, anchors, counter, tmp_path, log=lambda *a: None)
        assert counter.total(bk.BANK_BUCKETS) <= cfg["bank_budget_steps"]
        assert all(len(g["branches"]) == 3 for g in groups)
        for g in groups:
            np.testing.assert_allclose(g["branches"][1]["planned"], -g["branches"][2]["planned"], atol=1e-6)
            assert np.abs(g["branches"][0]["planned"]).max() == 0
            assert all(br["executed_steps"] <= br["branch_window"] for br in g["branches"])
            if np.isfinite(g["restore_err_vs_pool"]):
                assert g["restore_err_vs_pool"] < 1e-9
        if mode == "replay":
            assert counter.steps["bank_prefix_replay"] > 0
    out = tmp_path / "libero_10_task0" / "bank"
    bk.save_bank(out, groups, eps, {"complete": True, "cost": counter.to_dict()})
    write_json(out / "obs_norm.json", ObsNormalizer.fit(bk.pool_obs_matrix(bounds, cfg)).to_dict())
    from flow_residual_rl.stages import build_bases

    r = build_bases(cfg, 0)
    assert (tmp_path / "libero_10_task0" / "basis" / "pca12" / "basis.npy").exists()
    assert r["status"] == "ok", r
    if r["status"] == "ok":
        assert (tmp_path / "libero_10_task0" / "basis" / "outcome12" / "basis.npy").exists()
    print(r)
