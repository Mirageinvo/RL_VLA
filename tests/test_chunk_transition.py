import numpy as np

from fakes import FakeProvider, FakeTask
from flow_residual_rl.basis import ResidualMap, random_basis
from flow_residual_rl.budget import StepCounter
from flow_residual_rl.config import splits
from flow_residual_rl.envs.chunk_env import ChunkEnv, EpisodeSchedule, ObsNormalizer

C = 8
DIM = 128 + 8 + C * 7 + 1


def make_env(tmp_path, cap=64, drift=0.3, goal=0.5):
    counter = StepCounter(tmp_path / "cost.json")
    task = FakeTask(counter, episode_cap=cap, goal=goal)
    prov = FakeProvider(C, drift=drift)
    rm = ResidualMap(random_basis(48, 12, 0), C, tuple(range(6)), 0.1, task.ACTION_LOW, task.ACTION_HIGH)
    env = ChunkEnv(task, prov, rm, ObsNormalizer.identity(DIM), EpisodeSchedule(range(5), "t", 0),
                   bucket="online_train", settle_bucket="online_settling")
    return env, task, counter


def test_success_terminal_reward_and_early_stop(tmp_path):
    env, task, counter = make_env(tmp_path, cap=640, drift=0.9, goal=0.1)
    obs, _ = env.reset()
    assert obs.shape == (DIM,)
    total, rewards = 0, []
    while True:
        obs, r, term, trunc, info = env.step(np.zeros(12, np.float32))
        total += info["n_env_steps"]
        rewards.append(r)
        if not term:
            assert info["n_env_steps"] == C and r == 0.0
        if term:
            break
    assert info["success"] and r == 1.0 and not trunc
    assert sum(rewards) == 1.0
    assert counter.steps["online_train"] == total == task.elapsed


def test_horizon_is_terminal_failure(tmp_path):
    env, task, counter = make_env(tmp_path, cap=32, drift=-0.5)
    env.reset()
    n = 0
    while True:
        _, r, term, trunc, info = env.step(np.zeros(12, np.float32))
        n += 1
        if term:
            break
    assert n == 4 and r == 0.0 and term and not trunc and not info["success"]
    assert info["episode_record"]["length"] == 32


def test_same_schedule_same_base_noise_across_arms(tmp_path):
    e1, *_ = make_env(tmp_path / "a")
    e2, *_ = make_env(tmp_path / "b")
    o1, _ = e1.reset()
    o2, _ = e2.reset()
    assert np.array_equal(o1, o2) and np.array_equal(e1.proposal.a0_env, e2.proposal.a0_env)


def test_splits_disjoint_and_fixed():
    cfg = {"split_seed": 61, "split_fractions": [0.6, 0.2, 0.2]}
    s = splits(cfg, 50)
    assert len(s["train"]) == 30 and len(s["validation"]) == 10 and len(s["evaluation"]) == 10
    assert not (set(s["train"]) & set(s["evaluation"])) and not (set(s["validation"]) & set(s["evaluation"]))
    assert s == splits(cfg, 50)


def test_sac_end_to_end_resume(tmp_path):
    """SB3 SAC on the fake env: init contract, warmup RMS, block training, save/resume."""
    import torch
    from flow_residual_rl.train_sac import grad_report, make_sac, train_block

    cfg = {"sac_lr": 3e-4, "sac_buffer_size": 1000, "sac_learning_starts_chunks": 256, "sac_batch_size": 256,
           "smoke_sac_learning_starts": 4, "smoke_sac_batch_size": 8, "sac_tau": 0.005, "sac_gamma_chunk": 0.99,
           "sac_train_freq_chunks": 1, "sac_gradient_steps": 1, "sac_ent_coef": "auto_0.01", "sac_hidden": [32, 32],
           "sac_device": "cpu", "warmup_radius": [0.25, 1.0], "sac_log_std_bias_init": -3.0,
           "policy_checkpoint_every": 100}
    env, task, counter = make_env(tmp_path, cap=64)
    m = make_sac(env, cfg, 0, smoke=True)
    assert float(m.actor.mu.weight.abs().sum()) == 0 and torch.allclose(m.actor.log_std.bias, torch.tensor(-3.0))
    a, _ = m._sample_action(4, n_envs=1)
    assert 0.25 - 1e-6 <= np.linalg.norm(a) <= 1.0 + 1e-6
    st = train_block(tmp_path / "cell", env, cfg, 0, 200, smoke=True)
    assert 200 <= st["online_steps"] < 200 + C
    assert counter.steps["online_train"] == st["online_steps"]
    st2 = train_block(tmp_path / "cell", env, cfg, 0, 400, smoke=True)
    assert 400 <= st2["online_steps"] < 400 + C and len(st2["blocks"]) == 2
    assert counter.steps["online_train"] == st2["online_steps"]
    from flow_residual_rl.train_sac import ResidualSAC

    from flow_residual_rl.train_sac import model_dir

    m2 = ResidualSAC.load(model_dir(tmp_path / "cell", st2) / "model.zip", env=env)
    g = grad_report(m2)
    assert m2.replay_buffer is not None
    assert (tmp_path / "cell" / "policy_ckpts" / "policy_0000100.pt").exists()


def test_eval_paths_from_model_and_policy_ckpt(tmp_path):
    from stable_baselines3.sac.policies import SACPolicy

    from flow_residual_rl.evaluate import eval_schedule, run_episodes, sac_policy, zero_policy
    from flow_residual_rl.train_sac import ResidualSAC, make_sac

    cfg = {"sac_lr": 3e-4, "sac_buffer_size": 1000, "sac_learning_starts_chunks": 4, "sac_batch_size": 8,
           "smoke_sac_learning_starts": 4, "smoke_sac_batch_size": 8, "sac_tau": 0.005, "sac_gamma_chunk": 0.99,
           "sac_train_freq_chunks": 1, "sac_gradient_steps": 1, "sac_ent_coef": "auto_0.01", "sac_hidden": [32, 32],
           "sac_device": "cpu", "warmup_radius": [0.25, 1.0], "sac_log_std_bias_init": -3.0}
    env, task, counter = make_env(tmp_path, cap=32, drift=0.9, goal=0.1)
    env.schedule = eval_schedule([0, 1, 2])
    m = make_sac(env, cfg, 0)
    m.policy.save(tmp_path / "p.pt")
    m.save(tmp_path / "m.zip")
    s0 = run_episodes(env, zero_policy(12), 4, tmp_path / "base.json")
    s1 = run_episodes(env, sac_policy(ResidualSAC.load(tmp_path / "m.zip")), 4, tmp_path / "m.json")
    pol = SACPolicy.load(tmp_path / "p.pt")
    s2 = run_episodes(env, lambda o: pol.predict(o, deterministic=True)[0], 4, tmp_path / "p.json")
    # zero-init actor mean == base policy on identical paired schedules
    assert s0["success_rate"] == s1["success_rate"] == s2["success_rate"]
    assert s0["n_unique_states"] == 3
