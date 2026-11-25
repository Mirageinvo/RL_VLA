"""Time SAC gradient updates and action sampling on the configured device at several torch
intra-op thread counts. No simulator, no GR00T: a dummy env with the experiment's observation
and action sizes, a replay buffer filled with random transitions, the configured network,
batch size and optimiser.

Stage A ran ~5.3 online env steps/s vs ~9.4 for pure evaluation. Training does one gradient
step per chunk (C env steps), so ms_per_update / C is the update cost per env step. Run this
while nothing else uses the CPU, and compare with the evaluation-only cost per env step.

  python scripts/profile_sac.py --config configs/pilot.yaml
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from flow_residual_rl.config import base_parser, load_config


def make_model(cfg, d, obs_dim):
    import gymnasium as gym
    from stable_baselines3 import SAC
    from stable_baselines3.common.logger import configure

    class Dummy(gym.Env):
        observation_space = gym.spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        action_space = gym.spaces.Box(-1, 1, (d,), np.float32)

        def reset(self, seed=None, options=None):
            return np.zeros(obs_dim, np.float32), {}

        def step(self, a):
            return np.zeros(obs_dim, np.float32), 0.0, False, False, {}

    m = SAC("MlpPolicy", Dummy(), learning_rate=cfg["sac_lr"], buffer_size=20000, batch_size=cfg["sac_batch_size"],
            tau=cfg["sac_tau"], gamma=cfg["sac_gamma_chunk"], ent_coef=cfg["sac_ent_coef"], target_entropy=-float(d),
            policy_kwargs=dict(net_arch=dict(pi=list(cfg["sac_hidden"]), qf=list(cfg["sac_hidden"]))),
            device=cfg["sac_device"], seed=0, verbose=0)
    m.set_logger(configure(None, []))
    rng = np.random.default_rng(0)
    for _ in range(5000):
        o, o2 = rng.standard_normal((2, 1, obs_dim)).astype(np.float32)
        a = rng.uniform(-1, 1, (1, d)).astype(np.float32)
        m.replay_buffer.add(o, o2, a, np.array([float(rng.random() < 0.05)]), np.array([False]), [{}])
    return m


def main():
    p = base_parser(__doc__)
    p.add_argument("--threads", type=int, nargs="*", default=[0, 1, 2, 4, 8], help="0 = torch default")
    p.add_argument("--dims", type=int, nargs="*", default=[12, 48])
    p.add_argument("--updates", type=int, default=200)
    a = p.parse_args()
    cfg = load_config(a.config, a.set)
    import torch

    from flow_residual_rl.pipeline import obs_dim

    default = torch.get_num_threads()
    print(f"cpu_count={os.cpu_count()} torch={torch.__version__} default_threads={default} "
          f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')} MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS')} "
          f"device={cfg['sac_device']} obs_dim={obs_dim(cfg)} batch={cfg['sac_batch_size']} hidden={cfg['sac_hidden']}")
    C = cfg["execution_horizon"]
    print(f"{'d':>3} {'threads':>7} {'ms/update':>10} {'ms/predict':>10} {'update ms per env step':>22}")
    for d in a.dims:
        m = make_model(cfg, d, obs_dim(cfg))
        obs = np.zeros(obs_dim(cfg), np.float32)
        for n in a.threads:
            torch.set_num_threads(n or default)
            m.train(gradient_steps=10, batch_size=cfg["sac_batch_size"])  # warm-up
            t0 = time.perf_counter()
            m.train(gradient_steps=a.updates, batch_size=cfg["sac_batch_size"])
            up = (time.perf_counter() - t0) / a.updates * 1e3
            t0 = time.perf_counter()
            for _ in range(200):
                m.predict(obs, deterministic=False)
            pr = (time.perf_counter() - t0) / 200 * 1e3
            print(f"{d:>3} {n or default:>7} {up:>10.2f} {pr:>10.2f} {up / C:>22.2f}")
    torch.set_num_threads(default)


if __name__ == "__main__":
    main()
