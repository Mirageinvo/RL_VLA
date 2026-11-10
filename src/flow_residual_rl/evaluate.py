"""Full-episode evaluation on held-out init states with a fixed paired schedule.

Episode e of the evaluation schedule uses init state eval_ids[e % n] (shuffled per
epoch with a fixed seed) and base-noise key "eval/<seed>/<e>", identical for the
base VLA and every arm, so outcomes are paired per (init state, noise schedule).
SAC actions are deterministic (actor mean).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .envs.chunk_env import ChunkEnv, EpisodeSchedule

EVAL_STREAM = "eval"
EVAL_SEED = 0  # fixed: eval schedule never depends on the training seed


def run_episodes(env: ChunkEnv, policy_fn, n_episodes: int, out_path: Path | None = None, video_dir: Path | None = None,
                 n_videos: int = 0) -> dict:
    recs = []
    for e in range(n_episodes):
        ep = env.schedule.get(e)
        obs, _ = env.reset(options={"episode": ep})
        frames = [] if (video_dir and e < n_videos) else None
        t0 = time.time()
        while True:
            u = policy_fn(obs)
            obs, r, term, trunc, info = env.step(u)
            if frames is not None:
                po = env.task.policy_observation()
                if "video" in po:  # real LIBERO task (CPU test fakes have no camera)
                    frames.append(po["video"]["image"][0, -1])
            if term or trunc:
                break
        rec = dict(info["episode_record"], episode=e, wall_s=time.time() - t0)
        recs.append(rec)
        if frames:
            _save_video(frames, Path(video_dir) / f"ep{e:03d}_s{rec['init_state_id']}_{'ok' if rec['success'] else 'fail'}.mp4")
    sr = float(np.mean([r["success"] for r in recs])) if recs else float("nan")
    summary = {"n_episodes": len(recs), "success_rate": sr,
               "n_unique_states": len({r["init_state_id"] for r in recs}),
               "mean_length": float(np.mean([r["length"] for r in recs])),
               "rms_median": float(np.median([r["rms_median"] for r in recs])),
               "clip_mean": float(np.mean([r["clip_mean"] for r in recs]))}
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"summary": summary, "episodes": recs}, indent=1))
    return summary


def _save_video(frames, path: Path):
    try:
        import imageio

        path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(path, frames, fps=20)
    except Exception as e:  # noqa: BLE001
        print(f"[video] skipped {path}: {e}")


def eval_schedule(eval_ids) -> EpisodeSchedule:
    return EpisodeSchedule(eval_ids, EVAL_STREAM, EVAL_SEED)


def zero_policy(d: int):
    z = np.zeros(d, np.float32)
    return lambda obs: z


def sac_policy(model):
    return lambda obs: model.predict(obs, deterministic=True)[0]
