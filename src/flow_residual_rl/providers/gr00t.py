"""Frozen GR00T N1.7 proposal provider (plan sec. 6).

Wraps upstream gr00t.policy.gr00t_policy.Gr00tPolicy (official decode /
unnormalisation path). Adds:
  * per-call base-noise seeding via torch.random.fork_rng (flow-matching noise
    is drawn with the global torch.randn in Gr00tN1d7ActionHead.get_action_with_features);
  * one read-only forward hook on `model.backbone` capturing
    backbone_features [B, L, D] and backbone_attention_mask [B, L]; mean-pool over
    valid tokens, fixed random projection to `context_dim`;
  * conversion to LIBERO commands identical to LiberoEnv.step
    (gripper: normalize [0,1]->[-1,1], binarize by sign, invert) and arm clip to
    the env Box [-1,1] (robosuite OSC clips its input to the same range).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ARM_KEYS = ("x", "y", "z", "roll", "pitch", "yaw")
GRIPPER_KEY = "gripper"
HOOK_MODULE = "model.backbone"


@dataclass
class Proposal:
    a0_env: np.ndarray  # float32 [C, 7]
    context: np.ndarray  # float32 [context_dim]
    proprio: np.ndarray  # float32 [8]
    base_seed: int
    metadata: dict = field(default_factory=dict)
    raw_chunk: np.ndarray | None = None  # float32 [H, 7] model-space (decoded) chunk, for logging


def model_to_env(chunk7: np.ndarray) -> np.ndarray:
    """[T,7] decoded model actions -> LIBERO env commands (GR00T LiberoEnv convention)."""
    a = np.array(chunk7, dtype=np.float32, copy=True)
    a[:, :6] = np.clip(a[:, :6], -1.0, 1.0)
    g = 2.0 * a[:, 6] - 1.0
    a[:, 6] = -np.sign(g)
    return a


class Gr00tProvider:
    def __init__(self, checkpoint_dir: str, embodiment_tag: str, device: str = "cuda:0", dtype: str = "auto",
                 chunk: int = 8, context_dim: int = 128, projection_path: Path | None = None,
                 projection_seed: int = 0, counter=None):
        import torch
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        self.torch = torch
        self.policy = Gr00tPolicy(embodiment_tag=embodiment_tag, model_path=checkpoint_dir, device=device)
        model = self.policy.model
        if dtype == "auto":
            native = torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False)
            dtype = "bfloat16" if native else "float32"
        self.dtype_name = dtype
        tdt = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
        model.to(dtype=tdt)  # Gr00tPolicy casts inputs to bf16; prepare_input re-casts to the model dtype
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model
        self.device = device
        self.chunk = chunk
        self.counter = counter
        mc = self.policy.modality_configs
        self.video_keys = list(mc["video"].modality_keys)
        self.state_keys = list(mc["state"].modality_keys)
        self.action_keys = list(mc["action"].modality_keys)
        self.language_key = self.policy.language_key
        self.video_delta = list(mc["video"].delta_indices)
        self.state_delta = list(mc["state"].delta_indices)
        self.action_horizon = len(mc["action"].delta_indices)
        assert set(ARM_KEYS) | {GRIPPER_KEY} <= set(self.action_keys), self.action_keys
        assert chunk <= self.action_horizon
        self._captured = None
        self._hook = model.backbone.register_forward_hook(self._capture)
        self.context_dim = context_dim
        self.projection_path = Path(projection_path) if projection_path else None
        self.projection_seed = projection_seed
        self.P = None
        if self.projection_path and self.projection_path.exists():
            self.P = np.load(self.projection_path)

    # ---------------- feature hook ----------------
    def _capture(self, module, inputs, output):
        feats = output["backbone_features"].detach()
        mask = output.get("backbone_attention_mask") if hasattr(output, "get") else None
        if mask is None:
            pooled = feats.float().mean(1)
        else:
            m = mask.to(feats.device).float().unsqueeze(-1)
            pooled = (feats.float() * m).sum(1) / m.sum(1).clamp_min(1.0)
        self._captured = pooled.cpu().numpy().astype(np.float32)
        self._captured_shape = tuple(feats.shape)

    def _project(self, pooled: np.ndarray) -> np.ndarray:
        if self.P is None:
            rng = np.random.default_rng(self.projection_seed)
            D = pooled.shape[-1]
            self.P = (rng.standard_normal((D, self.context_dim)) / np.sqrt(self.context_dim)).astype(np.float32)
            if self.projection_path:
                self.projection_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(self.projection_path, self.P)
        return pooled @ self.P

    # ---------------- main API ----------------
    def history_spec(self) -> tuple[int, int]:
        d = self.video_delta
        stride = (d[1] - d[0]) if len(d) > 1 else 1
        assert self.state_delta == self.video_delta or len(self.state_delta) == 1, (self.video_delta, self.state_delta)
        return len(d), stride

    def propose(self, observation: dict, task: str | None, proposal_seed: int) -> Proposal:
        """observation: LiberoTask.policy_observation() (batch=1 nested dict)."""
        torch = self.torch
        obs = {
            "video": {k: observation["video"][k] for k in self.video_keys},
            "state": {k: observation["state"][k][:, -len(self.state_delta):] for k in self.state_keys},
            "language": {self.language_key: [[task if task is not None else observation["language"]]]},
        }
        devs = [torch.device(self.device)] if torch.cuda.is_available() and "cuda" in str(self.device) else []
        t0 = time.time()
        with torch.random.fork_rng(devices=devs):
            torch.manual_seed(int(proposal_seed) % (2**63))
            self._captured = None
            action, _info = self.policy.get_action(obs)
        if devs:
            torch.cuda.synchronize()
        dt = time.time() - t0
        if self.counter is not None:
            self.counter.on_base_call(dt)
        assert self._captured is not None, "backbone hook did not fire"
        raw = np.concatenate([action[k][0] for k in (*ARM_KEYS, GRIPPER_KEY)], axis=-1).astype(np.float32)  # [H,7]
        a0 = model_to_env(raw[: self.chunk])
        proprio = np.concatenate([observation["state"][k][0, -1] for k in ("x", "y", "z", "roll", "pitch", "yaw", "gripper")]).astype(np.float32)
        return Proposal(
            a0_env=a0,
            context=self._project(self._captured[0]).astype(np.float32),
            proprio=proprio,
            base_seed=int(proposal_seed),
            metadata=self.metadata(),
            raw_chunk=raw,
        )

    def metadata(self) -> dict:
        return {
            "H": self.action_horizon,
            "C": self.chunk,
            "source_dtype": self.dtype_name,
            "action_conversion": "gr00t_libero_env_v1: arm clip[-1,1]; gripper -sign(2g-1)",
            "hook_module": HOOK_MODULE,
            "hook_shape": list(getattr(self, "_captured_shape", ())),
            "num_inference_timesteps": int(getattr(self.model.action_head, "num_inference_timesteps", -1)),
            "video_delta_indices": self.video_delta,
            "state_delta_indices": self.state_delta,
            "video_keys": self.video_keys,
            "language_key": self.language_key,
        }

    def check_frozen(self) -> dict:
        n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.model.parameters())
        return {"trainable": n_train, "total": n_total, "training_mode": self.model.training}
