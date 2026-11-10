"""Correction bases and the fixed latent-action -> env-correction map.

All bases live in the flattened arm-correction space of size N = C * n_dims
(row-major over [C, n_dims], i.e. index = t * n_dims + k), in env action units.
Every basis B has shape [N, d] with B^T B = I.

Map (plan sec. 7):
    w     = u / max(1, ||u||_2)
    delta = epsilon * sqrt(N) * B @ w        -> RMS(delta) <= epsilon for any d
    a_arm = clip(a0_arm + delta.reshape(C, n_dims), low, high)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ARM_NAMES = ("outcome12", "random12", "dct12", "pca12", "full48")


def thin_qr(w: np.ndarray) -> np.ndarray:
    """Thin QR in float32 with column signs fixed so that diag(R) > 0."""
    w = np.asarray(w, dtype=np.float32)
    q, r = np.linalg.qr(w, mode="reduced")
    s = np.sign(np.diag(r))
    s[s == 0] = 1.0
    return (q * s[None, :]).astype(np.float32)


def random_basis(n: int, d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return thin_qr(rng.standard_normal((n, d)))


def dct_basis(chunk: int, n_dims: int, modes_per_dim: int) -> np.ndarray:
    """Orthonormal DCT-II temporal modes (first `modes_per_dim`) per coordinate.

    Column order: for mode m, for coordinate k. Kronecker product of the
    [C, m] DCT matrix and the [n_dims, n_dims] identity.
    """
    t = np.arange(chunk)
    cols = []
    for m in range(modes_per_dim):
        v = np.cos(np.pi * (t + 0.5) * m / chunk)
        v = v / np.linalg.norm(v)
        cols.append(v)
    dct = np.stack(cols, axis=1)  # [C, M]
    b = np.kron(dct, np.eye(n_dims))  # [C*n_dims, M*n_dims]
    return b.astype(np.float32)


def pca_basis(corrections: np.ndarray, d: int) -> np.ndarray:
    """Top-d eigenvectors of the (uncentered-mean-free) covariance of planned
    corrections. The mean is subtracted for the covariance but never added back
    in the decoder."""
    x = np.asarray(corrections, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    cov = x.T @ x / max(1, x.shape[0] - 1)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1][:d]
    v = evecs[:, order]
    # deterministic sign: largest-|.| entry positive
    idx = np.argmax(np.abs(v), axis=0)
    v = v * np.sign(v[idx, np.arange(d)])[None, :]
    return v.astype(np.float32)


def identity_basis(n: int) -> np.ndarray:
    return np.eye(n, dtype=np.float32)


def check_orthonormal(b: np.ndarray, atol: float = 1e-5) -> float:
    err = float(np.abs(b.T @ b - np.eye(b.shape[1])).max())
    if err > atol:
        raise AssertionError(f"B^T B != I (max err {err:.2e})")
    return err


@dataclass
class ResidualMap:
    """Fixed map from SAC latent action u in [-1,1]^d to an executed chunk."""

    basis: np.ndarray  # [N, d]
    chunk: int
    corrected_dims: tuple[int, ...]
    epsilon: float
    low: np.ndarray  # env action bounds, [A]
    high: np.ndarray

    def __post_init__(self):
        self.basis = np.asarray(self.basis, dtype=np.float32)
        self.n = self.chunk * len(self.corrected_dims)
        assert self.basis.shape[0] == self.n, (self.basis.shape, self.n)
        check_orthonormal(self.basis)
        self.scale = self.epsilon * np.sqrt(self.n)

    @property
    def dim(self) -> int:
        return self.basis.shape[1]

    def delta(self, u: np.ndarray) -> np.ndarray:
        """Planned correction [C, n_dims] before clipping."""
        u = np.asarray(u, dtype=np.float32).reshape(self.dim)
        w = u / max(1.0, float(np.linalg.norm(u)))
        return (self.scale * (self.basis @ w)).reshape(self.chunk, len(self.corrected_dims))

    def apply(self, a0: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, dict]:
        """a0: [C, A] base env commands. Returns executed [C, A] and stats.

        u == 0 returns a0 bit-exactly (no clip, no dtype change)."""
        a0 = np.asarray(a0)
        if not np.any(u):
            return a0.copy(), {"rms_planned": 0.0, "rms_actual": 0.0, "clip_frac": 0.0}
        dims = list(self.corrected_dims)
        d = self.delta(u)
        out = a0.copy()
        raw = a0[:, dims] + d
        clipped = np.clip(raw, self.low[dims], self.high[dims])
        out[:, dims] = clipped
        actual = clipped - a0[:, dims]
        stats = {
            "rms_planned": float(np.sqrt(np.mean(d**2))),
            "rms_actual": float(np.sqrt(np.mean(actual**2))),
            "clip_frac": float(np.mean(raw != clipped)),
        }
        return out, stats

    def encode(self, delta: np.ndarray) -> np.ndarray:
        """Coefficients c = B^T delta (in scaled units, as fed to the predictor)."""
        return (self.basis.T @ np.asarray(delta, dtype=np.float32).reshape(-1)).astype(np.float32)


def build_fixed_basis(arm: str, chunk: int, n_dims: int, rank: int, seed: int) -> np.ndarray:
    n = chunk * n_dims
    if arm.startswith("random"):
        return random_basis(n, rank, seed)
    if arm.startswith("dct"):
        assert rank % n_dims == 0, "DCT rank must be a multiple of n_dims"
        return dct_basis(chunk, n_dims, rank // n_dims)
    if arm.startswith("full"):
        return identity_basis(n)
    raise ValueError(f"{arm} is data-dependent; load it from disk")


def save_basis(path: Path, b: np.ndarray, meta: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    err = check_orthonormal(b)
    np.save(path / "basis.npy", b.astype(np.float32))
    meta = dict(meta, shape=list(b.shape), orthonormality_max_err=err)
    (path / "basis_metadata.json").write_text(json.dumps(meta, indent=2))


def load_basis(path: Path) -> np.ndarray:
    b = np.load(Path(path) / "basis.npy")
    check_orthonormal(b)
    return b
