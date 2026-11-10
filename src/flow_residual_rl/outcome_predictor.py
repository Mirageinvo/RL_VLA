"""Outcome-supervised basis learning (plan sec. 10).

B = thin_QR(W), W in R^{N x d}. For branch j of anchor g:
    c_j = B^T delta_j / (epsilon * sqrt(N))          (fixed scale -> unit ball)
    s_j = f_phi([x_g, c_j]),  x_g = [context, proprio, a0_env, remaining_frac]
Loss = group-weighted BCE(s_j, y_j) + lambda_pair * pairwise logistic over
discordant pairs inside an anchor. Every anchor group has equal total weight.

Inputs are strictly pre-branch information. Outcome-derived quantities
(first_success_step, executed_steps, termination masks) never enter f_phi.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def torch_thin_qr(w: torch.Tensor) -> torch.Tensor:
    q, r = torch.linalg.qr(w, mode="reduced")
    s = torch.sign(torch.diagonal(r))
    s = torch.where(s == 0, torch.ones_like(s), s)
    return q * s[None, :]


@dataclass
class GroupBatch:
    """Padded tensors for G anchor groups with up to K branches each."""

    x: torch.Tensor  # [G, Dx] normalised pre-branch features
    delta: torch.Tensor  # [G, K, N] planned corrections (after known clip), env units
    y: torch.Tensor  # [G, K] float {0,1}
    mask: torch.Tensor  # [G, K] bool valid branch


class OutcomeModel(nn.Module):
    def __init__(self, x_dim: int, n: int, rank: int, hidden=(64, 64), delta_scale: float = 1.0, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W = nn.Parameter(torch.randn(n, rank, generator=g))
        self.delta_scale = float(delta_scale)
        layers, d = [], x_dim + rank
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, 1))
        self.f = nn.Sequential(*layers)

    def basis(self) -> torch.Tensor:
        return torch_thin_qr(self.W)

    def forward(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        b = self.basis()  # [N, d]
        c = (delta @ b) / self.delta_scale  # [G, K, d]
        xk = x[:, None, :].expand(-1, delta.shape[1], -1)
        return self.f(torch.cat([xk, c], dim=-1)).squeeze(-1)  # [G, K]


def group_losses(s: torch.Tensor, b: GroupBatch, pair_weight: float):
    m = b.mask.float()
    k = m.sum(1).clamp_min(1)
    bce = F.binary_cross_entropy_with_logits(s, b.y, reduction="none")
    bce_g = (bce * m).sum(1) / k  # per-group mean -> equal group weight
    # pairwise logistic inside a group over pairs with y_i != y_j
    dy = b.y[:, :, None] - b.y[:, None, :]
    ds = s[:, :, None] - s[:, None, :]
    pm = (dy > 0) & b.mask[:, :, None] & b.mask[:, None, :]  # ordered i>j pairs once
    pl = F.softplus(-ds) * pm.float()
    npairs = pm.float().sum((1, 2))
    pair_g = torch.where(npairs > 0, pl.sum((1, 2)) / npairs.clamp_min(1), torch.zeros_like(npairs))
    loss = (bce_g + pair_weight * pair_g).mean()
    return loss, bce_g.mean().detach(), pair_g.detach(), npairs


@torch.no_grad()
def evaluate(model: OutcomeModel, b: GroupBatch, pair_weight: float) -> dict:
    s = model(b.x, b.delta)
    loss, bce, pair_g, npairs = group_losses(s, b, pair_weight)
    disc = npairs > 0
    # held-out pair accuracy over discordant pairs
    dy = b.y[:, :, None] - b.y[:, None, :]
    ds = s[:, :, None] - s[:, None, :]
    pm = (dy > 0) & b.mask[:, :, None] & b.mask[:, None, :]
    correct = ((ds > 0) & pm).float().sum()
    ties = ((ds == 0) & pm).float().sum()
    total = pm.float().sum()
    # constant predictor reference: p = mean(y) of THIS split (optimistic for the constant)
    m = b.mask.float()
    p = float((b.y * m).sum() / m.sum().clamp_min(1))
    p = min(max(p, 1e-4), 1 - 1e-4)
    const_bce = F.binary_cross_entropy(torch.full_like(b.y, p), b.y, reduction="none")
    const_bce = float(((const_bce * m).sum(1) / m.sum(1).clamp_min(1)).mean())
    return {
        "loss": float(loss),
        "bce": float(bce),
        "pair_loss_discordant": float(pair_g[disc].mean()) if disc.any() else float("nan"),
        "n_groups": int(b.x.shape[0]),
        "n_discordant_groups": int(disc.sum()),
        "pair_accuracy": float((correct + 0.5 * ties) / total) if total > 0 else float("nan"),
        "n_pairs": int(total),
        "const_bce": const_bce,
        "const_p": p,
    }


def _slice(b: GroupBatch, idx) -> GroupBatch:
    return GroupBatch(b.x[idx], b.delta[idx], b.y[idx], b.mask[idx])


def train_outcome_basis(
    train: GroupBatch,
    val: GroupBatch,
    rank: int,
    delta_scale: float,
    out_dir: Path,
    updates: int = 2000,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    batch_groups: int = 16,
    pair_weight: float = 1.0,
    hidden=(64, 64),
    eval_every: int = 100,
    seed: int = 0,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = train.delta.shape[-1]
    model = OutcomeModel(train.x.shape[1], n, rank, hidden, delta_scale, seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    G = train.x.shape[0]
    bs = min(batch_groups, G)
    log, best = [], (float("inf"), -1)
    for it in range(1, updates + 1):
        idx = torch.as_tensor(rng.choice(G, size=bs, replace=False))
        mb = _slice(train, idx)
        loss, *_ = group_losses(model(mb.x, mb.delta), mb, pair_weight)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % eval_every == 0 or it == updates:
            model.eval()
            rec = {"update": it, "train": evaluate(model, train, pair_weight), "val": evaluate(model, val, pair_weight)}
            model.train()
            log.append(rec)
            if rec["val"]["loss"] < best[0]:
                best = (rec["val"]["loss"], it)
                torch.save(model.state_dict(), out_dir / "predictor_best.pt")
    torch.save(model.state_dict(), out_dir / "predictor_last.pt")
    (out_dir / "train_log.json").write_text(json.dumps(log, indent=2))
    model.load_state_dict(torch.load(out_dir / "predictor_best.pt"))
    with torch.no_grad():
        b = model.basis().float().numpy()
    return {"basis": b, "best_update": best[1], "best_val_loss": best[0], "log": log}
