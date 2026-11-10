import numpy as np
import pytest
import torch

from flow_residual_rl.basis import (ResidualMap, build_fixed_basis, check_orthonormal, dct_basis, pca_basis,
                                    random_basis, thin_qr)
from flow_residual_rl.budget import BudgetExceeded, StepCounter
from flow_residual_rl.outcome_predictor import GroupBatch, group_losses, torch_thin_qr, train_outcome_basis
from flow_residual_rl.providers.gr00t import model_to_env

C, ND, D, EPS = 8, 6, 12, 0.10
N = C * ND
LO, HI = -np.ones(7, np.float32), np.ones(7, np.float32)


def bases():
    rng = np.random.default_rng(0)
    return {"random12": random_basis(N, D, 1), "dct12": dct_basis(C, ND, 2), "full48": np.eye(N, dtype=np.float32),
            "pca12": pca_basis(rng.standard_normal((200, N)), D), "qr": thin_qr(rng.standard_normal((N, D)))}


@pytest.mark.parametrize("name", ["random12", "dct12", "full48", "pca12", "qr"])
def test_orthonormal_and_rms_bound(name):
    b = bases()[name]
    assert b.dtype == np.float32 and b.shape[0] == N
    check_orthonormal(b)
    rm = ResidualMap(b, C, tuple(range(6)), EPS, LO, HI)
    rng = np.random.default_rng(1)
    for _ in range(500):
        u = rng.uniform(-1, 1, rm.dim) * rng.uniform(0, 3)
        rms = np.sqrt(np.mean(rm.delta(u) ** 2))
        w = np.linalg.norm(u) / max(1.0, np.linalg.norm(u))
        assert rms <= EPS * (1 + 1e-5)
        np.testing.assert_allclose(rms, EPS * w, rtol=1e-4)  # same physical RMS for any rank


def test_zero_u_is_bit_exact_including_gripper():
    a0 = model_to_env(np.random.default_rng(0).uniform(-1.5, 1.5, (C, 7)).astype(np.float32))
    for b in bases().values():
        rm = ResidualMap(b, C, tuple(range(6)), EPS, LO, HI)
        out, st = rm.apply(a0, np.zeros(rm.dim, np.float32))
        assert np.array_equal(out, a0) and out.dtype == a0.dtype and st["rms_actual"] == 0


def test_gripper_untouched_and_clip():
    a0 = np.full((C, 7), 0.99, np.float32)
    a0[:, 6] = -1
    rm = ResidualMap(bases()["full48"], C, tuple(range(6)), EPS, LO, HI)
    out, st = rm.apply(a0, np.ones(N, np.float32))
    assert np.array_equal(out[:, 6], a0[:, 6])
    assert out[:, :6].max() <= 1.0 and st["clip_frac"] > 0 and st["rms_actual"] < st["rms_planned"]


def test_dct_modes():
    b = dct_basis(C, ND, 2)
    assert np.allclose(b[:, :ND].reshape(C, ND, ND)[:, 0, 0], 1 / np.sqrt(C))  # mode 0 constant
    assert b.shape == (N, D)


def test_model_to_env_gripper_convention():
    raw = np.zeros((2, 7), np.float32)
    raw[0, 6], raw[1, 6] = 0.9, 0.1  # model: 1=open? -> env: -sign(2g-1)
    out = model_to_env(raw)
    assert out[0, 6] == -1 and out[1, 6] == 1


def test_torch_qr_matches_numpy():
    w = np.random.default_rng(3).standard_normal((N, D)).astype(np.float32)
    np.testing.assert_allclose(torch_thin_qr(torch.as_tensor(w)).numpy(), thin_qr(w), atol=1e-5)


def test_budget_hard_limit_and_buckets(tmp_path):
    c = StepCounter(tmp_path / "cost.json")
    c.set_limit(("bank_pool", "bank_branch"), 10)
    with c.charge("bank_pool"):
        for _ in range(6):
            c.on_env_step()
    with c.charge("bank_branch"):
        for _ in range(4):
            c.on_env_step()
        with pytest.raises(BudgetExceeded):
            c.on_env_step()
    with c.charge("evaluation"):
        c.on_env_step()  # unrelated bucket not limited
    c.save()
    c2 = StepCounter(tmp_path / "cost.json")
    assert c2.steps["bank_pool"] == 6 and c2.steps["bank_branch"] == 4 and c2.total() == 11


def test_pairwise_loss_and_group_weighting():
    s = torch.tensor([[2.0, -2.0, 0.0]])
    y = torch.tensor([[1.0, 0.0, 0.0]])
    b = GroupBatch(torch.zeros(1, 1), torch.zeros(1, 3, N), y, torch.ones(1, 3, dtype=torch.bool))
    loss, bce, pair, npairs = group_losses(s, b, 1.0)
    assert int(npairs) == 2
    expect_pair = (torch.nn.functional.softplus(torch.tensor(-4.0)) + torch.nn.functional.softplus(torch.tensor(-2.0))) / 2
    assert torch.allclose(pair[0], expect_pair)


def test_outcome_basis_recovers_planted_direction(tmp_path):
    """Synthetic sanity: success depends on the projection of delta on a planted 2-D subspace."""
    rng = np.random.default_rng(0)
    planted = thin_qr(rng.standard_normal((N, 2)))
    G = 400
    scale = EPS * np.sqrt(N)
    x = torch.zeros(G, 4)
    dirs = rng.standard_normal((G, N))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    half = rng.random(G) < 0.5
    dirs[half] = (planted @ rng.standard_normal((2, half.sum()))).T
    dirs[half] /= np.linalg.norm(dirs[half], axis=1, keepdims=True)
    delta = np.stack([np.zeros((G, N)), dirs * scale, -dirs * scale], 1)
    score = (delta @ planted)[..., 0] / scale
    y = (score > 0.3).astype(np.float32)
    batch = GroupBatch(x, torch.as_tensor(delta, dtype=torch.float32), torch.as_tensor(y), torch.ones(G, 3, dtype=torch.bool))
    tr = GroupBatch(*(t[:300] for t in (batch.x, batch.delta, batch.y, batch.mask)))
    va = GroupBatch(*(t[300:] for t in (batch.x, batch.delta, batch.y, batch.mask)))
    res = train_outcome_basis(tr, va, D, scale, tmp_path, updates=1500, eval_every=250)
    b = res["basis"]
    check_orthonormal(b)
    captured = np.linalg.norm(b.T @ planted[:, :1]) ** 2  # fraction of the planted direction inside span(B)
    rand = np.linalg.norm(random_basis(N, D, 5).T @ planted[:, :1]) ** 2
    assert captured > 0.8 and captured > rand + 0.3, (captured, rand)
    assert res["log"][-1]["val"]["pair_accuracy"] > 0.8
