"""Physical-effect Jacobian diagnostic (after Stage A; no new bank, no RL).

Question: is the response of the physical state to a chunk correction known in advance
(an integrator / a fixed controller-lag model, i.e. DCT-like bases suffice), or does it
depend on the state (contact, grasp) in a way that changes the useful correction subspaces?

Definitions (all corrections in env action units, flattened index j = t * n_dims + k):
  e_s(A)      physical difference after s executed steps (s = 1..C) between two executions
              from the same restored state; positions subtract, rotations use the world-frame
              relative rotation log(R_a R_b^T) (never axis-angle subtraction).
  J_s[:, j]   = (e_s(A0 + eta e_j) - e_s(A0 - eta e_j)) / (actual_plus - actual_minus)
              central difference; `actual` is the executed change after clipping.

Models fitted on the FIT anchors (episodes disjoint from the HELD-OUT anchors):
  integrator  J_s[:, t-block] = S          for t < s   (one S, state independent)
  lag         J_s[:, t-block] = K_{s-t}    for t < s   (fixed controller response)
  global      J_s = mean over fit anchors
  local       the anchor's own measured J (oracle; needs simulator branching)

OER basis from a set of Jacobians (features normalised per group):
  B_end   = top-k eigenvectors of mean_a J_C^T J_C
  B_shape = top-k eigenvectors of P (mean_a sum_{s in shape_steps} J_s^T J_s) P,  P = I - B_end B_end^T
"""
from __future__ import annotations

import numpy as np

from .basis import dct_basis

# ----------------------------------------------------------------------------- geometry


def rotvec(R: np.ndarray) -> np.ndarray:
    """SO(3) log map: rotation vector (axis * angle, radians) of a rotation matrix."""
    R = np.asarray(R, np.float64).reshape(3, 3)
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    th = float(np.arccos(c))
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th < 1e-6:
        return 0.5 * w
    if np.pi - th < 1e-4:
        a = (R + np.eye(3)) / 2.0
        k = int(np.argmax(np.diag(a)))
        ax = a[:, k] / np.sqrt(max(a[k, k], 1e-12))
        return th * ax / np.linalg.norm(ax)
    return th / (2.0 * np.sin(th)) * w


def feature_layout(n_grip: int, obj_names) -> dict:
    """Feature names and index groups of an effect vector."""
    names = [f"eef_p{a}" for a in "xyz"] + [f"eef_r{a}" for a in "xyz"] + [f"grip_q{i}" for i in range(n_grip)]
    for n in obj_names:
        names += [f"{n}_p{a}" for a in "xyz"] + [f"{n}_r{a}" for a in "xyz"]
    idx = {k: i for i, k in enumerate(names)}
    groups = {"eef_pos": [0, 1, 2], "eef_rot": [3, 4, 5], "grip": list(range(6, 6 + n_grip)),
              "obj_pos": [idx[f"{n}_p{a}"] for n in obj_names for a in "xyz"],
              "obj_rot": [idx[f"{n}_r{a}"] for n in obj_names for a in "xyz"]}
    groups["eef"] = groups["eef_pos"] + groups["eef_rot"]
    return {"names": names, "groups": groups, "obj_names": list(obj_names)}


def effect(p: dict, q: dict) -> np.ndarray:
    """Physical state p relative to q (same layout)."""
    parts = [p["eef_pos"] - q["eef_pos"], rotvec(p["eef_mat"] @ q["eef_mat"].T), p["grip_qpos"] - q["grip_qpos"]]
    for k in range(len(p["obj_pos"])):
        parts += [p["obj_pos"][k] - q["obj_pos"][k], rotvec(p["obj_mat"][k] @ q["obj_mat"][k].T)]
    return np.concatenate(parts).astype(np.float64)


# ----------------------------------------------------------------------------- simulator readers (LIBERO)


def read_physical(task) -> dict:
    """Ground-truth physical state from MuJoCo (not the sampled robosuite observables).

    Kinematic quantities are as of the last substep's mj_forward (robosuite calls forward
    before every substep), identical convention for every branch. Nothing is written to sim."""
    e = task.env.env
    sim, r = e.sim, e.robots[0]
    names = sorted(e.objects_dict)
    ids = [e.obj_body_id[n] for n in names]
    return {"eef_pos": np.array(sim.data.site_xpos[r.eef_site_id], np.float64),
            "eef_mat": np.array(sim.data.site_xmat[r.eef_site_id], np.float64).reshape(3, 3),
            "grip_qpos": np.array(sim.data.qpos[r._ref_gripper_joint_pos_indexes], np.float64),
            "obj_names": names,
            "obj_pos": np.array([sim.data.body_xpos[i] for i in ids], np.float64).reshape(-1, 3),
            "obj_mat": np.array([sim.data.body_xmat[i] for i in ids], np.float64).reshape(-1, 3, 3)}


def phase_label(task, phys: dict, near_m: float = 0.08) -> dict:
    """Interaction phase at a chunk boundary: grasp > contact > near > free."""
    e = task.env.env
    g = e.robots[0].gripper
    grasp, contact = {}, {}
    for n, obj in e.objects_dict.items():
        try:
            grasp[n] = bool(e._check_grasp(gripper=g, object_geoms=obj))
        except Exception:  # noqa: BLE001
            grasp[n] = None
        try:
            contact[n] = bool(e.check_contact(g, obj))
        except Exception:  # noqa: BLE001
            contact[n] = None
    d = np.linalg.norm(phys["obj_pos"] - phys["eef_pos"][None], axis=1)
    k = int(np.argmin(d)) if len(d) else -1
    if any(grasp.values()):
        ph = "grasp"
    elif any(contact.values()):
        ph = "contact"
    elif len(d) and d[k] < near_m:
        ph = "near"
    else:
        ph = "free"
    return {"phase": ph, "grasp": grasp, "contact": contact,
            "nearest_object": phys["obj_names"][k] if k >= 0 else None, "nearest_dist": float(d[k]) if k >= 0 else None}


# ----------------------------------------------------------------------------- measurement


def execute(task, snap, actions, restore, measure) -> list:
    """Restore `snap`, execute `actions` [T, A]; physical state after every step."""
    restore(task, snap)
    out = []
    for a in actions:
        task.step(a)
        out.append(measure(task))
    return out


def fd_jacobian(task, snap, a0, dims, eta, restore, measure, low, high) -> tuple[np.ndarray, dict]:
    """Central-difference Jacobian J[s-1, F, C*nd] of the effect after s steps w.r.t. single
    command coordinates; NaN columns where clipping left no usable difference."""
    C, nd = a0.shape[0], len(dims)
    J, actual, n_clipped = None, np.zeros((C * nd, 2)), 0
    for t in range(C):
        for ki, k in enumerate(dims):
            j = t * nd + ki
            ap, am = a0.copy(), a0.copy()
            ap[t, k] = np.clip(a0[t, k] + eta, low[k], high[k])
            am[t, k] = np.clip(a0[t, k] - eta, low[k], high[k])
            dp, dm = float(ap[t, k] - a0[t, k]), float(am[t, k] - a0[t, k])
            actual[j] = (dp, dm)
            n_clipped += int(abs(dp - eta) > 1e-7 or abs(dm + eta) > 1e-7)
            sp, sm = execute(task, snap, ap, restore, measure), execute(task, snap, am, restore, measure)
            if J is None:
                J = np.full((C, len(effect(sp[0], sm[0])), C * nd), np.nan)
            if dp - dm < 1e-6:
                continue
            for s in range(C):
                J[s, :, j] = effect(sp[s], sm[s]) / (dp - dm)
    return J, {"actual": actual, "n_clipped": n_clipped}


# ----------------------------------------------------------------------------- models


def group_scales(Js, layout, step: int = -1, floor: float = 1e-9) -> np.ndarray:
    """Per-feature scale: RMS response of the feature's group at `step` over the given Jacobians.
    Uniform inside a group (rotation invariant); unresponsive groups get scale 1."""
    F = Js[0].shape[1]
    sc = np.ones(F)
    stack = np.stack([J[step] for J in Js])  # [A, F, N]
    for g in ("eef_pos", "eef_rot", "grip", "obj_pos", "obj_rot"):
        rows = layout["groups"][g]
        if not rows:
            continue
        v = np.sqrt(np.nanmean(stack[:, rows, :] ** 2))
        sc[rows] = v if np.isfinite(v) and v > floor else 1.0
    return sc


def normalise(J, scales):
    return J / scales[None, :, None]


def fit_integrator(Js, nd: int) -> np.ndarray:
    C = Js[0].shape[0]
    blocks = [J[s - 1][:, t * nd:(t + 1) * nd] for J in Js for s in range(1, C + 1) for t in range(s)]
    return np.nanmean(np.stack(blocks), axis=0)  # [F, nd]


def fit_lag(Js, nd: int) -> np.ndarray:
    C = Js[0].shape[0]
    K = []
    for lag in range(1, C + 1):
        blocks = [J[s - 1][:, t * nd:(t + 1) * nd] for J in Js for s in range(1, C + 1) for t in range(s) if s - t == lag]
        K.append(np.nanmean(np.stack(blocks), axis=0))
    return np.stack(K)  # [C, F, nd], K[lag-1]


def integrator_J(S: np.ndarray, C: int) -> np.ndarray:
    F, nd = S.shape
    J = np.zeros((C, F, C * nd))
    for s in range(1, C + 1):
        for t in range(s):
            J[s - 1][:, t * nd:(t + 1) * nd] = S
    return J


def lag_J(K: np.ndarray) -> np.ndarray:
    C, F, nd = K.shape
    J = np.zeros((C, F, C * nd))
    for s in range(1, C + 1):
        for t in range(s):
            J[s - 1][:, t * nd:(t + 1) * nd] = K[s - t - 1]
    return J


def rel_error(J_pred, J_true, rows, step: int) -> float:
    a, b = J_pred[step][rows], J_true[step][rows]
    m = np.isfinite(b) & np.isfinite(a)
    den = np.sqrt(np.sum(b[m] ** 2))
    return float(np.sqrt(np.sum((a[m] - b[m]) ** 2)) / den) if den > 0 else float("nan")


# ----------------------------------------------------------------------------- bases


def _top_eig(G: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    G = (G + G.T) / 2.0
    w, v = np.linalg.eigh(G)
    o = np.argsort(w)[::-1][:k]
    v = v[:, o]
    idx = np.argmax(np.abs(v), axis=0)
    v = v * np.sign(v[idx, np.arange(v.shape[1])])[None, :]
    return v, w[o]


def oer_basis(Js, rows, end_step: int, shape_steps, k_end: int = 6, k_shape: int = 6) -> dict:
    """B_end / B_shape from one or several (normalised) Jacobians; NaN columns count as 0."""
    N = Js[0].shape[2]
    sel = [np.nan_to_num(J[:, rows, :]) for J in Js]
    G_end = np.mean([J[end_step - 1].T @ J[end_step - 1] for J in sel], axis=0)
    B_end, ev_end = _top_eig(G_end, k_end)
    P = np.eye(N) - B_end @ B_end.T
    G_sh = np.mean([sum(J[s - 1].T @ J[s - 1] for s in shape_steps) for J in sel], axis=0)
    B_shape, ev_sh = _top_eig(P @ G_sh @ P, k_shape)
    return {"end": B_end, "shape": B_shape, "ev_end": ev_end, "ev_shape": ev_sh}


def dct_oer(C: int, nd: int) -> dict:
    b = dct_basis(C, nd, 2).astype(np.float64)
    return {"end": b[:, :nd], "shape": b[:, nd:2 * nd]}


def overlap(B1, B2) -> float:
    """||B1^T B2||_F^2 / min(k1, k2): 1 = one subspace contains the other, ~k/N for random."""
    return float(np.sum((B1.T @ B2) ** 2) / min(B1.shape[1], B2.shape[1]))


def principal_angles_deg(B1, B2) -> list:
    s = np.clip(np.linalg.svd(B1.T @ B2, compute_uv=False), -1.0, 1.0)
    return [float(np.degrees(np.arccos(x))) for x in s]


def intervention_metrics(e_steps: np.ndarray, rows, scales, shape_steps, end_step: int) -> dict:
    """e_steps [T, F] effects vs the nominal execution after 1..T steps (normalised units)."""
    z = e_steps[:, rows] / scales[rows][None]
    end = float(np.linalg.norm(z[end_step - 1]))
    shape = float(np.sqrt(sum(np.sum(z[s - 1] ** 2) for s in shape_steps)))
    out = {"endpoint_drift": end, "shape_change": shape}
    if e_steps.shape[0] > end_step:
        out["drift_after_continuation"] = float(np.linalg.norm(z[-1]))
    return out
