"""Physical-effect Jacobian diagnostic (after Stage A; no new bank, no RL).

Question: is the response of the physical state to a chunk correction known in advance
(an integrator / a fixed controller-lag model, i.e. DCT-like bases suffice), or does it
depend on the state (contact, grasp) in a way that changes the useful correction subspaces?

Definitions (all corrections in env action units, flattened index j = t * n_dims + k):
  e_s(A, B)   physical difference after s executed steps (s = 1..C) between two executions
              from the same restored state; positions subtract, rotations use the world-frame
              relative rotation log(R_a R_b^T) (never axis-angle subtraction).
  J_s[:, j]   finite difference w.r.t. ONE command coordinate. Symmetric central difference
              with h = min(eta, distance to the action bound) when h >= min_frac * eta;
              otherwise an explicitly flagged one-sided difference into the feasible side.
              The executed step sizes are stored per coordinate.

Models fitted on the FIT anchors (episodes disjoint from the HELD-OUT anchors):
  integrator   J_s[:, t-block] = S          for t < s   (one S, state independent)
  lag          J_s[:, t-block] = K_{s-t}    for t < s   (fixed controller response)
  global_mean  J_s = mean over fit anchors (prediction model)
  global_gram  bases from the mean Gram matrix over fit anchors (basis model)
  local        the anchor's own measured J (oracle; needs simulator branching)

OER basis from a set of Jacobians (features normalised per group):
  B_end   = eigenvectors of mean_a J_C^T J_C with non-negligible eigenvalue (effective rank
            reported; never padded with arbitrary null-space vectors)
  B_shape = top eigenvectors of Q^T (mean_a sum_{s in shape_steps} J_s^T J_s) Q mapped back by Q,
            Q = explicit orthonormal complement of B_end, so B_end^T B_shape = 0 always.
"""
from __future__ import annotations

import numpy as np

from .basis import dct_basis

GROUPS = ("eef_pos", "eef_rot", "grip", "obj_pos_task", "obj_rot_task", "obj_pos_other", "obj_rot_other")

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


def feature_layout(n_grip: int, obj_names, task_objects=()) -> dict:
    """Feature names and index groups of an effect vector. Objects named in `task_objects`
    (LIBERO obj_of_interest) are grouped separately from the rest of the scene."""
    names = [f"eef_p{a}" for a in "xyz"] + [f"eef_r{a}" for a in "xyz"] + [f"grip_q{i}" for i in range(n_grip)]
    for n in obj_names:
        names += [f"{n}_p{a}" for a in "xyz"] + [f"{n}_r{a}" for a in "xyz"]
    idx = {k: i for i, k in enumerate(names)}
    task_objects = [n for n in obj_names if n in set(task_objects)]
    other = [n for n in obj_names if n not in set(task_objects)]
    g = {"eef_pos": [0, 1, 2], "eef_rot": [3, 4, 5], "grip": list(range(6, 6 + n_grip)),
         "obj_pos_task": [idx[f"{n}_p{a}"] for n in task_objects for a in "xyz"],
         "obj_rot_task": [idx[f"{n}_r{a}"] for n in task_objects for a in "xyz"],
         "obj_pos_other": [idx[f"{n}_p{a}"] for n in other for a in "xyz"],
         "obj_rot_other": [idx[f"{n}_r{a}"] for n in other for a in "xyz"]}
    g["eef"] = g["eef_pos"] + g["eef_rot"]
    g["full"] = g["eef"] + g["grip"] + g["obj_pos_task"] + g["obj_rot_task"]  # basis rows of the object-aware oracle
    return {"names": names, "groups": g, "obj_names": list(obj_names), "task_objects": task_objects}


def effect(p: dict, q: dict) -> np.ndarray:
    """Physical state p relative to q (same layout)."""
    parts = [p["eef_pos"] - q["eef_pos"], rotvec(p["eef_mat"] @ q["eef_mat"].T), p["grip_qpos"] - q["grip_qpos"]]
    for k in range(len(p["obj_pos"])):
        parts += [p["obj_pos"][k] - q["obj_pos"][k], rotvec(p["obj_mat"][k] @ q["obj_mat"][k].T)]
    return np.concatenate(parts).astype(np.float64)


def max_state_diff(xs, ys) -> float:
    """Max |effect| over the common steps of two state sequences (restore checks)."""
    n = min(len(xs), len(ys))
    return float(max((np.abs(effect(xs[i], ys[i])).max() for i in range(n)), default=0.0))


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
            "obj_names": names, "task_objects": list(getattr(e, "obj_of_interest", []) or []),
            "obj_pos": np.array([sim.data.body_xpos[i] for i in ids], np.float64).reshape(-1, 3),
            "obj_mat": np.array([sim.data.body_xmat[i] for i in ids], np.float64).reshape(-1, 3, 3)}


def interaction_flags(task) -> dict:
    """Grasp / contact of the gripper with every object. Failures are reported, never hidden."""
    e = task.env.env
    g = e.robots[0].gripper
    grasp, contact, errors = {}, {}, []
    for n, obj in e.objects_dict.items():
        try:
            grasp[n] = bool(e._check_grasp(gripper=g, object_geoms=obj))
        except Exception as ex:  # noqa: BLE001
            grasp[n] = None
            errors.append(f"grasp {n}: {ex!r}")
        try:
            contact[n] = bool(e.check_contact(g, obj))
        except Exception as ex:  # noqa: BLE001
            contact[n] = None
            errors.append(f"contact {n}: {ex!r}")
    return {"grasp": grasp, "contact": contact, "errors": errors}


def phase_from(flags: dict, phys: dict, near_m: float = 0.08) -> dict:
    """grasp > contact > near > free; 'unknown' if any interaction check failed."""
    d = np.linalg.norm(phys["obj_pos"] - phys["eef_pos"][None], axis=1)
    k = int(np.argmin(d)) if len(d) else -1
    if flags["errors"]:
        ph = "unknown"
    elif any(flags["grasp"].values()):
        ph = "grasp"
    elif any(flags["contact"].values()):
        ph = "contact"
    elif len(d) and d[k] < near_m:
        ph = "near"
    else:
        ph = "free"
    return {"phase": ph, **flags, "nearest_object": phys["obj_names"][k] if k >= 0 else None,
            "nearest_dist": float(d[k]) if k >= 0 else None}


def phase_label(task, phys: dict, near_m: float = 0.08) -> dict:
    return phase_from(interaction_flags(task), phys, near_m)


def flag_changes(a: dict, b: dict) -> dict:
    """Objects whose grasp / contact flag differs between two flag sets."""
    return {k: sorted(n for n in a[k] if a[k].get(n) != b[k].get(n)) for k in ("grasp", "contact")}


# ----------------------------------------------------------------------------- measurement


def execute(task, snap, actions, restore, measure) -> tuple[list, int]:
    """Restore `snap`, execute all of `actions` [T, A] (fixed horizon, also after success);
    physical state after every step and the first step (1-based) with success, -1 if none."""
    restore(task, snap)
    out, first = [], -1
    for i, a in enumerate(actions):
        task.step(a)
        out.append(measure(task))
        if first < 0 and task.success:
            first = i + 1
    return out, first


def fd_jacobian(task, snap, a0, dims, eta, restore, measure, low, high, nominal, min_frac: float = 0.25):
    """Finite-difference Jacobian J[s-1, F, C*nd] of the effect after s steps.

    nominal: physical states of the unmodified chunk (for one-sided differences).
    info: per coordinate step size h, kind (0 central, +1 forward, -1 backward), success step of
    every branch, and the causality check max |J_s[:, t-block]| over t >= s (must be ~0)."""
    C, nd = a0.shape[0], len(dims)
    N = C * nd
    F = len(effect(nominal[0], nominal[0]))
    J = np.full((C, F, N), np.nan)
    h_used, kind, succ = np.zeros(N), np.zeros(N, int), []
    for t in range(C):
        for ki, k in enumerate(dims):
            j = t * nd + ki
            up, dn = float(high[k] - a0[t, k]), float(a0[t, k] - low[k])
            room = min(up, dn)
            if room >= min_frac * eta:
                h = min(eta, room)
                ap, am = a0.copy(), a0.copy()
                ap[t, k] += h
                am[t, k] -= h
                (sp, fp), (sm, fm) = execute(task, snap, ap, restore, measure), execute(task, snap, am, restore, measure)
                for s in range(C):
                    J[s, :, j] = effect(sp[s], sm[s]) / (2 * h)
                succ += [fp, fm]
            else:
                sign = 1 if up >= dn else -1
                h = min(eta, up if sign > 0 else dn)
                if h < 1e-6:
                    continue
                ab = a0.copy()
                ab[t, k] += sign * h
                sb, fb = execute(task, snap, ab, restore, measure)
                for s in range(C):
                    J[s, :, j] = sign * effect(sb[s], nominal[s]) / h
                succ.append(fb)
                kind[j] = sign
            h_used[j] = h
    caus = max((float(np.abs(np.nan_to_num(J[s - 1][:, t * nd:(t + 1) * nd])).max()) for s in range(1, C + 1)
                for t in range(s, C)), default=0.0)
    return J, {"h": h_used, "kind": kind, "branch_success_steps": succ, "causality_max": caus,
               "n_one_sided": int(np.sum(kind != 0)), "n_unmeasured": int(np.isnan(J[-1, 0]).sum())}


# ----------------------------------------------------------------------------- models


def group_scales(Js, layout, step: int = -1, floor: float = 1e-9) -> np.ndarray:
    """Per-feature scale: RMS response of the feature's group at `step` over the given Jacobians.
    Uniform inside a group (rotation invariant); unresponsive groups get scale 1."""
    F = Js[0].shape[1]
    sc = np.ones(F)
    stack = np.stack([J[step] for J in Js])  # [A, F, N]
    for g in GROUPS:
        rows = layout["groups"].get(g) or []
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


def _err_parts(J_pred, J_true, rows, step):
    a, b = J_pred[step][rows], J_true[step][rows]
    m = np.isfinite(b) & np.isfinite(a)
    return float(np.sum((a[m] - b[m]) ** 2)), float(np.sum(b[m] ** 2)), int(m.sum())


def rel_error(J_pred, J_true, rows, step: int) -> float:
    """||pred - true|| / ||true||; a non-zero prediction of a zero response is inf, not NaN."""
    num, den, _ = _err_parts(J_pred, J_true, rows, step)
    if den > 0:
        return float(np.sqrt(num / den))
    return 0.0 if num == 0 else float("inf")


def pooled_errors(J_pred, J_trues, rows, step: int) -> dict:
    """Errors pooled over anchors: sqrt(sum num / sum den) (robust to anchors with zero response)
    and the absolute RMS error per matrix entry (in the units of the rows)."""
    parts = [_err_parts(J_pred, J, rows, step) for J in J_trues]
    num, den, n = (sum(p[i] for p in parts) for i in range(3))
    return {"pooled_rel": (float(np.sqrt(num / den)) if den > 0 else (0.0 if num == 0 else float("inf"))),
            "abs_rms": float(np.sqrt(num / n)) if n else None, "true_rms": float(np.sqrt(den / n)) if n else None}


# ----------------------------------------------------------------------------- bases


def _eig_desc(G):
    w, v = np.linalg.eigh((G + G.T) / 2.0)
    o = np.argsort(w)[::-1]
    v = v[:, o]
    idx = np.argmax(np.abs(v), axis=0)
    return w[o], v * np.sign(v[idx, np.arange(v.shape[1])])[None, :]


def complement(B: np.ndarray, N: int) -> np.ndarray:
    """Orthonormal basis of the orthogonal complement of span(B) in R^N."""
    if B.shape[1] == 0:
        return np.eye(N)
    u, _, _ = np.linalg.svd(B, full_matrices=True)
    return u[:, B.shape[1]:]


def oer_basis(Js, rows, end_step: int, shape_steps, k_end: int = 6, k_shape: int = 6, rel_tol: float = 1e-8) -> dict:
    """B_end / B_shape from one or several (normalised) Jacobians.

    Only measured (finite) entries enter; unmeasured columns are counted, not treated as zero
    response. Eigenvalues below rel_tol * (largest eigenvalue of the shape Gram) are not
    physical directions: they are excluded and the effective ranks are reported."""
    N = Js[0].shape[2]
    sel = [J[:, rows, :] for J in Js]
    n_unmeasured = int(sum(np.isnan(J[-1]).any(axis=0).sum() for J in sel))
    sel = [np.nan_to_num(J) for J in sel]
    G_end = np.mean([J[end_step - 1].T @ J[end_step - 1] for J in sel], axis=0)
    G_sh = np.mean([sum(J[s - 1].T @ J[s - 1] for s in shape_steps) for J in sel], axis=0)
    scale = max(float(np.linalg.eigvalsh((G_end + G_end.T) / 2).max(initial=0)),
                float(np.linalg.eigvalsh((G_sh + G_sh.T) / 2).max(initial=0)))
    tol = rel_tol * scale if scale > 0 else np.inf
    w_end, v_end = _eig_desc(G_end)
    r_end = int(min(k_end, np.sum(w_end > tol)))
    B_end = v_end[:, :r_end]
    Q = complement(B_end, N)
    w_sh, v_sh = _eig_desc(Q.T @ G_sh @ Q)
    r_sh = int(min(k_shape, np.sum(w_sh > tol)))
    B_shape = Q @ v_sh[:, :r_sh]
    return {"end": B_end, "shape": B_shape, "ev_end": w_end[:k_end], "ev_shape": w_sh[:k_shape],
            "rank_end": r_end, "rank_shape": r_sh, "n_unmeasured_cols": n_unmeasured}


def dct_oer(C: int, nd: int) -> dict:
    b = dct_basis(C, nd, 2).astype(np.float64)
    return {"end": b[:, :nd], "shape": b[:, nd:2 * nd], "rank_end": nd, "rank_shape": nd}


def overlap(B1, B2) -> float | None:
    """||B1^T B2||_F^2 / min(k1, k2): 1 = one subspace contains the other, ~k/N for random."""
    if B1.shape[1] == 0 or B2.shape[1] == 0:
        return None
    return float(np.sum((B1.T @ B2) ** 2) / min(B1.shape[1], B2.shape[1]))


def principal_angles_deg(B1, B2) -> list:
    if B1.shape[1] == 0 or B2.shape[1] == 0:
        return []
    s = np.clip(np.linalg.svd(B1.T @ B2, compute_uv=False), -1.0, 1.0)
    return [float(np.degrees(np.arccos(x))) for x in s]


def project_direction(B: np.ndarray, v: np.ndarray) -> np.ndarray | None:
    """Unit vector along the projection of a common random v onto span(B) (basis-rotation invariant)."""
    if B.shape[1] == 0:
        return None
    p = B @ (B.T @ v)
    n = np.linalg.norm(p)
    return p / n if n > 1e-12 else None


def intervention_metrics(e_steps: np.ndarray, layout: dict, scales, shape_steps, end_step: int) -> dict:
    """e_steps [T, F] effects vs the nominal execution after 1..T steps."""
    rows = layout["groups"]["eef"]
    z = e_steps[:, rows] / scales[rows][None]
    out = {"endpoint_drift": float(np.linalg.norm(z[end_step - 1])),
           "shape_change": float(np.sqrt(sum(np.sum(z[s - 1] ** 2) for s in shape_steps)))}
    if e_steps.shape[0] > end_step:
        out["drift_after_continuation"] = float(np.linalg.norm(z[-1]))
    G = layout["groups"]
    for g in ("obj_pos_task", "obj_pos_other", "grip"):
        for name, s in (("end", end_step - 1), ("cont", e_steps.shape[0] - 1)):
            out[f"{g}_absmax_{name}"] = float(np.abs(e_steps[s][G[g]]).max(initial=0.0))
    out["eef_pos_m_end"] = float(np.linalg.norm(e_steps[end_step - 1][G["eef_pos"]]))
    return out
