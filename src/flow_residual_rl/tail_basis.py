"""Fixed rank-12 "tail12" correction basis for the RL pilot (after the tail OER diagnostic).

    columns 0..5   DCT mode 0 per corrected coordinate: the mean command over the chunk, i.e. the
                   lasting displacement channel of the integrating controller
    columns 6..11  the 6 shape directions of lag_tail: zero command sum, end pose and last-step
                   change of the FITTED CONTROLLER-LAG MODEL kept, largest intermediate response

lag_tail is fitted on the fit anchors of a finished Jacobian diagnostic (task 0) and transferred
FROZEN to the pilot task (same robot / controller; no new Jacobians). Column layout matches dct12
(mode 0, then 6 more directions), so the two arms differ only in the 6 shape directions.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import effects as fx
from .basis import check_orthonormal, dct_basis, save_basis

SHAPE_STEPS = (2, 4, 6)


def tail12_from_diagnostic(diag_dir: Path) -> tuple[np.ndarray, dict]:
    diag_dir = Path(diag_dir)
    anc = json.loads((diag_dir / "anchors.json").read_text())
    vmeta = json.loads((diag_dir / "validation_meta.json").read_text())
    J = np.load(diag_dir / "jacobians.npz")["J"]
    C, nd = anc["C"], len(anc["dims"])
    rows, sc = anc["layout"]["groups"]["eef"], np.array(vmeta["scales"])
    fitn = [fx.normalise(J[i], sc) for i in vmeta["fit_idx"]]
    lag = fx.tail_basis(fx.lag_J(fx.fit_lag(fitn, nd)), rows, SHAPE_STEPS, nd)
    if lag["rank_shape"] != 6:
        raise ValueError(f"lag_tail has rank {lag['rank_shape']} != 6")
    mode0 = dct_basis(C, nd, 1).astype(np.float64)
    B = np.hstack([mode0, lag["shape"]]).astype(np.float32)
    err = check_orthonormal(B)
    dct1 = dct_basis(C, nd, 2)[:, nd:].astype(np.float64)
    meta = {"arm": "tail12", "source_diagnostic": str(diag_dir), "source_task_fit_idx": vmeta["fit_idx"],
            "layout": "cols 0-5 DCT mode 0 (mean command), cols 6-11 lag_tail shape (zero sum)",
            "lag_tail_rank_constraints": int(lag["rank_constraints"]), "orthonormality_err": err,
            "shape_overlap_with_dct_mode1": fx.overlap(lag["shape"], dct1),
            "shape_sum_leak": float(np.linalg.norm(fx.sum_operator(C, nd) @ lag["shape"])), "C": C, "n_dims": nd}
    return B, meta


def build_tail12(diag_dir: Path, out_dir: Path) -> dict:
    B, meta = tail12_from_diagnostic(diag_dir)
    save_basis(Path(out_dir), B, meta)
    return meta
