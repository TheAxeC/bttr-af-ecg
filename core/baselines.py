"""Baselines: the clinical yardstick and the thesis's CP-features surrogate.

`clinical_lr`  - interpretable RR/P-wave/QRS features + logistic regression. For
                 SR-vs-AF this should sit near the achievable ceiling from a handful
                 of numbers; it is what BTTR's interpretable blocks are measured against.
`cp_surrogate` - exactly what the thesis did: fit one CP/PARAFAC on the TRAINING
                 tensor, use the sample-mode factor as features, logistic regression.
                 Made inductive here (CP fit on train; test projected onto the fixed
                 factors) so the comparison to BTTR is fair rather than leaky.
"""
from __future__ import annotations

import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac

from core import config
from core import preprocess as pp


# --------------------------------------------------------------------------- #
# Clinical features                                                           #
# --------------------------------------------------------------------------- #
def _clinical_one(rec):
    lead = config.LEAD_NAMES.index(config.RHYTHM_LEAD)
    sig = pp.clean(rec.signal, rec.fs, decimate_factor=1)
    fs = rec.fs
    peaks = pp.detect_rpeaks(sig[:, lead], fs)
    if len(peaks) < 3:
        return [0, 0, 0, 0, 0, 0, float(np.median(np.ptp(sig, 0)))]
    rr = np.diff(peaks) / fs
    drr = np.diff(rr)
    # P-wave energy proxy in the PR segment relative to QRS
    a, bb, q = int(0.20 * fs), int(0.08 * fs), int(0.05 * fs)
    pr, qrs = [], []
    for r in peaks:
        if r - a >= 0 and r + q <= len(sig):
            pr.append(np.sum(sig[r - a:r - bb, lead] ** 2))
            qrs.append(np.sum(sig[r - q:r + q, lead] ** 2) + 1e-9)
    pwave = float(np.mean(pr) / np.mean(qrs)) if pr else 0.0
    return [rr.mean(), rr.std(), np.sqrt(np.mean(drr ** 2)),      # rr_mean, rr_std, rmssd
            rr.std() / rr.mean() if rr.mean() else 0,             # rr_cv
            float(np.mean(np.abs(drr) > 0.05)),                   # pNN50
            pwave, float(np.median(np.ptp(sig, 0)))]              # pwave proxy, qrs amp


def clinical_features(recordings):
    return np.array([_clinical_one(r) for r in recordings], dtype=float)


# --------------------------------------------------------------------------- #
# CP-features surrogate (the thesis pipeline, made inductive)                  #
# --------------------------------------------------------------------------- #
def _fit_cp(tensor, rank, seed):
    (w, factors) = parafac(tl.tensor(tensor), rank=rank, n_iter_max=100, tol=1e-7,
                           init="svd", svd="truncated_svd", normalize_factors=True,
                           l2_reg=1e-3, random_state=seed)
    A, B, C = factors
    return A * w[None, :], B, C


def cp_train_test_features(train_tensor, test_tensor, rank=8, seed=0):
    """Fit CP on TRAIN only; project TEST onto the fixed (B,C) factors via
    Khatri-Rao least squares (so the surrogate is inductive, not leaky)."""
    A, B, C = _fit_cp(train_tensor, rank, seed)
    T, R = B.shape
    L = C.shape[0]
    kr = np.einsum("lr,tr->tlr", C, B).reshape(T * L, R)
    pinv = np.linalg.pinv(kr)
    Xtr = train_tensor.reshape(len(train_tensor), T * L) @ pinv.T   # == A up to lstsq
    Xte = test_tensor.reshape(len(test_tensor), T * L) @ pinv.T
    return Xtr, Xte
