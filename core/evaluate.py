"""Patient-grouped evaluation: AUROC + bootstrap CI per method, plus the
data-efficiency learning curve that backs the small-data selling point.

Every method is fit on the training fold only:
  * bttr         -> BTTRClassifier (inductive by construction)
  * clinical_lr  -> clinical features (computed per-record) + LogisticRegression
  * cp_surrogate -> CP fit on train tensor, test projected onto fixed factors, + LR
All scalers/decompositions are fit on train rows alone.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from core import baselines as bl
from core import config
from core import preprocess as pp
from core.bttr_model import BTTRClassifier


def bootstrap_auroc_ci(y, score, n=None, seed=0):
    n = n or config.BOOTSTRAP_N
    rng = np.random.default_rng(seed)
    y, score = np.asarray(y), np.asarray(score)
    idx = np.arange(len(y))
    aucs = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) == 2:
            aucs.append(roc_auc_score(y[b], score[b]))
    if not aucs:
        return np.nan, np.nan, np.nan
    return float(np.mean(aucs)), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def _scores_for_fold(method, recs, tensor, tr, te, y, seed):
    """Return out-of-fold decision scores for the test rows of one fold."""
    if method == "bttr":
        clf = BTTRClassifier().fit(tensor[tr], y[tr])
        return clf.decision_function(tensor[te])

    if method == "clinical_lr":
        Xtr = bl.clinical_features([recs[i] for i in tr])
        Xte = bl.clinical_features([recs[i] for i in te])

    elif method == "cp_surrogate":
        Xtr, Xte = bl.cp_train_test_features(tensor[tr], tensor[te], seed=seed)

    else:
        raise ValueError(method)

    sc = StandardScaler().fit(np.nan_to_num(Xtr))
    Xtr, Xte = sc.transform(np.nan_to_num(Xtr)), sc.transform(np.nan_to_num(Xte))
    lr = LogisticRegression(max_iter=1000, random_state=seed).fit(Xtr, y[tr])
    return lr.predict_proba(Xte)[:, 1]


def evaluate_method(method, recs, y, groups, seed=0, frac=1.0, tensor=None):
    """Patient-grouped CV. `frac` subsamples each TRAIN fold (learning curve)."""
    if tensor is None:
        tensor = pp.build_tensor(recs)
    skf = StratifiedGroupKFold(n_splits=config.CV_N_SPLITS, shuffle=True, random_state=seed)
    rng = np.random.default_rng(seed)
    oof_y, oof_s = [], []
    for tr, te in skf.split(np.zeros(len(y)), y, groups):
        if frac < 1.0:
            keep = max(4, int(len(tr) * frac))
            # subsample train while keeping both classes present
            tr = _stratified_subsample(tr, y, keep, rng)
        oof_s.append(_scores_for_fold(method, recs, tensor, tr, te, y, seed))
        oof_y.append(y[te])
    oof_y, oof_s = np.concatenate(oof_y), np.concatenate(oof_s)
    auc, lo, hi = bootstrap_auroc_ci(oof_y, oof_s, seed=seed)
    return {
        "method": method, "frac": frac,
        "auroc": roc_auc_score(oof_y, oof_s),
        "auroc_boot": auc, "ci_lo": lo, "ci_hi": hi,
        "auprc": average_precision_score(oof_y, oof_s),
        "bal_acc": balanced_accuracy_score(oof_y, (oof_s >= np.median(oof_s)).astype(int)),
        "n": int(len(oof_y)), "n_pos": int(oof_y.sum()),
    }


def _stratified_subsample(tr, y, keep, rng):
    pos = tr[y[tr] == 1]; neg = tr[y[tr] == 0]
    kp = max(2, int(keep * len(pos) / len(tr)))
    kn = max(2, keep - kp)
    sel = np.concatenate([rng.choice(pos, min(kp, len(pos)), replace=False),
                          rng.choice(neg, min(kn, len(neg)), replace=False)])
    return sel
