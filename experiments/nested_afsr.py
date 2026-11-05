"""Nested-CV, honest 'tune as hard as possible' estimate for AF-in-SR.

The flat K sweep in harden_afsr.py is optimistically biased (K chosen by looking at
the same CV it is scored on). Here K is chosen by an INNER CV on each outer-train
fold only; the OUTER CV scores the tuned pipeline on never-touched test rows. This
is the unbiased estimate of 'BTTR with K tuned per fold'. Tensorisation is fixed a
priori to morphology-only median_beat (RR irregularity is absent in SR by
construction). Multi-seed varies outer folds + the healthy-control draw.

Also computes a paired BTTR-vs-clinical comparison: per seed, AUROC(bttr) -
AUROC(clinical) on identical folds, then a Wilcoxon signed-rank test across seeds.
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from core import baselines as bl
from core import config
from core import preprocess as pp
from core.bttr_model import BTTRClassifier
from experiments.harden_afsr import _afsr_frames, _build_recs, load_db

N_SEEDS = 10
K_GRID = (6, 8, 12, 16)
OUTER, INNER = 5, 3


def _bttr_oof(tensor, y, groups, seed):
    """Outer CV with inner-CV K selection. Returns pooled (y, score) over outer test."""
    outer = StratifiedGroupKFold(OUTER, shuffle=True, random_state=seed)
    oy, os_ = [], []
    for tr, te in outer.split(np.zeros(len(y)), y, groups):
        # inner CV to pick K on the outer-train rows only
        gtr = groups[tr]
        best_k, best_auc = K_GRID[0], -1.0
        inner = StratifiedGroupKFold(INNER, shuffle=True, random_state=seed)
        for K in K_GRID:
            iy, is_ = [], []
            for itr, ite in inner.split(np.zeros(len(tr)), y[tr], gtr):
                clf = BTTRClassifier(nfactor=K).fit(tensor[tr][itr], y[tr][itr])
                is_.append(clf.decision_function(tensor[tr][ite])); iy.append(y[tr][ite])
            iy, is_ = np.concatenate(iy), np.concatenate(is_)
            auc = roc_auc_score(iy, is_) if len(np.unique(iy)) == 2 else 0.5
            if auc > best_auc:
                best_auc, best_k = auc, K
        clf = BTTRClassifier(nfactor=best_k).fit(tensor[tr], y[tr])
        os_.append(clf.decision_function(tensor[te])); oy.append(y[te])
    return np.concatenate(oy), np.concatenate(os_)


def _clinical_oof(recs, y, groups, seed):
    outer = StratifiedGroupKFold(OUTER, shuffle=True, random_state=seed)
    oy, os_ = [], []
    for tr, te in outer.split(np.zeros(len(y)), y, groups):
        Xtr = np.nan_to_num(bl.clinical_features([recs[i] for i in tr]))
        Xte = np.nan_to_num(bl.clinical_features([recs[i] for i in te]))
        sc = StandardScaler().fit(Xtr)
        lr = LogisticRegression(max_iter=1000, random_state=seed).fit(sc.transform(Xtr), y[tr])
        os_.append(lr.predict_proba(sc.transform(Xte))[:, 1]); oy.append(y[te])
    return np.concatenate(oy), np.concatenate(os_)


def main():
    config.TENSORISATION = "median_beat"
    db = load_db()
    rows = []
    for include_flutter in (False, True):
        pos, healthy = _afsr_frames(db, include_flutter)
        tag = "AFIB+AFLT" if include_flutter else "AFIB"
        b_aucs, c_aucs = [], []
        for seed in range(N_SEEDS):
            recs = _build_recs(pos, healthy, seed)
            y = np.array([r.label for r in recs]); groups = np.array([r.patient_id for r in recs])
            tensor = pp.build_tensor(recs)
            by, bs = _bttr_oof(tensor, y, groups, seed)
            cy, cs = _clinical_oof(recs, y, groups, seed)
            b_aucs.append(roc_auc_score(by, bs)); c_aucs.append(roc_auc_score(cy, cs))
        b_aucs, c_aucs = np.array(b_aucs), np.array(c_aucs)
        diff = b_aucs - c_aucs
        try:
            w, p = wilcoxon(b_aucs, c_aucs)
        except Exception:
            p = float("nan")
        print(f"\n[{tag}] n_pos={len(pos)} nested-CV (K tuned in {K_GRID})")
        print(f"  BTTR     {b_aucs.mean():.3f} +/- {b_aucs.std():.3f}")
        print(f"  clinical {c_aucs.mean():.3f} +/- {c_aucs.std():.3f}")
        print(f"  paired delta = {diff.mean():+.3f} (min {diff.min():+.3f}, all_positive={bool((diff>0).all())}) "
              f"Wilcoxon p={p:.4f}")
        rows.append({"cohort": tag, "n_pos": len(pos),
                     "bttr_mean": b_aucs.mean(), "bttr_std": b_aucs.std(),
                     "clinical_mean": c_aucs.mean(), "clinical_std": c_aucs.std(),
                     "delta_mean": diff.mean(), "delta_min": diff.min(), "wilcoxon_p": p})
    pd.DataFrame(rows).to_csv(config.RESULTS_DIR / "nested_afsr.csv", index=False)
    print(f"\nWrote {config.RESULTS_DIR / 'nested_afsr.csv'}")


if __name__ == "__main__":
    main()
