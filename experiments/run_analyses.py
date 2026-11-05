"""Supporting analyses for the AF-in-SR benchmark.

(1) Honest uncertainty: patient-level bootstrap 95% CIs (resampling PATIENTS, not
    folds) for AUROC, plus AUPRC, replacing the misleading 10-seed SD and the
    floor-value Wilcoxon p. Paired BTTR-clinical delta CI on the same resamples.
(2) Symmetric tuning: every method gets inner-CV hyperparameter selection on the
    training rows only -- BTTR over K, clinical-LR over C, CP over rank -- so the
    comparison is not tuning-asymmetric.
(3) AFIB-only is now the PRIMARY cohort; AFIB+flutter is secondary.
(4) Block-map stability across 10 seeds:
    per-lead importance profiles correlated across seeds -> a stability score, so
    the interpretability claim is quantitative, not a single-figure narrative.
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from core import baselines as bl
from core import config
from core import preprocess as pp
from core.bttr_model import BTTRClassifier
from experiments.harden_afsr import _afsr_frames, _build_recs, load_db

config.TENSORISATION = "median_beat"
OUTER, INNER, SEED = 5, 3, 0
K_GRID = (6, 8, 12, 16)
C_GRID = (0.01, 0.1, 1.0, 10.0)
CP_GRID = (4, 8, 16)
N_BOOT = 2000
N_STAB_SEEDS = 10


# ---- inner-tuned out-of-fold scoring (aligned to recs order) ---------------- #
def _fit_score(method, hp, tensor, recs, tr, te, y):
    if method == "bttr":
        clf = BTTRClassifier(nfactor=hp).fit(tensor[tr], y[tr])
        return clf.decision_function(tensor[te])
    if method == "clinical":
        Xtr = np.nan_to_num(bl.clinical_features([recs[i] for i in tr]))
        Xte = np.nan_to_num(bl.clinical_features([recs[i] for i in te]))
        sc = StandardScaler().fit(Xtr)
        lr = LogisticRegression(max_iter=1000, C=hp, random_state=SEED).fit(sc.transform(Xtr), y[tr])
        return lr.predict_proba(sc.transform(Xte))[:, 1]
    if method == "cp":
        Xtr, Xte = bl.cp_train_test_features(tensor[tr], tensor[te], rank=hp, seed=SEED)
        sc = StandardScaler().fit(np.nan_to_num(Xtr))
        lr = LogisticRegression(max_iter=1000, random_state=SEED).fit(sc.transform(np.nan_to_num(Xtr)), y[tr])
        return lr.predict_proba(sc.transform(np.nan_to_num(Xte)))[:, 1]
    raise ValueError(method)


def _grid(method):
    return {"bttr": K_GRID, "clinical": C_GRID, "cp": CP_GRID}[method]


def oof_tuned(method, recs, tensor, y, groups):
    """Outer 5-fold with inner-CV hyperparameter selection; return score array aligned to recs."""
    outer = StratifiedGroupKFold(OUTER, shuffle=True, random_state=SEED)
    s = np.full(len(y), np.nan)
    for tr, te in outer.split(np.zeros(len(y)), y, groups):
        best_hp, best = _grid(method)[0], -1
        inner = StratifiedGroupKFold(INNER, shuffle=True, random_state=SEED)
        for hp in _grid(method):
            iy, isx = [], []
            for itr, ite in inner.split(np.zeros(len(tr)), y[tr], groups[tr]):
                isx.append(_fit_score(method, hp, tensor, recs, tr[itr], tr[ite], y))
                iy.append(y[tr][ite])
            iy, isx = np.concatenate(iy), np.concatenate(isx)
            auc = roc_auc_score(iy, isx) if len(np.unique(iy)) == 2 else 0.5
            if auc > best:
                best, best_hp = auc, hp
        s[te] = _fit_score(method, best_hp, tensor, recs, tr, te, y)
    return s


def patient_bootstrap(scores, y, groups, n=N_BOOT):
    """Resample PATIENTS with replacement; return the AUROC point estimate on the out-of-fold scores and the 95% percentile CI per method,
    plus paired delta(bttr-clinical) CI. scores: dict method->aligned score array."""
    rng = np.random.default_rng(SEED)
    pats = np.unique(groups)
    idx_of = {p: np.where(groups == p)[0] for p in pats}
    aucs = {m: [] for m in scores}
    deltas = []
    for _ in range(n):
        chosen = rng.choice(pats, len(pats), replace=True)
        idx = np.concatenate([idx_of[p] for p in chosen])
        if len(np.unique(y[idx])) < 2:
            continue
        a = {}
        for m, s in scores.items():
            a[m] = roc_auc_score(y[idx], s[idx])
            aucs[m].append(a[m])
        if "bttr" in a and "clinical" in a:
            deltas.append(a["bttr"] - a["clinical"])
    out = {}
    for m, v in aucs.items():
        v = np.array(v)
        out[m] = (float(roc_auc_score(y, scores[m])), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    d = np.array(deltas)
    out["_delta"] = (float(roc_auc_score(y, scores["bttr"]) - roc_auc_score(y, scores["clinical"])), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)),
                     float((d > 0).mean()))
    return out


def map_stability(pos, healthy, k=8, n_seeds=N_STAB_SEEDS, permute=False):
    """Per-lead importance profile (mean|weight| over time per lead, summed over blocks
    weighted by |b|) for each seed; report mean pairwise Pearson r across seeds.

    With permute=True the labels are shuffled (per seed, breaking the X-label link)
    before fitting -- the label-permutation null for the stability score: how
    reproducible the recovered profile is when there is no real label structure to
    recover. The real stability r should sit well above this null."""
    profiles = []
    for seed in range(n_seeds):
        recs = _build_recs(pos, healthy, seed)
        y = np.array([r.label for r in recs])
        if permute:
            y = np.random.default_rng(1000 + seed).permutation(y)
        X = pp.build_tensor(recs)
        clf = BTTRClassifier(nfactor=k).fit(X, y)
        maps, b = clf.block_maps()                      # (K,T,L),(K,)
        imp = np.sum(np.abs(maps) * np.abs(b)[:, None, None], axis=0).mean(axis=0)  # (L,)
        profiles.append(imp / (imp.sum() + 1e-9))
    P = np.array(profiles)
    rs = [np.corrcoef(P[i], P[j])[0, 1] for i in range(len(P)) for j in range(i + 1, len(P))]
    return float(np.mean(rs)), float(np.std(rs)), P.mean(0)


def main():
    db = load_db()
    rows = []
    for include_flutter, tag in ((False, "AFIB (primary)"), (True, "AFIB+AFLT (secondary)")):
        pos, healthy = _afsr_frames(db, include_flutter)
        recs = _build_recs(pos, healthy, SEED)
        y = np.array([r.label for r in recs]); groups = np.array([r.patient_id for r in recs])
        tensor = pp.build_tensor(recs)
        scores = {m: oof_tuned(m, recs, tensor, y, groups) for m in ("bttr", "clinical", "cp")}
        ci = patient_bootstrap(scores, y, groups)
        print(f"\n=== {tag}  n={len(y)} pos={int(y.sum())}  (symmetric inner-CV tuning, patient bootstrap) ===")
        for m in ("bttr", "clinical", "cp"):
            mu, lo, hi = ci[m]
            ap = average_precision_score(y, scores[m])
            print(f"  {m:9s} AUROC={mu:.3f} [{lo:.3f},{hi:.3f}]  AUPRC={ap:.3f}")
            rows.append({"cohort": tag, "method": m, "auroc": mu, "ci_lo": lo, "ci_hi": hi, "auprc": ap})
        dm, dlo, dhi, dpos = ci["_delta"]
        print(f"  paired delta(bttr-clinical) = {dm:+.3f} [{dlo:+.3f},{dhi:+.3f}]  P(delta>0)={dpos:.3f}")
        rows.append({"cohort": tag, "method": "delta_bttr_clin", "auroc": dm, "ci_lo": dlo, "ci_hi": dhi, "auprc": dpos})

    # interpretability stability on the PRIMARY cohort
    pos, healthy = _afsr_frames(db, include_flutter=False)
    r_mean, r_std, prof = map_stability(pos, healthy)
    rp_mean, rp_std, _ = map_stability(pos, healthy, permute=True)   # label-permutation null
    leads = config.LEAD_NAMES
    top = ", ".join(f"{leads[i]}={prof[i]:.2f}" for i in np.argsort(-prof)[:4])
    print(f"\n=== block-map stability (AFIB primary, K=8, {N_STAB_SEEDS} seeds) ===")
    print(f"  mean pairwise per-lead-profile Pearson r = {r_mean:.3f} +/- {r_std:.3f}")
    print(f"  label-permutation null Pearson r        = {rp_mean:.3f} +/- {rp_std:.3f}")
    print(f"  consensus top leads: {top}")
    rows.append({"cohort": "AFIB (primary)", "method": "mapstab_pearson_r", "auroc": r_mean, "ci_lo": r_mean - r_std, "ci_hi": r_mean + r_std, "auprc": np.nan})
    rows.append({"cohort": "AFIB (primary)", "method": "mapstab_pearson_r_permuted", "auroc": rp_mean, "ci_lo": rp_mean - rp_std, "ci_hi": rp_mean + rp_std, "auprc": np.nan})

    pd.DataFrame(rows).to_csv(config.RESULTS_DIR / "analyses.csv", index=False)
    print(f"\nWrote {config.RESULTS_DIR / 'analyses.csv'}")


if __name__ == "__main__":
    main()
