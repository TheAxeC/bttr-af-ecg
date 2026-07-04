"""Unified, consistent benchmark table: BTTR vs CP vs CNN vs HuBERT-ECG (foundation
model) vs clinical, ALL under identical patient-grouped 5-fold + patient-bootstrap
CIs. HuBERT uses the saved embeddings from hubert_probe.py."""
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
from experiments.attia_cnn import cnn_oof
from core.bttr_model import BTTRClassifier
from experiments.harden_afsr import _afsr_frames, _build_recs, _db_with_ondisk

K, SEED = 8, 0


def oof(method, recs, T, emb, y, g):
    s = np.full(len(y), np.nan)
    sk = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    for tr, te in sk.split(np.zeros(len(y)), y, g):
        if method == "bttr":
            s[te] = BTTRClassifier(nfactor=K).fit(T[tr], y[tr]).decision_function(T[te])
        elif method == "cp":
            Xtr, Xte = bl.cp_train_test_features(T[tr], T[te], rank=8, seed=SEED)
            sc = StandardScaler().fit(np.nan_to_num(Xtr))
            lr = LogisticRegression(max_iter=1000, random_state=SEED).fit(sc.transform(np.nan_to_num(Xtr)), y[tr])
            s[te] = lr.predict_proba(sc.transform(np.nan_to_num(Xte)))[:, 1]
        elif method == "clinical":
            Xtr = np.nan_to_num(bl.clinical_features([recs[i] for i in tr]))
            Xte = np.nan_to_num(bl.clinical_features([recs[i] for i in te]))
            sc = StandardScaler().fit(Xtr)
            lr = LogisticRegression(max_iter=1000, random_state=SEED).fit(sc.transform(Xtr), y[tr])
            s[te] = lr.predict_proba(sc.transform(Xte))[:, 1]
        elif method == "hubert":
            sc = StandardScaler().fit(emb[tr])
            lr = LogisticRegression(max_iter=2000, C=0.1, random_state=SEED).fit(sc.transform(emb[tr]), y[tr])
            s[te] = lr.predict_proba(sc.transform(emb[te]))[:, 1]
    return s


def boot(scores, y, g, n=2000):
    rng = np.random.default_rng(SEED); pats = np.unique(g)
    idx = {p: np.where(g == p)[0] for p in pats}; coll = {m: [] for m in scores}
    for _ in range(n):
        ch = rng.choice(pats, len(pats), True); ii = np.concatenate([idx[p] for p in ch])
        if len(np.unique(y[ii])) < 2:
            continue
        for m, s in scores.items():
            coll[m].append(roc_auc_score(y[ii], s[ii]))
    return {m: (float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for m, v in coll.items()}


def main():
    db = _db_with_ondisk(); rows = []
    for inc, tag, embf in ((False, "AFIB (primary)", "afib"), (True, "AFIB+AFLT (secondary)", "aflt")):
        pos, healthy = _afsr_frames(db, inc); recs = _build_recs(pos, healthy, SEED)
        y = np.array([r.label for r in recs]); g = np.array([r.patient_id for r in recs])
        config.TENSORISATION = "median_beat"; T = pp.build_tensor(recs)
        emb = np.load(config.RESULTS_DIR / f"hubert_emb_{embf}.npy")
        Xcnn = np.transpose(T, (0, 2, 1)).astype(np.float32)
        scores = {"bttr": oof("bttr", recs, T, emb, y, g),
                  "cp": oof("cp", recs, T, emb, y, g),
                  "hubert": oof("hubert", recs, T, emb, y, g),
                  "clinical": oof("clinical", recs, T, emb, y, g),
                  "cnn": cnn_oof(Xcnn, y, g)}
        ci = boot(scores, y, g)
        print(f"\n=== {tag}  n={len(y)} pos={int(y.sum())} ===")
        for m in ("bttr", "cp", "cnn", "hubert", "clinical"):
            mu, lo, hi = ci[m]; ap = average_precision_score(y, scores[m])
            print(f"  {m:9s} AUROC={mu:.3f} [{lo:.3f},{hi:.3f}]  AUPRC={ap:.3f}")
            rows.append({"cohort": tag, "method": m, "auroc": round(mu, 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3), "auprc": round(ap, 3)})
    pd.DataFrame(rows).to_csv(config.RESULTS_DIR / "final_table.csv", index=False)
    print(f"\nWrote {config.RESULTS_DIR / 'final_table.csv'}")


if __name__ == "__main__":
    main()
