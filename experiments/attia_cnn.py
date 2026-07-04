"""Attia-style 1D-CNN baseline for AF-in-SR on PTB-XL, under the SAME protocol as
BTTR (patient-grouped 5-fold, patient-level bootstrap CIs).
Equal-footing deep-learning comparison.

A faithful Attia model is a deep CNN trained on hundreds of thousands of ECGs; here
the public cohort has 42-55 positives, so a deep net is expected to OVERFIT. We use
a small, heavily-regularised 1D-CNN with class-weighted loss, inner-validation early
stopping, and 3-seed ensembling to give the CNN its best honest shot. Two inputs:
  - 'beat'   : aligned median-beat tensor (same input BTTR sees) -> fairest head-to-head
  - 'raw'    : full denoised 10 s strip (true Attia-style raw input)
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from core import config
from core import preprocess as pp
from experiments.harden_afsr import _afsr_frames, _build_recs, _db_with_ondisk

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
OUTER, SEED, N_ENS = 5, 0, 3
MAX_EPOCH, PATIENCE = 120, 18
N_BOOT = 2000


class SmallCNN(nn.Module):
    def __init__(self, n_leads):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_leads, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Dropout(0.3),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
            nn.Flatten(), nn.Dropout(0.5), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _standardize(Xtr, Xte):
    m = Xtr.mean(axis=(0, 2), keepdims=True); s = Xtr.std(axis=(0, 2), keepdims=True) + 1e-6
    return (Xtr - m) / s, (Xte - m) / s


def _train_one(Xtr, ytr, Xva, yva, n_leads, seed):
    torch.manual_seed(seed)
    model = SmallCNN(n_leads).to(DEV)
    pos_w = torch.tensor([(ytr == 0).sum() / max(1, (ytr == 1).sum())], dtype=torch.float32, device=DEV)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=DEV); ytr_t = torch.tensor(ytr, dtype=torch.float32, device=DEV)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=DEV)
    best_auc, best_state, bad = -1, None, 0
    for ep in range(MAX_EPOCH):
        model.train(); opt.zero_grad()
        loss = lossf(model(Xtr_t), ytr_t); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            va = torch.sigmoid(model(Xva_t)).cpu().numpy()
        auc = roc_auc_score(yva, va) if len(np.unique(yva)) == 2 else 0.5
        if auc > best_auc:
            best_auc, best_state, bad = auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model


def cnn_oof(X, y, groups):
    """X: (N, leads, time). Returns OOF scores aligned to samples (3-seed ensemble)."""
    s = np.zeros(len(y))
    outer = StratifiedGroupKFold(OUTER, shuffle=True, random_state=SEED)
    for tr, te in outer.split(np.zeros(len(y)), y, groups):
        # inner validation split (patient-grouped) for early stopping
        inner = StratifiedGroupKFold(4, shuffle=True, random_state=SEED)
        itr, iva = next(inner.split(np.zeros(len(tr)), y[tr], groups[tr]))
        tri, vai = tr[itr], tr[iva]
        Xtr, Xte = _standardize(X[tri], X[te]); _, Xva = _standardize(X[tri], X[vai])
        ens = np.zeros(len(te))
        for sd in range(N_ENS):
            m = _train_one(Xtr, y[tri], Xva, y[vai], X.shape[1], SEED + sd)
            m.eval()
            with torch.no_grad():
                ens += torch.sigmoid(m(torch.tensor(Xte, dtype=torch.float32, device=DEV))).cpu().numpy()
        s[te] = ens / N_ENS
    return s


def patient_boot(scores_dict, y, groups, n=N_BOOT):
    rng = np.random.default_rng(SEED)
    pats = np.unique(groups); idx_of = {p: np.where(groups == p)[0] for p in pats}
    out = {}
    coll = {m: [] for m in scores_dict}
    for _ in range(n):
        chosen = rng.choice(pats, len(pats), replace=True)
        idx = np.concatenate([idx_of[p] for p in chosen])
        if len(np.unique(y[idx])) < 2:
            continue
        for m, s in scores_dict.items():
            coll[m].append(roc_auc_score(y[idx], s[idx]))
    for m, v in coll.items():
        v = np.array(v); out[m] = (float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return out


def main():
    print(f"device={DEV}")
    db = _db_with_ondisk()
    rows = []
    for include_flutter, tag in ((False, "AFIB (primary)"), (True, "AFIB+AFLT (secondary)")):
        pos, healthy = _afsr_frames(db, include_flutter)
        recs = _build_recs(pos, healthy, SEED)
        y = np.array([r.label for r in recs]); groups = np.array([r.patient_id for r in recs])
        scores = {}
        for inp in ("beat", "raw"):
            config.TENSORISATION = "median_beat" if inp == "beat" else "window"
            T = pp.build_tensor(recs)                       # (N, time, leads)
            X = np.transpose(T, (0, 2, 1)).astype(np.float32)   # (N, leads, time)
            scores[f"cnn_{inp}"] = cnn_oof(X, y, groups)
        ci = patient_boot(scores, y, groups)
        print(f"\n=== {tag}  n={len(y)} pos={int(y.sum())}  (Attia-style CNN, patient bootstrap) ===")
        for m in scores:
            mu, lo, hi = ci[m]; ap = average_precision_score(y, scores[m])
            print(f"  {m:10s} AUROC={mu:.3f} [{lo:.3f},{hi:.3f}]  AUPRC={ap:.3f}")
            rows.append({"cohort": tag, "method": m, "auroc": mu, "ci_lo": lo, "ci_hi": hi, "auprc": ap})
    pd.DataFrame(rows).to_csv(config.RESULTS_DIR / "attia_cnn.csv", index=False)
    print(f"\nWrote {config.RESULTS_DIR / 'attia_cnn.csv'}")


if __name__ == "__main__":
    main()
