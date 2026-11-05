"""Alignment ablation with BTTR and the CNN under the same two protocols (Table 3).

Ten-draw protocol: for each control draw (seed 0..9; the positives are fixed, the
controls are redrawn) both models are scored out-of-fold on the same
patient-grouped 5-fold split (fold seed = draw seed). Reported as mean and SD of
the ten AUROCs.

Single-split protocol: the seed-0 draw and split of the primary benchmark,
reported as AUROC with the 95% patient-level bootstrap interval.

Aligned = R-peak-aligned median beat; unaligned = the full 10 s window without
registration. Writes results/matched_ablation.csv and the out-of-fold scores of
every draw, with labels and patient ids, to results/matched_ablation_scores.npz
(keys "<cohort>|<seed>|<name>", cohort afib or aflt, name <model>_<tensorisation>)
for the paired alignment CIs in pooled_paired.py.
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from core import config
from core import preprocess as pp
from core.bttr_model import BTTRClassifier
from experiments.attia_cnn import cnn_oof, patient_boot
from experiments.harden_afsr import N_SEEDS, _afsr_frames, _build_recs, load_db

K = 8
TENS = {"aligned": "median_beat", "unaligned": "window"}


def bttr_oof(T, y, g, fold_seed):
    s = np.full(len(y), np.nan)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=fold_seed).split(np.zeros(len(y)), y, g):
        s[te] = BTTRClassifier(nfactor=K).fit(T[tr], y[tr]).decision_function(T[te])
    return s


def scores_for_draw(pos, healthy, seed):
    recs = _build_recs(pos, healthy, seed)
    y = np.array([r.label for r in recs]); g = np.array([r.patient_id for r in recs])
    out = {}
    for name, mode in TENS.items():
        T = pp.build_tensor(recs, mode=mode)
        out[("bttr", name)] = bttr_oof(T, y, g, seed)
        out[("cnn", name)] = cnn_oof(np.transpose(T, (0, 2, 1)).astype(np.float32), y, g, fold_seed=seed)
    return out, y, g


def main():
    db = load_db(); rows = []; saved = {}
    for inc, tag, ck in ((False, "AFIB (primary)", "afib"), (True, "AFIB+AFLT (secondary)", "aflt")):
        pos, healthy = _afsr_frames(db, inc)
        per = {}
        for seed in range(N_SEEDS):
            sc, y, g = scores_for_draw(pos, healthy, seed)
            saved[f"{ck}|{seed}|y"] = y; saved[f"{ck}|{seed}|patient"] = g.astype(str)
            for (m, t), s in sc.items():
                saved[f"{ck}|{seed}|{m}_{t}"] = np.asarray(s, dtype=np.float64)
            for key, s in sc.items():
                per.setdefault(key, []).append(roc_auc_score(y, s))
            if seed == 0:
                ci = patient_boot({f"{m}_{t}": s for (m, t), s in sc.items()}, y, g)
                for (m, t), s in sc.items():
                    _, lo, hi = ci[f"{m}_{t}"]
                    rows.append({"cohort": tag, "model": m, "tensorisation": t, "protocol": "single_split",
                                 "auroc": round(roc_auc_score(y, s), 3), "lo": round(lo, 3), "hi": round(hi, 3)})
            print(f"  {tag} draw {seed}: " + "  ".join(f"{m}_{t}={v[-1]:.3f}" for (m, t), v in per.items()), flush=True)
        for (m, t), v in per.items():
            rows.append({"cohort": tag, "model": m, "tensorisation": t, "protocol": "ten_draw",
                         "auroc": round(float(np.mean(v)), 3), "lo": np.nan, "hi": np.nan,
                         "sd": round(float(np.std(v)), 3)})
    out = config.RESULTS_DIR / "matched_ablation.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    np.savez(config.RESULTS_DIR / "matched_ablation_scores.npz", **saved)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
