"""All methods over the ten matched control draws.

For each cohort and each draw (seed 0..9: same positives, newly matched controls)
every method is scored out-of-fold on the same patient-grouped 5-fold split (fold
seed = draw seed): BTTR, CP surrogate, clinical features, the HuBERT-ECG probe with
C = 0.1 and with C tuned by inner CV in each training fold, and the CNN, all on the aligned median beat, plus age alone as a check on the matching.
Reported per draw and as mean +/- SD over the draws.

Usage: python -m experiments.benchmark_draws [--ratio 3]
Writes results/benchmark_draws_r<ratio>.csv and the out-of-fold scores of every
draw, with labels and patient ids, to results/benchmark_draws_r<ratio>_scores.npz
(keys "<cohort>|<seed>|<name>", cohort afib or aflt) for pooled_paired.py.
"""
from __future__ import annotations

import argparse
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from core import config
from core import preprocess as pp
from experiments.attia_cnn import cnn_oof
from experiments.final_eval import oof
from experiments.harden_afsr import N_SEEDS, NEG_RATIO, _afsr_frames, _build_recs, load_db
from experiments.hubert_probe import embeddings_for

METHODS = ("bttr", "cp", "clinical", "hubert", "hubert_tuned")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=int, default=NEG_RATIO)
    ratio = ap.parse_args().ratio
    db = load_db(); age = db.set_index("ecg_id").age; rows = []; saved = {}
    for inc, tag, key in ((False, "AFIB (primary)", "afib"), (True, "AFIB+AFLT (secondary)", "aflt")):
        pos, healthy = _afsr_frames(db, inc)
        for seed in range(N_SEEDS):
            recs = _build_recs(pos, healthy, seed, ratio)
            y = np.array([r.label for r in recs]); g = np.array([r.patient_id for r in recs])
            T = pp.build_tensor(recs, mode="median_beat")
            emb = embeddings_for(recs)
            scores = {m: oof(m, recs, T, emb, y, g, fold_seed=seed) for m in METHODS}
            scores["cnn"] = cnn_oof(np.transpose(T, (0, 2, 1)).astype(np.float32), y, g, fold_seed=seed)
            scores["age"] = age.loc[[int(r.record_id) for r in recs]].to_numpy().astype(float)
            res = {m: roc_auc_score(y, s) for m, s in scores.items()}
            saved[f"{key}|{seed}|y"] = y; saved[f"{key}|{seed}|patient"] = g.astype(str)
            for m, s in scores.items():
                saved[f"{key}|{seed}|{m}"] = np.asarray(s, dtype=np.float64)
            for m, v in res.items():
                rows.append({"cohort": tag, "ratio": ratio, "seed": seed, "method": m, "auroc": round(v, 6)})
            print(f"  {tag} r={ratio} draw {seed}: " + "  ".join(f"{m}={v:.3f}" for m, v in res.items()), flush=True)
    df = pd.DataFrame(rows)
    out = config.RESULTS_DIR / f"benchmark_draws_r{ratio}.csv"
    df.to_csv(out, index=False)
    np.savez(config.RESULTS_DIR / f"benchmark_draws_r{ratio}_scores.npz", **saved)
    print(df.groupby(["cohort", "method"]).auroc.agg(["mean", lambda v: v.std(ddof=0)]).round(3).to_string())
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
