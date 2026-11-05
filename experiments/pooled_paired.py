"""Paired 95% CIs for ten-draw mean AUROC differences, pooled over the control draws.

The ten matched control draws share the same positive records, so they are not
independent samples. This script resamples patients once per bootstrap iteration
and applies the same resample to every draw: each iteration draws patients with
replacement from the union of the patients of all ten draws, keeps in each draw the
records of the drawn patients (with their multiplicity), computes the AUROC
difference of each contrast per draw, and averages over the draws. The point
estimate is the mean over the draws of the difference on the full out-of-fold
scores (the protocol P1 mean difference); the CI is the 2.5 and 97.5 percentile of
the bootstrap means, and the fraction is the share of resamples with a positive
mean difference.

Two sets of contrasts:
  - methods: BTTR minus each baseline, from results/benchmark_draws_r3_scores.npz
    (written by benchmark_draws.py), to results/pooled_paired.csv;
  - alignment: aligned minus unaligned for BTTR and for the CNN, from
    results/matched_ablation_scores.npz (written by matched_ablation.py), to
    results/pooled_paired_alignment.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from core import config
from experiments.harden_afsr import N_SEEDS, NEG_RATIO

N_BOOT, SEED = 2000, 0
COHORTS = (("afib", "AFIB (primary)"), ("aflt", "AFIB+AFLT (secondary)"))
METHOD_CONTRASTS = {m: ("bttr", m) for m in ("hubert_tuned", "hubert", "cnn", "clinical", "cp")}
ALIGNMENT_CONTRASTS = {m: (f"{m}_aligned", f"{m}_unaligned") for m in ("bttr", "cnn")}


def _draws(z, key, names):
    """Per draw: labels, patient ids, and the named score vectors."""
    return [{"y": z[f"{key}|{s}|y"], "g": z[f"{key}|{s}|patient"],
             "s": {n: z[f"{key}|{s}|{n}"] for n in names}} for s in range(N_SEEDS)]


def _mean_diff(draws, idx_per_draw, contrasts):
    """Mean over draws of the AUROC difference a minus b of each contrast, on the given row indices."""
    out = {c: [] for c in contrasts}
    for d, ii in zip(draws, idx_per_draw):
        y = d["y"][ii]
        if len(np.unique(y)) < 2:
            return None
        auc = {n: roc_auc_score(y, v[ii]) for n, v in d["s"].items()}
        for c, (a, b) in contrasts.items():
            out[c].append(auc[a] - auc[b])
    return {c: float(np.mean(v)) for c, v in out.items()}


def pooled(z, contrasts, label_col, rng):
    """Rows of point estimate, pooled CI and fraction positive for every contrast and cohort."""
    names = sorted({n for pair in contrasts.values() for n in pair})
    rows = []
    for key, tag in COHORTS:
        draws = _draws(z, key, names)
        rows_of = [{p: np.where(d["g"] == p)[0] for p in np.unique(d["g"])} for d in draws]
        patients = np.unique(np.concatenate([d["g"] for d in draws]))
        point = _mean_diff(draws, [np.arange(len(d["y"])) for d in draws], contrasts)
        boot = {c: [] for c in contrasts}
        while len(boot[next(iter(contrasts))]) < N_BOOT:
            chosen = rng.choice(patients, len(patients), replace=True)
            idx = [np.concatenate([r[p] for p in chosen if p in r]) for r in rows_of]
            res = _mean_diff(draws, idx, contrasts)
            if res is None:
                continue
            for c in contrasts:
                boot[c].append(res[c])
        print(f"\n=== {tag}: ten-draw mean AUROC difference, pooled patient bootstrap ({len(patients)} patients) ===")
        for c, (a, b) in contrasts.items():
            v = np.array(boot[c]); lo, hi = np.percentile(v, [2.5, 97.5])
            print(f"  {a} - {b:16s} {point[c]:+.3f} [{lo:+.3f}, {hi:+.3f}]  P(>0)={np.mean(v > 0):.2f}")
            rows.append({"cohort": tag, label_col: c, "mean_delta": round(point[c], 3),
                         "ci_lo": round(float(lo), 3), "ci_hi": round(float(hi), 3),
                         "frac_positive": round(float(np.mean(v > 0)), 3)})
    return rows


def main():
    for src, contrasts, label_col, dst in (
            (f"benchmark_draws_r{NEG_RATIO}_scores.npz", METHOD_CONTRASTS, "baseline", "pooled_paired.csv"),
            ("matched_ablation_scores.npz", ALIGNMENT_CONTRASTS, "model", "pooled_paired_alignment.csv")):
        rows = pooled(np.load(config.RESULTS_DIR / src), contrasts, label_col, np.random.default_rng(SEED))
        out = config.RESULTS_DIR / dst
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
