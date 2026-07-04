"""Hardening + principled tuning of the AF-in-SR task.

The early afsr result (BTTR 0.667) was produced with the DETECTION-winning
config (median_beat_rr, K=8). That is mis-specified for afsr: every sample is in
sinus rhythm, so the RR-tachogram pseudo-lead encodes no AF-relevant irregularity
by construction -- it is dead weight. The exp_improve*.py scripts only ever tuned
load_detection, never afsr.

This script:
  1. Multi-seeds the pilot properly -- the original load_afsr fixes the negative
     draw at random_state=0, so run.py --seed only varied the CV folds. Here each
     seed also re-draws the healthy controls (from the 500 on-disk healthy-SR
     records), so we measure robustness to BOTH fold and control selection.
  2. Compares tensorisations: median_beat_rr (baseline) vs morphology-only
     median_beat (pre-registered principled choice) vs window (unaligned ablation).
  3. Reports an exploratory K sweep on the principled tensorisation -- flagged as
     exploratory because n_pos is tiny; the headline uses K=8 (unchanged from
     detection), not the sweep max.
  4. Optionally expands positives with atrial flutter (AFLT), a related atrial
     arrhythmia (42 -> 55 positives).

All numbers are patient-grouped StratifiedGroupKFold AUROC, mean +/- std over seeds.
"""
from __future__ import annotations

import ast
import glob
import os
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from core import config
from core import evaluate as ev
from core import preprocess as pp
from core.data import Recording, _read_signal

N_SEEDS = 10
NEG_RATIO = 3


def _db_with_ondisk():
    db = pd.read_csv(config.PTBXL_DIR / "ptbxl_database.csv")
    db.scp_codes = db.scp_codes.apply(ast.literal_eval)
    db["codes"] = db.scp_codes.apply(lambda c: set(c))
    db["recording_date"] = pd.to_datetime(db["recording_date"], errors="coerce")
    ondisk = {os.path.basename(h).replace("_lr.hea", "")
              for h in glob.glob(str(config.PTBXL_DIR / "records100" / "*" / "*_lr.hea"))}
    db["stem"] = db.filename_lr.apply(lambda f: os.path.basename(f).replace("_lr", ""))
    db["ondisk"] = db.stem.isin(ondisk)
    return db


def _afsr_frames(db, include_flutter: bool):
    af_codes = {"AFIB", "AFLT"} if include_flutter else {"AFIB"}
    is_af = db.codes.apply(lambda k: bool(k & af_codes))
    is_sr = db.codes.apply(lambda k: "SR" in k)
    af_pat = set(db[is_af].patient_id)

    sr = db[is_sr & db.patient_id.isin(af_pat)]
    afd = db[is_af][["patient_id", "recording_date"]]
    m = sr.merge(afd, on="patient_id", suffixes=("_sr", "_af"))
    m["dt"] = (m.recording_date_af - m.recording_date_sr).dt.days
    pos_ids = set(m[(m.dt >= 0) & (m.dt <= config.PRED_WINDOW_DAYS)].ecg_id)

    pos = db[db.ecg_id.isin(pos_ids) & db.ondisk]
    healthy = db[is_sr & (~db.patient_id.isin(af_pat)) & db.ondisk]
    healthy = healthy.groupby("patient_id", group_keys=False).head(1)   # one rec/patient
    return pos, healthy


def _build_recs(pos, healthy, seed):
    n_neg = min(len(pos) * NEG_RATIO, len(healthy))
    neg = healthy.sample(n_neg, random_state=seed)
    recs = []
    for sub, lab in ((pos, 1), (neg, 0)):
        for _, row in sub.iterrows():
            sig, fs = _read_signal(row, lowres=True)
            recs.append(Recording(sig, fs, lab, "afsr", str(row.patient_id), str(row.ecg_id)))
    return recs


def run_config(pos, healthy, tensorisation, K, methods=("bttr", "clinical_lr", "cp_surrogate")):
    config.TENSORISATION = tensorisation
    config.BTTR_NFACTOR = K
    per = {mth: [] for mth in methods}
    for seed in range(N_SEEDS):
        recs = _build_recs(pos, healthy, seed)
        y = np.array([r.label for r in recs])
        groups = np.array([r.patient_id for r in recs])
        tensor = pp.build_tensor(recs)
        for mth in methods:
            try:
                r = ev.evaluate_method(mth, recs, y, groups, seed=seed, tensor=tensor)
                per[mth].append(r["auroc"])
            except Exception as e:
                print(f"    !! {mth} seed{seed}: {e}")
    return {m: (float(np.mean(v)), float(np.std(v)), v) for m, v in per.items() if v}


def main():
    db = _db_with_ondisk()
    rows = []
    for include_flutter in (False, True):
        pos, healthy = _afsr_frames(db, include_flutter)
        tag = "AFIB+AFLT" if include_flutter else "AFIB"
        print(f"\n===== cohort={tag}  positives={len(pos)}  healthy_pool={len(healthy)} "
              f"(neg/seed={min(len(pos)*NEG_RATIO,len(healthy))})  seeds={N_SEEDS} =====")
        # 1) tensorisation comparison at K=8
        for tens in ("median_beat_rr", "median_beat", "window"):
            res = run_config(pos, healthy, tens, K=8)
            print(f"  [K=8 {tens:16s}] " + "  ".join(
                f"{m}={mu:.3f}±{sd:.3f}" for m, (mu, sd, _) in res.items()))
            for m, (mu, sd, v) in res.items():
                rows.append({"cohort": tag, "tens": tens, "K": 8, "method": m,
                             "auroc_mean": mu, "auroc_std": sd, "n_pos": len(pos)})
        # 2) exploratory K sweep on the principled (morphology-only) tensorisation, BTTR only
        print(f"  -- exploratory K sweep (median_beat, BTTR only; n_pos tiny, optimistic) --")
        for K in (4, 6, 8, 12, 16, 24):
            res = run_config(pos, healthy, "median_beat", K, methods=("bttr",))
            mu, sd, _ = res["bttr"]
            print(f"     K={K:<3d} bttr={mu:.3f}±{sd:.3f}")
            rows.append({"cohort": tag, "tens": "median_beat_Ksweep", "K": K, "method": "bttr",
                         "auroc_mean": mu, "auroc_std": sd, "n_pos": len(pos)})
    out = config.RESULTS_DIR / "harden_afsr.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
