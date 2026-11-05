"""AF-in-SR cohort builder and the ten-draw robustness runs.

Cohorts. A positive is a sinus-rhythm ECG of a patient whose AF ECG (AFIB; AFIB or
AFLT for the secondary cohort) is recorded 0 to PRED_WINDOW_DAYS days after it.
Controls are sinus-rhythm ECGs of patients with no AFIB or AFLT ECG at all, one
record per patient, matched to the positives on sex and age: for each positive
record, in a seed-shuffled order, the NEG_RATIO unused controls of the same sex
with the nearest age are taken (ties broken at random by the seed). PTB-XL codes
ages above 89 as 300, so those positives are matched within that group. Each seed
is one control draw; the positives are the same in every draw.

Runs (main): for each draw, patient-grouped StratifiedGroupKFold AUROC (fold seed
= draw seed), for three tensorisations (median_beat_rr, median_beat, window) and
a K sweep of BTTR on median_beat; reported as mean +/- std over the draws.
"""
from __future__ import annotations

import ast
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


def load_db():
    """PTB-XL metadata (ptbxl_database.csv) with the SCP code set and the recording date."""
    path = config.PTBXL_DIR / "ptbxl_database.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Download PTB-XL 1.0.3 to {config.PTBXL_DIR} (see README, Data).")
    db = pd.read_csv(path)
    db.scp_codes = db.scp_codes.apply(ast.literal_eval)
    db["codes"] = db.scp_codes.apply(lambda c: set(c))
    db["recording_date"] = pd.to_datetime(db["recording_date"], errors="coerce")
    return db


def _afsr_frames(db, include_flutter: bool):
    """Positive records and the control candidates (one per patient) of a cohort."""
    af_codes = {"AFIB", "AFLT"} if include_flutter else {"AFIB"}
    is_af = db.codes.apply(lambda k: bool(k & af_codes))
    is_sr = db.codes.apply(lambda k: "SR" in k)
    af_pat = set(db[is_af].patient_id)
    any_af_pat = set(db[db.codes.apply(lambda k: bool(k & {"AFIB", "AFLT"}))].patient_id)

    sr = db[is_sr & db.patient_id.isin(af_pat)]
    afd = db[is_af][["patient_id", "recording_date"]]
    m = sr.merge(afd, on="patient_id", suffixes=("_sr", "_af"))
    m["dt"] = (m.recording_date_af - m.recording_date_sr).dt.days
    pos_ids = set(m[(m.dt >= 0) & (m.dt <= config.PRED_WINDOW_DAYS)].ecg_id)

    pos = db[db.ecg_id.isin(pos_ids)].sort_values("ecg_id")
    healthy = db[is_sr & (~db.patient_id.isin(any_af_pat))].sort_values("ecg_id")
    healthy = healthy.groupby("patient_id", group_keys=False).head(1)   # one rec/patient
    return pos, healthy


def match_controls(pos, healthy, seed, ratio=NEG_RATIO):
    """Sex- and age-matched controls, `ratio` per positive record, without replacement."""
    rng = np.random.default_rng(seed)
    order = pos.iloc[rng.permutation(len(pos))]
    jitter = pd.Series(rng.random(len(healthy)) * 1e-3, index=healthy.index)
    used, picks = set(), []
    for _, r in order.iterrows():
        cand = healthy[(healthy.sex == r.sex) & ~healthy.index.isin(used)]
        dist = (cand.age - r.age).abs() + jitter[cand.index]
        pick = dist.nsmallest(ratio).index
        used.update(pick); picks.extend(pick)
    return healthy.loc[picks].sort_values("ecg_id")


def _build_recs(pos, healthy, seed, ratio=NEG_RATIO):
    neg = match_controls(pos, healthy, seed, ratio)
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
    db = load_db()
    rows = []
    for include_flutter in (False, True):
        pos, healthy = _afsr_frames(db, include_flutter)
        tag = "AFIB+AFLT" if include_flutter else "AFIB"
        print(f"\n===== cohort={tag}  positives={len(pos)}  control candidates={len(healthy)} "
              f"(matched controls/draw={len(pos)*NEG_RATIO})  draws={N_SEEDS} =====")
        # 1) tensorisation comparison at K=8
        for tens in ("median_beat_rr", "median_beat", "window"):
            res = run_config(pos, healthy, tens, K=8)
            print(f"  [K=8 {tens:16s}] " + "  ".join(
                f"{m}={mu:.3f}+/-{sd:.3f}" for m, (mu, sd, _) in res.items()))
            for m, (mu, sd, v) in res.items():
                rows.append({"cohort": tag, "tens": tens, "K": 8, "method": m,
                             "auroc_mean": mu, "auroc_std": sd, "n_pos": len(pos)})
        # 2) exploratory K sweep on the principled (morphology-only) tensorisation, BTTR only
        print(f"  -- exploratory K sweep (median_beat, BTTR only; n_pos tiny, optimistic) --")
        for K in (4, 6, 8, 12, 16, 24):
            res = run_config(pos, healthy, "median_beat", K, methods=("bttr",))
            mu, sd, _ = res["bttr"]
            print(f"     K={K:<3d} bttr={mu:.3f}+/-{sd:.3f}")
            rows.append({"cohort": tag, "tens": "median_beat_Ksweep", "K": K, "method": "bttr",
                         "auroc_mean": mu, "auroc_std": sd, "n_pos": len(pos)})
    out = config.RESULTS_DIR / "harden_afsr.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
