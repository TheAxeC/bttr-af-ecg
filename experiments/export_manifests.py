"""Write the record-level manifests of the AF-in-SR cohorts.

For each cohort (AFIB primary, AFIB+AFLT secondary) and each of the ten control
draws (seed 0..9) this lists every record with its patient, label, demographics,
the days from the sinus-rhythm ECG to the patient's first AF ECG in the 31-day
window (positives only), and the outer fold it falls in. Seed 0 is the draw and
fold split of the primary benchmark (Tables 1 and 2); seeds 0..9 are the draws of
the ten-draw protocol (alignment ablation, K sweep, map stability). The folds are
StratifiedGroupKFold(5, shuffle=True, random_state=seed) grouped by patient, the
split every driver uses.

Also writes a per-cohort demographic summary of the seed-0 draw.

Outputs: manifests/afsr_afib_draws.csv, manifests/afsr_aflt_draws.csv,
manifests/cohort_summary.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from core import config
from experiments.harden_afsr import N_SEEDS, _afsr_frames, load_db, match_controls


def _days_to_af(db, pos, include_flutter):
    af_codes = {"AFIB", "AFLT"} if include_flutter else {"AFIB"}
    af = db[db.codes.apply(lambda k: bool(k & af_codes))][["patient_id", "recording_date"]]
    m = pos[["ecg_id", "patient_id", "recording_date"]].merge(af, on="patient_id", suffixes=("_sr", "_af"))
    m["dt"] = (m.recording_date_af - m.recording_date_sr).dt.days
    m = m[(m.dt >= 0) & (m.dt <= config.PRED_WINDOW_DAYS)]
    return m.groupby("ecg_id").dt.min()


def main():
    db = load_db()
    summary = []
    for include_flutter, tag in ((False, "afib"), (True, "aflt")):
        pos, healthy = _afsr_frames(db, include_flutter)
        dt = _days_to_af(db, pos, include_flutter)
        rows = []
        for seed in range(N_SEEDS):
            neg = match_controls(pos, healthy, seed)
            sub = pd.concat([pos.assign(label=1), neg.assign(label=0)])   # the order _build_recs uses
            y = sub.label.to_numpy(); g = sub.patient_id.astype(str).to_numpy()
            fold = np.full(len(sub), -1)
            skf = StratifiedGroupKFold(config.CV_N_SPLITS, shuffle=True, random_state=seed)
            for k, (_, te) in enumerate(skf.split(np.zeros(len(y)), y, g)):
                fold[te] = k
            out = pd.DataFrame({
                "seed": seed, "ecg_id": sub.ecg_id.to_numpy(), "patient_id": sub.patient_id.astype(int).to_numpy(),
                "label": y, "age": sub.age.to_numpy(), "sex": sub.sex.to_numpy(),
                "recording_date": sub.recording_date.dt.strftime("%Y-%m-%d").to_numpy(),
                "days_to_af": sub.ecg_id.map(dt).to_numpy(), "fold": fold})
            rows.append(out)
            if seed == 0:
                for lab, part in out.groupby("label"):
                    summary.append({
                        "cohort": tag, "label": int(lab), "n_records": len(part),
                        "n_patients": part.patient_id.nunique(),
                        "age_median": float(part[part.age < 300].age.median()),
                        "age_q1": float(part[part.age < 300].age.quantile(0.25)),
                        "age_q3": float(part[part.age < 300].age.quantile(0.75)),
                        "female_frac": float((part.sex == 1).mean()),
                        "age_over_89_frac": float((part.age == 300).mean()),
                        "days_to_af_median": float(part.days_to_af.median()) if lab == 1 else np.nan,
                        "days_to_af_0_frac": float((part.days_to_af == 0).mean()) if lab == 1 else np.nan})
        pd.concat(rows).to_csv(config.MANIFEST_DIR / f"afsr_{tag}_draws.csv", index=False)
        print(f"{tag}: {len(pos)} positive records ({pos.patient_id.nunique()} patients), "
              f"{len(neg)} matched controls per draw from {len(healthy)} candidates")
    pd.DataFrame(summary).to_csv(config.MANIFEST_DIR / "cohort_summary.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
