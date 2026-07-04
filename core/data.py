"""PTB-XL loading and cohort construction for the two tasks.

* Detection (SR vs AF): balanced subset, PATIENT-GROUPED so the same patient never
  appears in train and test (the thesis sampled per-record).
* AF-in-SR identification (NOT prediction): a sinus-rhythm ECG is positive if the
  same patient also has an AF ECG within PRED_WINDOW_DAYS ("unhealthy" = concurrent
  paroxysmal AF caught in a sinus interval) vs SR from patients with no AF
  ("healthy"). Attia-style 31-day labelling; a CLASSIFICATION task, not future-onset
  prediction (PTB-XL has NO outcome follow-up). ~42 positives -> a pilot only.

Each sample is a `Recording`: raw (T x 12) signal, label, patient id, record id.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core import config


@dataclass
class Recording:
    signal: np.ndarray      # (n_samples, 12)
    fs: int
    label: int              # 0 / 1
    label_name: str
    patient_id: str
    record_id: str


def _rhythm_of(scp_codes: dict) -> str | None:
    codes = {k for k, v in scp_codes.items() if v >= config.MIN_SCP_LIKELIHOOD}
    if codes & config.AF_CODES:
        return "AF"
    if codes & config.SR_CODES:
        return "SR"
    return None


def _load_db() -> pd.DataFrame:
    db = pd.read_csv(config.PTBXL_DIR / "ptbxl_database.csv", index_col="ecg_id")
    db.scp_codes = db.scp_codes.apply(ast.literal_eval)
    db["rhythm"] = db.scp_codes.apply(_rhythm_of)
    db["recording_date"] = pd.to_datetime(db["recording_date"], errors="coerce")
    return db


def _read_signal(row, lowres: bool):
    import wfdb
    fname = row.filename_lr if lowres else row.filename_hr
    sig, meta = wfdb.rdsamp(str(config.PTBXL_DIR / fname))
    return sig.astype(np.float32), int(meta["fs"])


def load_detection(per_class: int | None = None, lowres: bool = False,
                   seed: int = 0) -> list[Recording]:
    per_class = per_class or config.DETECT_PER_CLASS
    db = _load_db()
    rng = np.random.default_rng(seed)

    sr = db[db.rhythm == "SR"]
    af = db[db.rhythm == "AF"]
    sr = sr[~sr.patient_id.isin(af.patient_id.unique())]            # SR patients w/o any AF
    sr = sr.groupby("patient_id", group_keys=False).sample(1, random_state=seed)
    af = af.groupby("patient_id", group_keys=False).sample(1, random_state=seed)

    n = min(per_class, len(sr), len(af))
    sr, af = sr.sample(n, random_state=seed), af.sample(n, random_state=seed)

    recs: list[Recording] = []
    for name, sub in (("SR", sr), ("AF", af)):
        for ecg_id, row in sub.iterrows():
            sig, fs = _read_signal(row, lowres)
            recs.append(Recording(sig, fs, 0 if name == "SR" else 1, name,
                                   str(row.patient_id), str(ecg_id)))
    rng.shuffle(recs)
    return recs


def load_afsr(lowres: bool = False) -> list[Recording]:
    """AF-during-sinus-rhythm IDENTIFICATION cohort (Attia-style, 31-day window).

    NOT future-onset prediction: PTB-XL has no longitudinal outcome follow-up. A
    positive is a sinus-rhythm ECG from a patient who ALSO has an AF ECG within
    PRED_WINDOW_DAYS (i.e. concurrent paroxysmal AF caught in a sinus interval);
    a negative is SR from a patient with no AF record. This is a CLASSIFICATION
    task on the AF substrate. Only ~42 positives -> a pilot. True incident-AF
    prediction needs a cohort with dated outcomes (e.g. MIMIC-IV-ECG)."""
    db = _load_db().reset_index()
    af_patients = set(db[db.rhythm == "AF"].patient_id)

    sr = db[(db.rhythm == "SR") & (db.patient_id.isin(af_patients))]
    af = db[(db.rhythm == "AF")][["patient_id", "recording_date"]]
    merged = sr.merge(af, on="patient_id", suffixes=("_sr", "_af"))
    merged["dt"] = (merged.recording_date_af - merged.recording_date_sr).dt.days
    pos_ids = set(merged[(merged.dt >= 0) & (merged.dt <= config.PRED_WINDOW_DAYS)].ecg_id)

    healthy = db[(db.rhythm == "SR") & (~db.patient_id.isin(af_patients))]
    pos = db[db.ecg_id.isin(pos_ids)]
    neg = (healthy.groupby("patient_id", group_keys=False).sample(1, random_state=0)
           .sample(min(len(pos) * 3, healthy.patient_id.nunique()), random_state=0))

    recs: list[Recording] = []
    for sub, lab, nm in ((pos, 1, "unhealthy"), (neg, 0, "healthy")):
        for _, row in sub.iterrows():
            sig, fs = _read_signal(row, lowres)
            recs.append(Recording(sig, fs, lab, nm, str(row.patient_id), str(row.ecg_id)))
    print(f"[af-in-sr] positives={len(pos)} negatives={len(neg)} -- pilot, NOT prediction")
    return recs
