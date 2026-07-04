"""Driver for the BTTR-on-ECG study.

Two CLASSIFICATION tasks (PTB-XL has no outcome follow-up, so neither is future-
onset prediction):
  detect : SR vs AF (the AF ECG is in AF).
  afsr   : AF-during-sinus-rhythm identification (Attia-style 31-day window) - a
           sinus ECG from a patient with concurrent paroxysmal AF vs SR from a
           patient with no AF. ~42 positives -> a pilot, NOT prediction.

Produces:
  1. Main table   - AUROC (+ bootstrap 95% CI) per method, patient-grouped CV.
  2. Glass-box    - BTTR block weight maps (interpret.py): spatio-temporal AF
                    signatures. Selling point #1.
  3. Learning curve - AUROC vs training-set fraction: BTTR's data efficiency.

Usage:
  python run.py --task detect                       # SR vs AF (500/class, 500 Hz)
  python run.py --task detect --lowres --per-class 150   # fast smoke test (100 Hz)
  python run.py --task afsr                         # AF-in-SR pilot (~42 pos, NOT prediction)
  python run.py --task detect --learning-curve      # add the data-efficiency sweep
  python run.py --task detect --interpret           # export BTTR block maps
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from core import config
from core import data as datamod
from core import evaluate as ev
from experiments import interpret as interp
from core import preprocess as pp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["detect", "afsr"], default="detect")
    ap.add_argument("--lowres", action="store_true")
    ap.add_argument("--per-class", type=int, default=None)
    ap.add_argument("--seed", type=int, default=config.RANDOM_STATE)
    ap.add_argument("--learning-curve", action="store_true")
    ap.add_argument("--interpret", action="store_true")
    args = ap.parse_args()

    if args.task == "detect":
        recs = datamod.load_detection(per_class=args.per_class, lowres=args.lowres, seed=args.seed)
    else:
        recs = datamod.load_afsr(lowres=args.lowres)

    y = np.array([r.label for r in recs])
    groups = np.array([r.patient_id for r in recs])
    print(f"[{args.task}] n={len(recs)} pos={int(y.sum())} patients={len(set(groups))} "
          f"tensorisation={config.TENSORISATION} BTTR_K={config.BTTR_NFACTOR}")
    if args.task == "afsr":
        print("  !! AF-in-SR pilot (n_pos~42), Attia-style identification, NOT future-onset "
              "prediction (PTB-XL has no follow-up). Report CIs, not the point estimate.")

    tensor = pp.build_tensor(recs)        # build once; reused across methods/folds
    print(f"  tensor shape: {tensor.shape}  (N, time, leads)")

    # ---- 1. main table ----
    rows = []
    for method in config.METHODS:
        try:
            r = ev.evaluate_method(method, recs, y, groups, seed=args.seed, tensor=tensor)
        except Exception as e:
            print(f"  !! {method}: {e}")
            continue
        rows.append({"task": args.task, **r})
        print(f"  {method:14s} AUROC={r['auroc']:.3f} [{r['ci_lo']:.3f},{r['ci_hi']:.3f}] "
              f"AUPRC={r['auprc']:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(config.RESULTS_DIR / f"results_{args.task}.csv", index=False)

    # ---- 2. interpretability ----
    if args.interpret:
        interp.export(recs, y, out_prefix=f"interpret_{args.task}")

    # ---- 3. learning curve ----
    if args.learning_curve:
        lc = []
        for frac in config.LEARNING_CURVE_FRACS:
            for method in config.METHODS:
                try:
                    r = ev.evaluate_method(method, recs, y, groups, seed=args.seed,
                                           frac=frac, tensor=tensor)
                except Exception as e:
                    print(f"  !! lc {method}@{frac}: {e}")
                    continue
                lc.append({"task": args.task, "frac": frac, "method": method,
                           "auroc": r["auroc"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"]})
                print(f"  [lc frac={frac:.2f}] {method:14s} AUROC={r['auroc']:.3f}")
        pd.DataFrame(lc).to_csv(config.RESULTS_DIR / f"learning_curve_{args.task}.csv", index=False)

    print(f"\nWrote results to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
