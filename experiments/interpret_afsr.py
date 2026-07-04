"""Regenerate the AF-in-SR interpretability block maps with the CORRECTED config.

The old interpret_predict_blocks.* used median_beat_rr (the detection config), whose RR pseudo-leads are
dead weight in afsr. Here: morphology-only median_beat, K=8. Fits BTTR on the full cohort for the
glass-box lead figure (substrate map); negatives drawn at seed 0.

Produces BOTH cohorts so the reproduce flow is complete:
  - AFIB-only (primary)  -> interpret_afsr_afibonly_blocks.{npz,png}  (the manuscript figure; weight_fractions primary)
  - AFIB+AFLT (secondary) -> interpret_afsr_blocks.{npz,png}          (weight_fractions secondary)
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")
import numpy as np

from core import config
from experiments import interpret as interp
from experiments.harden_afsr import _afsr_frames, _build_recs, _db_with_ondisk


def main():
    config.TENSORISATION = "median_beat"
    config.BTTR_NFACTOR = 8
    db = _db_with_ondisk()
    for include_flutter, prefix in ((False, "interpret_afsr_afibonly"), (True, "interpret_afsr")):
        pos, healthy = _afsr_frames(db, include_flutter)
        recs = _build_recs(pos, healthy, seed=0)
        y = np.array([r.label for r in recs])
        cohort = "AFIB+AFLT (secondary)" if include_flutter else "AFIB-only (primary)"
        print(f"[interpret afsr] cohort={cohort} pos={int(y.sum())} neg={int((y == 0).sum())} "
              f"tensorisation={config.TENSORISATION} K={config.BTTR_NFACTOR}")
        maps, b = interp.export(recs, y, out_prefix=prefix)
        print(f"  block coefficients b = {np.round(b, 3)}")


if __name__ == "__main__":
    main()
