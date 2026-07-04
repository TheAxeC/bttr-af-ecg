"""Temporal weight-fraction breakdown of the BTTR block maps (reproducible).

The interpretability paragraph reports what fraction of the model's discriminative
weight falls in each part of the cardiac cycle. That split was previously computed
by hand from the saved block-map npz with unpinned window boundaries; this script
pins the boundaries to standard ECG-interval definitions (relative to the R-peak at
t = 0, on the -250..+400 ms median-beat window) and regenerates the numbers so they
are byte-reproducible.

Importance at time t = sum_{blocks k, leads l} |map[k,t,l]| * |b[k]|  (same weighting
as the map-stability metric). Window fraction = importance summed over that window /
total importance.

Segment windows (R-peak at 0 ms; standard adult ECG intervals):
  atrial (P-wave + PR segment) : [-250, -50) ms   (all pre-QRS activity)
  QRS complex                  : [ -50, +50) ms   (~100 ms around R)
  ST segment                   : [ +50, +150) ms  (J-point to T onset)
  T-wave                       : [+150, +400] ms  (repolarisation)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from core import config

SEGMENTS = [
    ("atrial (P+PR)", -0.250, -0.050),
    ("QRS",           -0.050,  0.050),
    ("ST",             0.050,  0.150),
    ("T-wave",         0.150,  0.400),
]

COHORTS = [
    ("primary (AFIB-only)",       "interpret_afsr_afibonly_blocks.npz"),
    ("secondary (AFIB+AFLT)",     "interpret_afsr_blocks.npz"),
]


def fractions(npz_path):
    d = np.load(npz_path)
    maps, b, t = d["maps"], d["b"], d["t"]          # (K,T,L), (K,), (T,)
    imp_t = np.sum(np.abs(maps) * np.abs(b)[:, None, None], axis=(0, 2))  # (T,) per-time importance
    total = imp_t.sum() + 1e-12
    out = {}
    for name, lo, hi in SEGMENTS:
        m = (t >= lo) & (t < hi) if hi < t[-1] else (t >= lo) & (t <= hi)
        out[name] = float(imp_t[m].sum() / total)
    return out


def main():
    rows = []
    for label, fname in COHORTS:
        fr = fractions(config.RESULTS_DIR / fname)
        s = "  ".join(f"{k}={v:.3f}" for k, v in fr.items())
        print(f"{label:24s}  {s}   (sum={sum(fr.values()):.3f})")
        rows.append({"cohort": label, **{k: round(v, 3) for k, v in fr.items()}})
    out = config.RESULTS_DIR / "weight_fractions.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
