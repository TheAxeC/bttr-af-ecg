"""Central configuration for the BTTR-on-ECG study.

This study applies REAL Block-Term Tensor Regression (Faes et al.) - the vendored
`bttrlib` package - to two ECG CLASSIFICATION tasks: SR-vs-AF detection, and
AF-during-sinus-rhythm identification (Attia-style 31-day window). Neither is
future-onset prediction: PTB-XL has no outcome follow-up (for that, use a dated-
outcome cohort such as MIMIC-IV-ECG). BTTR is supervised, deflation/ACE-based and
INDUCTIVE: trained on the training fold, applied to unseen ECGs directly.

The thesis substituted an unsupervised CP-features surrogate because it assumed no
BTD implementation existed. It does (bttrlib). The CP surrogate is kept only as a
baseline (`baselines.py`) to show what that substitution cost.

Two selling points drive the experiments (see paper plan):
  (1) interpretability - each BTTR block is a (time x lead) weight map: a glass-box
      AF signature (interpret.py);
  (2) data efficiency - BTTR holds up with few labels (learning_curve in run.py),
      which is exactly the regime of the AF-in-SR pilot (~42 positives).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths - CHANGE THESE                                                          #
# --------------------------------------------------------------------------- #
PTBXL_DIR = Path("~/data/ptb-xl").expanduser()   # has ptbxl_database.csv, records500/, records100/
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"   # the package results/ dir (regenerable; deposit-excluded)
MANIFEST_DIR = Path(__file__).resolve().parents[1] / "manifests"   # record-level cohort and split manifests (deposited)

# --------------------------------------------------------------------------- #
# Reproducibility                                                             #
# --------------------------------------------------------------------------- #
SEEDS = list(range(5))
RANDOM_STATE = 0

# --------------------------------------------------------------------------- #
# PTB-XL signal / label settings                                              #
# --------------------------------------------------------------------------- #
PTBXL_FS = 500
LOWRES_FS = 100
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]
RHYTHM_LEAD = "II"

SR_CODES = {"SR"}
AF_CODES = {"AFIB"}          # add "AFLT" to include flutter
MIN_SCP_LIKELIHOOD = 0.0
PRED_WINDOW_DAYS = 31        # SR<->AF same-patient window for the AF-in-SR cohort (Attia-style)
DETECT_PER_CLASS = 500

# --------------------------------------------------------------------------- #
# Preprocessing (kept close to the thesis helpers.py, but no aggressive decimation:
# the thesis x20 -> 25 Hz erases P/f-waves, the AF signature. We keep 250 Hz.)     #
# --------------------------------------------------------------------------- #
DECIMATE_FACTOR = 5          # 500 -> 100 Hz; keeps P/f-wave band, shrinks the time mode
WAVELET = "sym4"
WAVELET_LEVEL = 3
WAVELET_THRESHOLD = 7.0

# --------------------------------------------------------------------------- #
# BTTR settings                                                                #
# --------------------------------------------------------------------------- #
BTTR_NFACTOR = 8             # number of block terms K; main interpretable knob
BTTR_ENFORCE_RANK1 = True    # rank-(L2,...,LN,1): the tensor-vector case for vector y
BTTR_SNRS = range(20, 41, 5)
BTTR_RATIOS = (95.0, 97.0, 99.0)

# --------------------------------------------------------------------------- #
# Tensorisation: how the ECG becomes a (sample x time x lead) tensor.          #
#   'window' : full denoised 10 s strip (time x lead) - the thesis layout.     #
#   'median_beat' : R-peak-aligned median beat (time x lead) - alignment-aware. #
#   'median_beat_rr' : aligned beat + RR-tachogram pseudo-leads - rhythm-aware. #
#       Winning tensorisation for DETECTION (BTTR 0.86 vs 0.79 plain).          #
# --------------------------------------------------------------------------- #
TENSORISATION = "median_beat_rr"

# --------------------------------------------------------------------------- #
# Experiment matrix: BTTR is the method; the rest are baselines/controls.      #
# --------------------------------------------------------------------------- #
METHODS = [
    "bttr",            # the method (interpretable, inductive)
    "clinical_lr",     # clinical RR/P-wave features + logistic regression (the yardstick)
    "cp_surrogate",    # what the thesis actually did: CP factors + logistic regression
]

CV_N_SPLITS = 5
BOOTSTRAP_N = 2000
LEARNING_CURVE_FRACS = (0.1, 0.2, 0.4, 0.7, 1.0)   # data-efficiency sweep


@dataclass
class RunResult:
    task: str
    method: str
    metrics: dict = field(default_factory=dict)


RESULTS_DIR.mkdir(parents=True, exist_ok=True)
