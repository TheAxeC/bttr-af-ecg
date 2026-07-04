"""Signal preprocessing and tensorisation.

Denoising mirrors the thesis helpers.py (wavelet soft-threshold + baseline-drift +
notch + per-lead amplitude normalisation) so BTTR is fed comparable inputs. Two
tensorisations are offered (config.TENSORISATION):

  'window'      : the full denoised 10 s strip -> (T, 12). The thesis layout. The
                  time mode is NOT aligned across recordings (different beat phase),
                  which is the structural weakness; kept for the ablation.
  'median_beat' : R-peaks detected on the rhythm lead, the median beat per recording
                  -> (T_beat, 12). Every time index now means the same cardiac-cycle
                  phase across recordings, so BTTR's per-block temporal factor is a
                  clinically-readable beat template (better interpretability + signal).
"""
from __future__ import annotations

import numpy as np
import pywt
from scipy.signal import butter, decimate, filtfilt, find_peaks, iirnotch

from core import config


def _baseline(sig, fs, cutoff=0.5, order=2):
    b, a = butter(order, cutoff / (0.5 * fs), btype="high")
    return filtfilt(b, a, sig)


def _notch(sig, fs, freq=50.0, q=30.0):
    b, a = iirnotch(w0=freq, Q=q, fs=fs)
    return filtfilt(b, a, sig)


def _denoise_1d(sig, fs):
    sig = _notch(_baseline(sig, fs), fs)
    coeffs = pywt.wavedec(sig, config.WAVELET, level=config.WAVELET_LEVEL)
    t = np.std(coeffs[-1]) * config.WAVELET_THRESHOLD
    coeffs = [c if i == 0 else pywt.threshold(c, t, mode="soft")
              for i, c in enumerate(coeffs)]
    return pywt.waverec(coeffs, config.WAVELET)[: len(sig)]


def clean(signal, fs, decimate_factor=None, normalize=True):
    """(T, L) raw -> (T', L) denoised, drift-free, per-lead max-abs normalised."""
    df = config.DECIMATE_FACTOR if decimate_factor is None else decimate_factor
    out = []
    for l in range(signal.shape[1]):
        x = _denoise_1d(signal[:, l], fs)
        if df and df > 1:
            x = decimate(x, df, zero_phase=True)
        if normalize:
            m = np.max(np.abs(x - x.mean()))
            x = (x - x.mean()) / m if m > 0 else x * 0.0
        out.append(x)
    n = min(map(len, out))
    return np.stack([o[:n] for o in out], axis=1)


def detect_rpeaks(sig, fs):
    """R-peak indices via a light Pan-Tompkins-style detector on one lead."""
    x = np.asarray(sig, dtype=float)
    e = np.diff(x, prepend=x[0]) ** 2
    w = max(1, int(0.150 * fs))
    integ = np.convolve(e, np.ones(w) / w, mode="same")
    thr = integ.mean() + 0.5 * integ.std()
    peaks, _ = find_peaks(integ, height=thr, distance=int(0.25 * fs))
    return peaks


def _median_beat(sig12, fs, win_ms=(250, 400)):
    lead = config.LEAD_NAMES.index(config.RHYTHM_LEAD)
    a, b = int(win_ms[0] / 1000 * fs), int(win_ms[1] / 1000 * fs)
    peaks = detect_rpeaks(sig12[:, lead], fs)
    beats = [sig12[p - a:p + b] for p in peaks if p - a >= 0 and p + b <= len(sig12)]
    if not beats:
        return np.zeros((a + b, sig12.shape[1]), dtype=sig12.dtype)
    return np.median(np.stack(beats, 0), axis=0)


def _rr_pseudo_leads(full, fs, T):
    """Rhythm pseudo-leads aligned to the beat-time axis: RR tachogram (detrended,
    resampled to T), constant rr_cv, constant pNN50. Injecting these makes a pure
    BTTR tensor rhythm-aware (median beat alone discards RR irregularity)."""
    lead = config.LEAD_NAMES.index(config.RHYTHM_LEAD)
    peaks = detect_rpeaks(full[:, lead], fs)
    if len(peaks) >= 3:
        rr = np.diff(peaks) / fs
        drr = np.diff(rr)
        rrcv = rr.std() / (rr.mean() + 1e-9)
        pnn = float(np.mean(np.abs(drr) > 0.05))
        tacho = rr - rr.mean()
        tacho = np.interp(np.linspace(0, 1, T), np.linspace(0, 1, len(tacho)), tacho)
    else:
        rrcv = pnn = 0.0
        tacho = np.zeros(T)
    return np.column_stack([tacho, np.full(T, rrcv), np.full(T, pnn)])   # (T, 3)


RR_LEAD_NAMES = ["RRtacho", "RRcv", "pNN50"]


def build_tensor(recordings, mode=None, decimate_factor=None):
    """List[Recording] -> (N, T, L) tensor (common T), per config.TENSORISATION.

    Modes: 'window' (unaligned strip), 'median_beat' (R-peak-aligned beat),
    'median_beat_rr' (aligned beat + RR-tachogram pseudo-leads -> rhythm-aware)."""
    mode = mode or config.TENSORISATION
    mats = []
    for r in recordings:
        if mode in ("median_beat", "median_beat_rr"):
            full = clean(r.signal, r.fs, decimate_factor=1)   # full-fs for accurate alignment
            mb = _median_beat(full, r.fs)
            if mode == "median_beat_rr":
                mb = np.concatenate([mb, _rr_pseudo_leads(full, r.fs, mb.shape[0])], axis=1)
            mats.append(mb)
        else:
            mats.append(clean(r.signal, r.fs, decimate_factor=decimate_factor))
    T = min(m.shape[0] for m in mats)
    return np.stack([m[:T] for m in mats], axis=0).astype(np.float64)


def lead_names(mode=None):
    mode = mode or config.TENSORISATION
    return config.LEAD_NAMES + (RR_LEAD_NAMES if mode == "median_beat_rr" else [])


def time_axis_seconds(T, mode=None):
    """Helper for interpretability plots: seconds per time index."""
    mode = mode or config.TENSORISATION
    if mode == "median_beat":
        return np.linspace(-0.25, 0.40, T)        # beat window
    fs = config.PTBXL_FS / max(1, config.DECIMATE_FACTOR)
    return np.arange(T) / fs
