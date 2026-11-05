"""Graphical abstract for the journal submission: 32.93 mm x 37.63 mm, portrait.

Three stacked elements:
  1. one sinus-rhythm ECG trace (lead II of the first AF-in-SR positive record of the primary cohort),
  2. the BTTR time-by-lead weight map of the block with the largest |b|, read from
     results/interpret_afsr_afibonly_blocks.npz,
  3. AUROC bars, BTTR vs the best other method per cohort, each the mean over the ten matched
     control draws in results/benchmark_draws_r3.csv (columns cohort, seed, method, auroc; the
     age-only score is excluded); the bars start at the chance level 0.5.

The only figures printed in the image are those ten-draw means, rounded to two decimals.

External input: the ECG trace needs PTB-XL at config.PTBXL_DIR (see code/README.md, section Data).

Writes Graphical_Abstract_Cornelissen.tif (RGB, 600 dpi) and Graphical_Abstract_Cornelissen.pdf
(vector, fonts embedded) into the figures folder of export_journal_figures.py.

Run from code/:  python -m experiments.graphical_abstract
"""
import warnings; warnings.filterwarnings("ignore")

import matplotlib; matplotlib.use("Agg")
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

from core import config
from core import preprocess as pp
from core.data import _read_signal
from experiments.export_journal_figures import FIGURES_DIR, MM, TIFF_DPI, choose_font, rc, save_tiff
from experiments.harden_afsr import _afsr_frames, load_db

NAME = "Graphical_Abstract_Cornelissen"
W_MM, H_MM = 32.93, 37.63          # journal size at 100 percent
PT = 6.0                           # smallest lettering, points at final size
BTTR_COLOR = "#1b7837"             # the BTTR colour of the forest figure
BASE_COLOR = "#6e6e6e"             # best-baseline bars and text (luminance differs from BTTR_COLOR)
INK = "#1a1a1a"
COHORTS = [("AFIB (primary)", "primary"), ("AFIB+AFLT (secondary)", "secondary")]


def headline_numbers():
    """Ten-draw mean AUROC of BTTR and of the best other method per cohort."""
    df = pd.read_csv(config.RESULTS_DIR / "benchmark_draws_r3.csv")
    out = []
    for key, _ in COHORTS:
        d = df[(df.cohort == key) & (df.method != "age")].groupby("method")["auroc"].mean()
        best = d.drop("bttr")
        out.append((float(d["bttr"]), float(best.max()), best.idxmax()))
    return out


def ecg_trace():
    """(t, x) of two beats of lead II, denoised, from the first AF-in-SR positive record."""
    if not (config.PTBXL_DIR / "ptbxl_database.csv").exists():
        raise FileNotFoundError(f"PTB-XL not found at {config.PTBXL_DIR}; fetch it as described in code/README.md (Data) or set PTBXL_DIR in core/config.py")
    pos, _ = _afsr_frames(load_db(), include_flutter=False)
    row = pos.sort_values("ecg_id").iloc[0]
    sig, fs = _read_signal(row, lowres=True)
    lead = config.LEAD_NAMES.index("II")
    x = pp.clean(sig, fs, decimate_factor=1)[:, lead]
    peaks = pp.detect_rpeaks(x, fs)
    p = peaks[(peaks > int(1.0 * fs))][:3]
    a, b = p[0] - int(0.30 * fs), p[2] - int(0.30 * fs)
    return (np.arange(a, b) - a) / fs, x[a:b]


def block_map():
    """(T, L) weight map of the block with the largest |b| and its colour limit."""
    z = np.load(config.RESULTS_DIR / "interpret_afsr_afibonly_blocks.npz")
    k = int(np.argmax(np.abs(z["b"])))
    m = z["maps"][k]
    return m, float(np.percentile(np.abs(m), 99)), z["t"]


def ax_mm(fig, x, y, w, h):
    """Axes at x, y (from the top-left corner), width w, height h, all in millimetres."""
    return fig.add_axes([x / W_MM, 1 - (y + h) / H_MM, w / W_MM, h / H_MM])


def text_mm(fig, x, y, s, **kw):
    """Text anchored at x, y (millimetres from the top-left); y is the text's vertical centre."""
    kw.setdefault("fontsize", PT)
    kw.setdefault("color", INK)
    return fig.text(x / W_MM, 1 - y / H_MM, s, ha=kw.pop("ha", "center"), va="center", **kw)


def rich_line(fig, y, parts, weight="bold"):
    """One centred line made of (string, colour) parts placed one after the other."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    items = [text_mm(fig, 0, y, s, color=c, weight=weight, ha="left") for s, c in parts]
    widths = [t.get_window_extent(rend).width / fig.dpi * MM for t in items]
    x = (W_MM - sum(widths)) / 2
    for t, w in zip(items, widths):
        t.set_x(x / W_MM)
        x += w
    return sum(widths)


def arrow(fig, x0, y0, x1, y1):
    fig.add_artist(matplotlib.patches.FancyArrowPatch(
        (x0 / W_MM, 1 - y0 / H_MM), (x1 / W_MM, 1 - y1 / H_MM), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=4.5, lw=0.6, color=BASE_COLOR, shrinkA=0, shrinkB=0))


def draw():
    nums = headline_numbers()
    t, x = ecg_trace()
    m, vmax, tm = block_map()

    # whole pixels at TIFF_DPI, rounded up so the raster is at least the journal size
    px_w, px_h = int(np.ceil(W_MM / MM * TIFF_DPI)), int(np.ceil(H_MM / MM * TIFF_DPI))
    fig = plt.figure(figsize=(px_w / TIFF_DPI, px_h / TIFF_DPI), dpi=TIFF_DPI, facecolor="white")

    # 1. ECG trace (left) and weight map (right)
    ax = ax_mm(fig, 0.9, 1.6, 15.4, 11.0)
    ax.plot(t, x, color=INK, lw=0.7, solid_joinstyle="round")
    pad = 0.08 * (x.max() - x.min())
    ax.set_xlim(t[0], t[-1]); ax.set_ylim(x.min() - pad, x.max() + pad)
    ax.axis("off")
    arrow(fig, 16.8, 7.1, 19.3, 7.1)
    ax = ax_mm(fig, 19.9, 1.6, 12.1, 11.0)
    ax.imshow(m.T, aspect="auto", cmap="PiYG", vmin=-vmax, vmax=vmax,
              extent=[tm[0], tm[-1], m.shape[1] - 0.5, -0.5], interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4); s.set_color(BASE_COLOR)
    text_mm(fig, 8.6, 14.5, "sinus ECG")
    text_mm(fig, 25.9, 14.5, "BTTR map")

    # 2. paired AUROC bars, from the chance level 0.5
    ax = ax_mm(fig, 1.2, 17.6, 30.5, 11.0)
    lo, hi = 0.5, 0.78
    bw = 0.36
    for i, (bttr, base, _) in enumerate(nums):
        for dx, v, col in ((-bw / 2 - 0.02, bttr, BTTR_COLOR), (bw / 2 + 0.02, base, BASE_COLOR)):
            ax.bar(i + dx, v - lo, width=bw, bottom=lo, color=col, edgecolor="none")
            ax.text(i + dx, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=PT, color=col, weight="bold")
    ax.axhline(lo, color=INK, lw=0.5)
    ax.set_xlim(-0.55, len(nums) - 0.45); ax.set_ylim(lo - 0.002, hi)
    ax.axis("off")
    ax.axhline(lo, color=INK, lw=0.5)
    for i, (_, name) in enumerate(COHORTS):
        text_mm(fig, 1.2 + 30.5 * (i + 0.55) / (len(nums) + 0.1), 30.2, name)

    # 3. headline
    rich_line(fig, 33.4, [("BTTR", BTTR_COLOR), (" vs best ", INK), ("baseline", BASE_COLOR)])
    rich_line(fig, 36.0, [("mean AUROC, 10 draws", INK)])
    return fig, nums


def main():
    font_name = choose_font()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(rc(font_name, PT)):
        fig, nums = draw()
        fig.savefig(FIGURES_DIR / f"{NAME}.pdf", format="pdf", facecolor="white")
        w, h = save_tiff(fig, FIGURES_DIR / f"{NAME}.tif")
    plt.close(fig)
    print(f"font: {font_name}")
    for (key, _), (bttr, base, who) in zip(COHORTS, nums):
        print(f"{key}: ten-draw mean AUROC BTTR {bttr:.3f}, best baseline {base:.3f} ({who})")
    print(f"wrote {NAME}.tif ({w} x {h} px = {w / TIFF_DPI * MM:.2f} x {h / TIFF_DPI * MM:.2f} mm at {TIFF_DPI} dpi) and {NAME}.pdf")


if __name__ == "__main__":
    main()
